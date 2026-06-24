"""
Fixed-base grasp environment — the Robotiq-gripper rebuild.

The robot stands fixed (base + legs + torso + left arm held each step, no recoil, no AMO).
RL controls the 7 right-arm joints + the gripper to grasp the box off the pedestal and lift it.

Reward is deliberately HARD TO HACK: the lift/hold bonuses are GATED on a real two-sided grip
(the tool must be touching BOTH the left and right gripper pads). You cannot farm reward by
hovering or by knocking the tool upward — only an actual pinch-then-lift pays off.

Model: g1_robotiq.mjb (built by build_grasp_model.py). gymnasium API, SB3-compatible.
"""
from __future__ import annotations
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))


class GraspFixedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 5, max_steps: int = 200, reward_w: dict | None = None,
                 robust: bool = False, model_path: str = "g1_robotiq.mjb"):
        super().__init__()
        self.model = mujoco.MjModel.from_binary_path(os.path.join(HERE, model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        m = self.model

        # reward weights (tunable; lift/hold are gated on a real grip so they can't be farmed)
        # gr-v1: saturate lift (no reward for flinging high), make sustained hold dominate,
        # anti-scoop grip-continuity gate, penalize ballistic tool vertical velocity post-grip.
        self.w = {"reach": 0.3, "grip": 2.0, "lift": 6.0, "hold": 4.0, "knock": 3.0,
                  "vel": 3.0, "stab": 1.5, "ctrl": 1e-3}
        if reward_w:
            self.w.update(reward_w)
        self.LIFT_CAP = 0.10        # lift reward plateaus at 10cm -> flinging higher pays nothing
        self.GRIP_K = 5             # continuous grip steps before lift/hold pay (kills flicker-scoop)
        self.VEL_OK = 0.30          # tool vertical speed (m/s) tolerated before the ballistic penalty bites
        self.STAB_OK = 0.004        # per-step tool-vs-grip slip (m) allowed before the stability penalty bites

        def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

        # ---- standing pose (from the original model's home keyframe), stance stepped left 0.15 ----
        m0 = mujoco.MjModel.from_xml_path(os.path.join(HERE, "g1.xml")); d0 = mujoco.MjData(m0)
        mujoco.mj_resetDataKeyframe(m0, d0, 0); mujoco.mj_forward(m0, d0)
        self._stand = {}
        for i in range(m0.njnt):
            nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, i)
            if nm and m0.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self._stand[nm] = float(d0.qpos[m0.jnt_qposadr[i]])
        pid0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        bq = d0.qpos[m0.jnt_qposadr[[i for i in range(m0.njnt)
              if m0.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m0.jnt_bodyid[i] == pid0][0]]:][:7].copy()
        bq[0] -= 0.15
        self._base_q = bq

        # ---- right arm (7-DOF): joint addresses, ranges, and how each is actuated ----
        self.arm_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                           "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
        self.arm_qadr = np.array([m.jnt_qposadr[jid(j)] for j in self.arm_joints])
        self.arm_dadr = np.array([m.jnt_dofadr[jid(j)] for j in self.arm_joints])
        self.arm_range = np.array([m.jnt_range[jid(j)] for j in self.arm_joints])
        self.arm_mid = self.arm_range.mean(axis=1)
        self.arm_half = (self.arm_range[:, 1] - self.arm_range[:, 0]) / 2
        # actuator per arm joint: (actuator_id, is_position)
        self.arm_act = []
        for j in self.arm_joints:
            jj = jid(j)
            a = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == jj)
            is_pos = m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            self.arm_act.append((a, is_pos))
        self.kp, self.kd = 120.0, 12.0  # PD gains for the torque-actuated shoulder/elbow (gravity-compensated)

        # gripper
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg_fingers_actuator")
        self.grip_drv = m.jnt_qposadr[jid("rg_right_driver_joint")]  # 0 open .. ~0.8 closed
        # grasp point = midpoint of the two pads; tool + pad geoms for contact-based grip detection
        self.lpad, self.rpad = bid("rg_left_pad"), bid("rg_right_pad")
        self.tool_bid = bid("task_tool")
        self.tool_qadr = m.jnt_qposadr[[i for i in range(m.njnt) if m.jnt_bodyid[i] == self.tool_bid][0]]
        self.tool_geom = gid("task_tool_geom")
        self.ped_geom = gid("task_pedestal_geom")
        if self.ped_geom < 0:                       # bench variant: the support surface is the bench top
            self.ped_geom = gid("task_bench_top")
        self.tool_half_z = float(m.geom_size[self.tool_geom][2])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.left_pad_geoms = {g for g, n in enumerate(names) if "left_pad" in n}
        self.right_pad_geoms = {g for g, n in enumerate(names) if "right_pad" in n}

        # ---- stiffen contacts so the strong (kp=300) arm can't bulldoze through the pedestal ----
        # default solref [0.02,1] let the hand penetrate ~7cm; stiffen pedestal + tool + gripper geoms.
        grip_collide = [g for g in range(m.ngeom)
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith("rg_")
                        and m.geom_contype[g]]
        for g in [self.ped_geom, self.tool_geom, *grip_collide]:
            m.geom_solref[g] = [0.004, 1.0]                                 # stiffer (smaller time const)
            m.geom_solimp[g] = [0.98, 0.99, 0.0005, 0.5, 2.0]              # high constraint impedance
        m.opt.iterations = max(int(m.opt.iterations), 150)                  # resolve stiff contacts
        m.opt.ls_iterations = max(int(m.opt.ls_iterations), 50)

        # ---- robustness mode: heavier tool + realistic friction (the honest test) ----
        # default (light 50g tool, friction 3.0) flatters the grip; robust=True makes it earn it.
        if robust:
            ratio = 0.25 / max(m.body_mass[self.tool_bid], 1e-6)            # -> 250g tool
            m.body_mass[self.tool_bid] = 0.25
            m.body_inertia[self.tool_bid] *= ratio                          # shape fixed -> inertia scales w/ mass
            real_fric = np.array([1.0, 0.05, 0.005])                        # rubber-on-plastic, not sticky 3.0
            # MuJoCo combines pair friction elementwise-max, so lower BOTH tool and pads
            for g in (self.tool_geom, *self.left_pad_geoms, *self.right_pad_geoms):
                m.geom_friction[g] = real_fric

        # ---- which DOF are "free" (driven): arm + gripper linkage + tool; everything else is held ----
        free_q, free_d = set(self.arm_qadr.tolist()), set(self.arm_dadr.tolist())
        for i in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            if nm.startswith("rg_"):
                free_q.add(m.jnt_qposadr[i]); free_d.add(m.jnt_dofadr[i])
        free_q |= set(range(self.tool_qadr, self.tool_qadr + 7))
        tdof = m.jnt_dofadr[[i for i in range(m.njnt) if m.jnt_bodyid[i] == self.tool_bid][0]]
        self.tool_dadr = tdof       # tool free-joint dof start; qvel[tool_dadr+2] = world vertical vel
        free_d |= set(range(tdof, tdof + 6))
        self.held_q = np.array([i for i in range(m.nq) if i not in free_q])
        self.held_d = np.array([i for i in range(m.nv) if i not in free_d])

        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)         # 7 arm targets + gripper
        obs_dim = 7 + 7 + 1 + 3 + 1                                          # q, qvel, grip, tool-rel, lift
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._qhome = None
        self._rest_z = 0.0
        self._tool_xy0 = np.zeros(2)
        self._held_since = 0
        self._grip_streak = 0
        self._prev_rel = np.zeros(3)
        self._t = 0

    # ----- helpers -----
    def _grasp_point(self):
        return 0.5 * (self.data.xpos[self.lpad] + self.data.xpos[self.rpad])

    def _grasped(self):
        """True iff the tool is pinched — touching BOTH a left pad and a right pad."""
        L = R = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = {c.geom1, c.geom2}
            if self.tool_geom in g:
                other = (g - {self.tool_geom}).pop()
                if other in self.left_pad_geoms: L = True
                elif other in self.right_pad_geoms: R = True
        return L and R

    def _apply_pose(self):
        d, m = self.data, self.model
        pjm = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
               and m.jnt_bodyid[i] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][0]
        d.qpos[m.jnt_qposadr[pjm]:m.jnt_qposadr[pjm] + 7] = self._base_q
        for nm, v in self._stand.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = v

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        self._apply_pose()
        # right arm starts at its standing (arm-down) pose + small noise; gripper open
        for k, jadr in enumerate(self.arm_qadr):
            base = self._stand.get(self.arm_joints[k], 0.0)
            d.qpos[jadr] = np.clip(base + self.np_random.uniform(-0.05, 0.05),
                                   self.arm_range[k, 0], self.arm_range[k, 1])
        # tool RESTING FLUSH on the pedestal top (not dropped from above) + small XY jitter
        mujoco.mj_forward(m, d)                                             # so pedestal world pos is valid
        ped_top = float(d.geom_xpos[self.ped_geom][2] + m.geom_size[self.ped_geom][2])
        jit = self.np_random.uniform(-0.01, 0.01, 2)
        d.qpos[self.tool_qadr:self.tool_qadr + 3] = [-0.415 + jit[0], -1.117 + jit[1],
                                                     ped_top + self.tool_half_z + 0.0005]
        d.qpos[self.tool_qadr + 3:self.tool_qadr + 7] = [1, 0, 0, 0]
        d.qvel[self.tool_dadr:self.tool_dadr + 6] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        self._qhome = d.qpos.copy()
        # settle the tool to REST on the pedestal (everything else frozen); stop when nearly still
        for _ in range(60):
            mujoco.mj_step(m, d)
            d.qpos[self.held_q] = self._qhome[self.held_q]; d.qvel[self.held_d] = 0
            d.qpos[self.arm_qadr] = self._qhome[self.arm_qadr]; d.qvel[self.arm_dadr] = 0
            if np.linalg.norm(d.qvel[self.tool_dadr:self.tool_dadr + 3]) < 1e-3:
                break
        self._rest_z = float(d.xpos[self.tool_bid][2])
        self._tool_xy0 = d.xpos[self.tool_bid][:2].copy()
        self._held_since = 0
        self._grip_streak = 0
        self._prev_rel = (d.xpos[self.tool_bid] - self._grasp_point()).copy()
        self._t = 0
        return self._obs(), {}

    def _obs(self):
        d = self.data
        q = (d.qpos[self.arm_qadr] - self.arm_mid) / self.arm_half          # arm angles, normalized
        qv = d.qvel[self.arm_dadr] / 10.0
        grip = np.array([d.qpos[self.grip_drv] / 0.8])
        tool_rel = d.xpos[self.tool_bid] - self._grasp_point()              # tool relative to grasp point
        lift = np.array([d.xpos[self.tool_bid][2] - self._rest_z])
        return np.concatenate([q, qv, grip, tool_rel, lift]).astype(np.float32)

    def step(self, action):
        d, m = self.data, self.model
        a = np.clip(action, -1, 1)
        targets = self.arm_mid + a[:7] * self.arm_half                      # arm joint targets
        grip_ctrl = (a[7] * 0.5 + 0.5) * 255.0                              # 0 open .. 255 closed
        for _ in range(self.frame_skip):
            for k, (act, is_pos) in enumerate(self.arm_act):
                if is_pos:
                    d.ctrl[act] = targets[k]
                else:                                                       # gravity-compensated PD on shoulder/elbow motors
                    tau = (self.kp * (targets[k] - d.qpos[self.arm_qadr[k]]) - self.kd * d.qvel[self.arm_dadr[k]]
                           + d.qfrc_bias[self.arm_dadr[k]])                  # feedforward gravity/coriolis -> no droop
                    lo, hi = m.actuator_forcerange[act] if m.actuator_forcelimited[act] else (-150, 150)
                    d.ctrl[act] = float(np.clip(tau, lo, hi))
            d.ctrl[self.grip_act] = grip_ctrl
            mujoco.mj_step(m, d)
            d.qpos[self.held_q] = self._qhome[self.held_q]; d.qvel[self.held_d] = 0   # hold the rest of the body

        self._t += 1
        obs = self._obs()
        tool_z = float(d.xpos[self.tool_bid][2])
        lift = max(0.0, tool_z - self._rest_z)
        grasped = self._grasped()
        reach = float(np.linalg.norm(d.xpos[self.tool_bid] - self._grasp_point()))
        lateral = float(np.linalg.norm(d.xpos[self.tool_bid][:2] - self._tool_xy0))
        knocked = (tool_z < self._rest_z - 0.05) or (lateral > 0.15 and not grasped)

        r = -self.w["reach"] * reach
        rel = d.xpos[self.tool_bid] - self._grasp_point()                   # tool offset from the pad midpoint
        if grasped:
            self._grip_streak += 1
            r += self.w["grip"]
            if self._grip_streak >= self.GRIP_K:                            # anti-scoop: stable grip required
                r += self.w["lift"] * min(lift, self.LIFT_CAP)              # saturated lift (no reward past 10cm)
                if lift > 0.05:
                    r += self.w["hold"]                                     # dominant per-step hold bonus
                    self._held_since += 1
                else:
                    self._held_since = 0
                vz = abs(float(d.qvel[self.tool_dadr + 2]))                 # ballistic-motion penalty post-grip
                r -= self.w["vel"] * max(0.0, vz - self.VEL_OK)
                slip = float(np.linalg.norm(rel - self._prev_rel))          # tool sliding WITHIN the grip
                r -= self.w["stab"] * max(0.0, slip - self.STAB_OK)         # penalize SLIP only; allow lift compliance
            else:
                self._held_since = 0
        else:
            self._grip_streak = 0
            self._held_since = 0
        self._prev_rel = rel.copy()
        if knocked:
            r -= self.w["knock"]
        r -= self.w["ctrl"] * float(np.sum(a ** 2))

        success = self._held_since >= 25                                    # gripped + lifted, held ~0.5s
        terminated = bool(success or knocked)
        truncated = self._t >= self.max_steps
        info = {"grasped": grasped, "lift": lift, "success": success, "knocked": knocked}
        return obs, float(r), terminated, truncated, info
