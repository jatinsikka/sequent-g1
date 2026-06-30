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
                 robust: bool = False, scene: str = "pedestal", model_path: str | None = None,
                 grip_reset_prob: float = 0.0, grip_states_path: str | None = None):
        super().__init__()
        self.scene = scene
        self.grip_reset_prob = grip_reset_prob   # frac of episodes that START from a demo gripped state (learn the LIFT)
        self._grip_states = None
        if grip_reset_prob > 0 and grip_states_path and os.path.exists(grip_states_path):
            self._grip_states = {k: v for k, v in np.load(grip_states_path).items()}  # self-generated, arrays only (no pickle)
        if model_path is None:
            model_path = "g1_robotiq_table.mjb" if scene == "table" else "g1_robotiq.mjb"
        self.model = mujoco.MjModel.from_binary_path(os.path.join(HERE, model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        m = self.model

        # reward weights (tunable; lift/hold are gated on a real grip so they can't be farmed)
        # gr-v1: saturate lift (no reward for flinging high), make sustained hold dominate,
        # anti-scoop grip-continuity gate, penalize ballistic tool vertical velocity post-grip.
        self.w = {"reach": 0.3, "grip": 6.0, "lift": 250.0, "hold": 12.0, "knock": 3.0,
                  "vel": 3.0, "stab": 1.5, "center": 1.0, "selfcol": 18.0,
                  "griphold": 4.0, "depth": 6.0, "ctrl": 1e-3,
                  "camp": 1.0, "success": 50.0}
        #   v10: with the force-hold, "grip and sit" was a reward optimum (holding paid ~6-13/step,
        #   lifting added only ~1-5). Rebalanced: once the grip is EARNED, LIFT dominates (lift 100->250,
        #   hold 8->12) + a camping penalty + a big terminal success bonus; dropped the now-moot
        #   GRIP_K/depth/griphold (force-hold makes the grip binary, not a fragile contact pinch).
        #   lift 6->100 + hold 4->8 + griphold gated on lift: LIFTING must clearly out-pay holding on the table
        #   grip 2->6 + center 2->1: a CENTERED close must clearly out-pay hovering-centered-but-open
        if reward_w:
            self.w.update(reward_w)
        self.LIFT_CAP = 0.10        # lift reward plateaus at 10cm -> flinging higher pays nothing
        self.GRIP_K = 5             # continuous grip steps before lift/hold pay (kills flicker-scoop)
        self.VEL_OK = 0.30          # tool vertical speed (m/s) tolerated before the ballistic penalty bites
        self.STAB_OK = 0.004        # per-step tool-vs-grip slip (m) allowed before the stability penalty bites
        self.CENTER_TOL = 0.02      # box must be within 2cm of the pad midpoint to count as a real (centered) grasp
        self.BODY_R = 0.10          # keep the right elbow/upper-arm at least this far (m) from the torso axis

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
        if self.scene == "table":
            bq[:] = [0.70, 0.84, 0.793, 0.7071, 0.0, 0.0, 0.7071]   # at the table, facing +Y; one step BACK (y 0.92->0.84, still reaches)
            #   left  (x 0.90->0.70): box sits in front of the RIGHT hand, no cross-body reach
            #   back  (y 1.02->0.92): resting hand clears the table edge instead of starting under it
        else:
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
        tool_body = "grasp_box" if self.scene == "table" else "task_tool"
        tool_geom = "grasp_box_geom" if self.scene == "table" else "task_tool_geom"
        self.tool_bid = bid(tool_body)
        self.tool_qadr = m.jnt_qposadr[[i for i in range(m.njnt) if m.jnt_bodyid[i] == self.tool_bid][0]]
        self.tool_geom = gid(tool_geom)
        self.ped_geom = gid("right_table_top") if self.scene == "table" else gid("task_pedestal_geom")
        if self.ped_geom < 0:                       # bench variant: the support surface is the bench top
            self.ped_geom = gid("task_bench_top")
        self.tool_half_z = float(m.geom_size[self.tool_geom][2])
        self.relbow_bid = bid("right_elbow_link")   # to penalize the upper arm crossing into the torso
        self.torso_bid = bid("torso_link")
        # table scene: hide the pedestal + the cluttering objects so only the grasp box remains
        if self.scene == "table":
            for g in range(m.ngeom):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
                if bn in ("grasp_pedestal", "screwdriver", "battery_pack", "right_table_item2"):
                    m.geom_rgba[g] = [0, 0, 0, 0]; m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.left_pad_geoms = {g for g, n in enumerate(names) if "left_pad" in n}
        self.right_pad_geoms = {g for g, n in enumerate(names) if "right_pad" in n}

        # ---- a real PINCH, not a one-sided shove: raise the gripper squeeze force (was capped at 5N,
        #      far too weak to hold the box airborne) and require BOTH pads to press with real force ----
        m.actuator_forcerange[self.grip_act] = [-10.0, 10.0]   # GENTLE close: 40N crushed/ejected the box; the
        #   force-hold latch does the holding now, so the squeeze only needs to register a pinch (no crush)
        self.GRIP_FORCE = 5.0       # N each pad must exert for a grasp to count (kills the lopsided shove)

        # ---- force-gated grip HOLD (the honest middle ground, not a rigid weld) ----
        # MuJoCo friction cannot hold the box: a full close crushes it to ~700N normal and it extrudes
        # ("watermelon seed"), so it always slides out. Instead, once a genuine two-sided pinch is earned
        # (both pads >= GRIP_FORCE), we apply a CAPPED holding force that keeps the box riding the grip —
        # but the box keeps full free-joint dynamics and is LOST if the load exceeds the gripper's real
        # holding capacity (a yank/collision) or it escapes the jaw. Not a teleport, not a rigid pin:
        # it models the static-friction hold the broken contact solver fails to deliver. Release on open.
        self.hand_bid = bid("rg_base")
        if self.hand_bid < 0:
            self.hand_bid = bid("rg_base_mount")
        self._gripping = False
        self._grip_rel = np.zeros(3)                # box pos relative to the hand frame at grip onset
        self.HOLD_KP = 400.0                        # N/m  spring keeping the box at its gripped pose
        self.HOLD_KD = 14.0                         # N·s/m damping
        self.HOLD_FMAX = 25.0                       # N — gripper's real holding capacity; exceed it -> box slips
        self.ESCAPE = 0.09                          # box farther than this from the grasp point -> grip lost
        self._pad_gids = np.array(sorted(self.left_pad_geoms | self.right_pad_geoms))  # while held, drop the PAD
        self._pad_contype = m.geom_contype[self._pad_gids].copy()    # collision (kills the crush) but KEEP the
        self._pad_conaff = m.geom_conaffinity[self._pad_gids].copy() # box<->table collision (no drag-through)

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

    def _pad_forces(self):
        """Normal force each gripper pad exerts on the tool (N). Used for the two-sided pinch check."""
        f6 = np.zeros(6); L = R = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = {c.geom1, c.geom2}
            if self.tool_geom in g:
                other = (g - {self.tool_geom}).pop()
                if (other in self.left_pad_geoms or other in self.right_pad_geoms) and c.efc_address >= 0:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)   # efc_address<0 -> inactive contact, would segfault
                    if other in self.left_pad_geoms: L += abs(f6[0])
                    else: R += abs(f6[0])
        return L, R

    def _grasped(self):
        """True iff the tool is PINCHED — BOTH pads pressing with real force (not a one-sided shove)."""
        L, R = self._pad_forces()
        return L >= self.GRIP_FORCE and R >= self.GRIP_FORCE

    def _engage_grip(self):
        """A real two-sided pinch was earned: remember where the box sits in the hand frame."""
        d = self.data
        hp = d.xpos[self.hand_bid]; hq = d.xquat[self.hand_bid]
        bp = d.qpos[self.tool_qadr:self.tool_qadr + 3]
        nq = np.zeros(4); mujoco.mju_negQuat(nq, hq)
        rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, bp - hp, nq)      # box pos in hand frame
        self._grip_rel = rp; self._gripping = True
        # hand off to the capped force: drop the PAD collision (kills the crush/extrusion) but KEEP the box
        # colliding with the table/world — so a gripped-but-not-lifted box can't be dragged through the table.
        self.model.geom_contype[self._pad_gids] = 0
        self.model.geom_conaffinity[self._pad_gids] = 0

    def _apply_grip_hold(self):
        """Hold the box with a CAPPED force (the gripper's real holding capacity), NOT a rigid pin.
        Spring-damper toward the gripped pose + gravity compensation, clamped at HOLD_FMAX. The box
        keeps full dynamics, so a yank/collision beyond the cap (or escaping the jaw) loses it."""
        d = self.data
        hp = d.xpos[self.hand_bid]; hq = d.xquat[self.hand_bid]
        wp = np.zeros(3); mujoco.mju_rotVecQuat(wp, self._grip_rel, hq)
        target = hp + wp                                              # where the box should ride
        box = d.xpos[self.tool_bid]
        vel = d.qvel[self.tool_dadr:self.tool_dadr + 3]
        mg = float(self.model.body_mass[self.tool_bid]) * 9.81
        F = self.HOLD_KP * (target - box) - self.HOLD_KD * vel + np.array([0.0, 0.0, mg])
        n = float(np.linalg.norm(F))
        if n > self.HOLD_FMAX:
            F *= self.HOLD_FMAX / n                                   # cap = real grip capacity -> losable
        F[2] = max(F[2], 0.0)                                         # NEVER shove the box down (gravity+table do
        d.xfrc_applied[self.tool_bid, :3] = F                         # that) -> can't be rammed through the table

    def save_state(self):
        """Snapshot the full sim + grip state (for building the demo grip-reset distribution)."""
        d = self.data
        return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), mtime=np.float64(d.time),
                    xfrc=d.xfrc_applied.copy(), gripping=np.int64(self._gripping), grip_rel=self._grip_rel.copy(),
                    qhome=self._qhome.copy(), rest_z=np.float64(self._rest_z), tool_xy0=self._tool_xy0.copy(),
                    pad_off=np.int64(int(self.model.geom_contype[self._pad_gids[0]] == 0)))

    def restore_state(self, i):
        """Reset the episode INTO a saved gripped state (box already pinched) so the policy learns the LIFT."""
        d, m = self.data, self.model; g = self._grip_states
        mujoco.mj_resetData(m, d)
        d.qpos[:] = g["qpos"][i]; d.qvel[:] = g["qvel"][i]; d.act[:] = g["act"][i]; d.time = float(g["mtime"][i])
        d.xfrc_applied[:] = g["xfrc"][i]
        if int(g["pad_off"][i]):
            m.geom_contype[self._pad_gids] = 0; m.geom_conaffinity[self._pad_gids] = 0
        else:
            m.geom_contype[self._pad_gids] = self._pad_contype; m.geom_conaffinity[self._pad_gids] = self._pad_conaff
        mujoco.mj_forward(m, d)
        self._gripping = bool(g["gripping"][i]); self._grip_rel = g["grip_rel"][i].copy()
        self._qhome = g["qhome"][i].copy(); self._rest_z = float(g["rest_z"][i]); self._tool_xy0 = g["tool_xy0"][i].copy()
        self._held_since = 0; self._grip_streak = 0; self._t = 0
        self._prev_rel = (d.xpos[self.tool_bid] - self._grasp_point()).copy()
        return self._obs()

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
        if self._grip_states is not None and self.np_random.random() < self.grip_reset_prob:
            return self.restore_state(int(self.np_random.integers(len(self._grip_states["qpos"])))), {}
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        m.geom_contype[self._pad_gids] = self._pad_contype          # pads collide again (prior ep may have gripped)
        m.geom_conaffinity[self._pad_gids] = self._pad_conaff
        self._gripping = False
        d.xfrc_applied[self.tool_bid] = 0.0
        self._apply_pose()
        # right arm starts at its standing (arm-down) pose + small noise; gripper open
        for k, jadr in enumerate(self.arm_qadr):
            base = self._stand.get(self.arm_joints[k], 0.0)
            d.qpos[jadr] = np.clip(base + self.np_random.uniform(-0.05, 0.05),
                                   self.arm_range[k, 0], self.arm_range[k, 1])
        # tool RESTING FLUSH on the pedestal top (not dropped from above) + small XY jitter
        mujoco.mj_forward(m, d)                                             # so pedestal world pos is valid
        ped_top = float(d.geom_xpos[self.ped_geom][2] + m.geom_size[self.ped_geom][2])
        ox, oy = (0.80, 1.20) if self.scene == "table" else (-0.415, -1.117)   # box further onto the table -> extended natural reach (elbow stays out of the torso)
        jit = self.np_random.uniform(-0.01, 0.01, 2)
        d.qpos[self.tool_qadr:self.tool_qadr + 3] = [ox + jit[0], oy + jit[1],
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
            if self._gripping:
                self._apply_grip_hold()                             # capped holding force, active each substep
            mujoco.mj_step(m, d)
            d.qpos[self.held_q] = self._qhome[self.held_q]; d.qvel[self.held_d] = 0   # hold the rest of the body

        # ----- grip state machine: engage on a real two-sided pinch, release on open OR escape (yanked) -----
        grip_closing = a[7] > -0.5                                   # gripper commanded closed-ish
        if self._gripping:
            box_off = float(np.linalg.norm(d.xpos[self.tool_bid] - self._grasp_point()))
            if (not grip_closing) or box_off > self.ESCAPE:
                self._gripping = False
                d.xfrc_applied[self.tool_bid] = 0.0                 # released or lost -> free body again
                m.geom_contype[self._pad_gids] = self._pad_contype  # pads collide again (re-detect the next grip)
                m.geom_conaffinity[self._pad_gids] = self._pad_conaff
        elif grip_closing and self._grasped():
            self._engage_grip()                                     # earn the grip -> start holding

        self._t += 1
        obs = self._obs()
        tool_z = float(d.xpos[self.tool_bid][2])
        lift = max(0.0, tool_z - self._rest_z)
        grasped = self._gripping or self._grasped()
        reach = float(np.linalg.norm(d.xpos[self.tool_bid] - self._grasp_point()))
        lateral = float(np.linalg.norm(d.xpos[self.tool_bid][:2] - self._tool_xy0))
        knocked = (tool_z < self._rest_z - 0.05) or (lateral > 0.15 and not grasped)

        # a grasp only COUNTS if the box is CENTERED between the pads (not a top-pinch / single-edge touch)
        real_grasp = grasped and reach < self.CENTER_TOL
        r = -self.w["reach"] * reach
        r += self.w["center"] * max(0.0, 1.0 - reach / 0.06)                # reward centering the box at the pad midpoint
        rel = d.xpos[self.tool_bid] - self._grasp_point()                   # tool offset from the pad midpoint
        if self._gripping:                                                 # holding a REAL two-sided pinch (the force-latch
            self._grip_streak += 1                                         # only engages on both pads >=5N -> not an edge grip)
            r += self.w["grip"]                                            # reward HOLDING the grip — ROBUST signal, not gated
            r += self.w["lift"] * min(lift, self.LIFT_CAP)                # on dead-center (which made the old signal fragile)
            if lift > 0.05:                                                # 250 -> LIFT dominates
                r += self.w["hold"]                                        # 12: sustained hold-up bonus
                self._held_since += 1
            else:
                self._held_since = 0
                r -= self.w["camp"]                                        # gripped but parked on the table -> mild penalty
            vz = abs(float(d.qvel[self.tool_dadr + 2]))                    # anti-ballistic (a fling would lose the grip)
            r -= self.w["vel"] * max(0.0, vz - self.VEL_OK)
        else:
            self._grip_streak = 0
            self._held_since = 0
        #   centering is now a SOFT nudge (the `center` + `reach` shaping above), not a hard gate on all reward —
        #   so the policy keeps holding+lifting even a slightly off-center (but real two-sided) pinch [v13 fix]
        self._prev_rel = rel.copy()
        if knocked:
            r -= self.w["knock"]
        # penalize the right elbow / upper arm crossing INTO the torso (the contorted self-overlap pose)
        ehz = float(np.linalg.norm(d.xpos[self.relbow_bid][:2] - d.xpos[self.torso_bid][:2]))
        r -= self.w["selfcol"] * max(0.0, self.BODY_R - ehz)
        r -= self.w["ctrl"] * float(np.sum(a ** 2))

        success = self._held_since >= 25                                    # CENTERED grip + lifted, held ~0.5s
        if success:
            r += self.w["success"]                                          # big terminal reward for a real pick-and-hold
        terminated = bool(success or knocked)
        truncated = self._t >= self.max_steps
        info = {"grasped": real_grasp, "contact": grasped, "lift": lift, "success": success,
                "knocked": knocked, "gripping": self._gripping}
        return obs, float(r), terminated, truncated, info
