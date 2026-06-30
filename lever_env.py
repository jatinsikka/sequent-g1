"""
LEVER task — the fixed-base G1 rotates a hinged handle to a target angle.

Mirrors PushEnv: subclasses GraspFixedEnv (fixed robot at the table, body/legs frozen each step,
RL drives the 7-DOF right arm). The grasp force-latch is DISABLED (GRIP_FORCE huge) so the closed
gripper acts purely as a pusher against the lever handle. Reward is STRONG hand->handle reach
shaping + DENSE progress on the hinge angle toward a target, with a success bonus when the lever is
held within tolerance of the target.

Model: g1_robotiq_lever.mjb (built by build_grasp_model.py 'lever'): the table scene + a fixed
mount with a hinged capsule handle on a world-X hinge, range [0, 1.4] rad.
"""
from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import spaces
from grasp_fixed_env import GraspFixedEnv


class LeverEnv(GraspFixedEnv):
    def __init__(self, max_steps: int = 200, target_angle: float | None = None,
                 randomize_target: bool = True, **kw):
        kw.pop("scene", None); kw.pop("model_path", None)
        # load the lever model through the "table" code path (grasp_box + right_table_top exist there)
        super().__init__(scene="table", model_path="g1_robotiq_lever.mjb", max_steps=max_steps, **kw)
        self.GRIP_FORCE = 1e9                        # disable the grasp latch -> handle is PUSHED, not gripped

        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "lever_hinge")
        self.lever_qadr = m.jnt_qposadr[jid]
        self.lever_dadr = m.jnt_dofadr[jid]
        self.lever_range = m.jnt_range[jid].copy()   # [0, 1.4]
        self.handle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lever_arm")
        self.handle_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lever_handle")

        # the grasp_box exists only to satisfy the parent table-scene lookups; for the lever task it
        # is parked off the table AND its collision is killed so it can never foul the handle.
        m.geom_contype[self.tool_geom] = 0
        m.geom_conaffinity[self.tool_geom] = 0
        m.geom_rgba[self.tool_geom] = [0, 0, 0, 0]

        # the lever hinge is NOT part of the robot body -> the parent's held-DOF freeze would PIN it.
        # Free it (and the gripper-disabled latch means nothing else touches it) so the arm can rotate it.
        self.held_q = np.array([i for i in self.held_q if i != self.lever_qadr])
        self.held_d = np.array([i for i in self.held_d if i != self.lever_dadr])

        self.target_angle = 1.0 if target_angle is None else float(target_angle)
        self.randomize_target = randomize_target
        self.ANGLE_TOL = 0.10                        # success: within ~0.1 rad (5.7 deg) of target
        self.HOLD_STEPS = 15                         # held within tol this many steps -> success
        self.wp = {"reach": 4.0, "progress": 40.0, "dist": 2.0, "success": 60.0, "ctrl": 1e-3,
                   "angle": 6.0}   # sharper reach + a small "any rotation pays" term to break the cold start

        self._target = self.target_angle
        self._prev_err = 0.0
        self._at_target = 0
        # obs = parent(19) + [lever_angle, angle_to_target, handle->hand xyz(3)]
        self.observation_space = spaces.Box(-np.inf, np.inf, (19 + 5,), np.float32)

    # handle tip (top of the capsule) in world coords — the point the hand should reach
    def _handle_tip(self):
        d = self.data
        # contact point ~mid-upper handle (handle spans local z 0..0.08); transform by lever_arm frame
        local = np.array([0.0, 0.0, 0.06])
        wp = np.zeros(3)
        mujoco.mju_rotVecQuat(wp, local, d.xquat[self.handle_bid])
        return d.xpos[self.handle_bid] + wp

    def _obs(self):
        base = super()._obs()                        # 19
        ang = float(self.data.qpos[self.lever_qadr])
        a2t = self._target - ang
        hand = self._grasp_point()
        h2h = self._handle_tip() - hand              # handle tip -> hand (xyz)
        return np.concatenate([base, [ang, a2t], h2h]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)    # body frozen, arm at rest, box settled
        d = self.data
        d.qpos[self.lever_qadr] = 0.0                # lever starts closed
        d.qvel[self.lever_dadr] = 0.0
        self._qhome[self.lever_qadr] = 0.0           # not a held dof, but keep home consistent
        if self.randomize_target:
            self._target = float(self.np_random.uniform(0.7, 1.2))
        else:
            self._target = self.target_angle
        mujoco.mj_forward(self.model, d)
        self._prev_err = abs(float(d.qpos[self.lever_qadr]) - self._target)
        self._at_target = 0
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.array(action, dtype=np.float32).copy()
        a[7] = 1.0                                   # keep the gripper CLOSED -> a solid pusher
        super().step(a)                              # advance physics + held-DOF freeze (latch disabled)
        d = self.data

        ang = float(d.qpos[self.lever_qadr])
        err = abs(ang - self._target)
        hand = self._grasp_point()
        reach = float(np.linalg.norm(self._handle_tip() - hand))

        r = self.wp["reach"] * max(0.0, 1.0 - reach / 0.12)     # SHARP dense pull: hand -> handle (gradient
                                                                 # holds all the way in, not saturated at 20cm)
        r += self.wp["progress"] * (self._prev_err - err)        # DENSE progress: shrink |angle - target|
        r += self.wp["angle"] * min(ang, self._target)           # any rotation toward target pays (cold-start break)
        r -= self.wp["dist"] * err                               # closer to target is better
        r -= self.wp["ctrl"] * float(np.sum(np.clip(a, -1, 1) ** 2))
        self._prev_err = err

        within = err < self.ANGLE_TOL
        self._at_target = self._at_target + 1 if within else 0
        success = self._at_target >= self.HOLD_STEPS
        if success:
            r += self.wp["success"]

        self._t += 1
        terminated = bool(success)
        truncated = self._t >= self.max_steps
        info = {"lever_angle": ang, "angle_err": err, "target": self._target,
                "success": success, "reach": reach}
        return self._obs(), float(r), terminated, truncated, info
