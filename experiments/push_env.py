"""
Non-prehensile PUSH task — the RL skill where RL genuinely shines (contact + friction dynamics,
no precise grip needed, so it sidesteps the grasp-pop wall entirely).

The fixed-base G1 pushes the cube across the table to a target spot. The grasp force-latch is
DISABLED (GRIP_FORCE huge) so the box is pushed/slid, never gripped+carried. Reward is dense
push-progress toward the target, with the box kept ON the table (no lifting/flinging).

Reuses GraspFixedEnv's robot + scene + held-DOF freezing; overrides reset/obs/reward.
"""
from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import spaces
from grasp_fixed_env import GraspFixedEnv


class PushEnv(GraspFixedEnv):
    def __init__(self, max_steps: int = 200, **kw):
        kw.pop("scene", None)
        super().__init__(scene="table", max_steps=max_steps, **kw)
        self.GRIP_FORCE = 1e9                       # disable the grasp latch -> box is PUSHED, not gripped
        self.TARGET_TOL = 0.03                      # success: box within 3cm of target (xy)
        self.wp = {"reach": 3.0, "progress": 50.0, "dist": 3.0, "lift_pen": 10.0,
                   "success": 60.0, "ctrl": 1e-3}
        self.target = np.zeros(2)
        self._prev_dist = 0.0
        # obs = parent(19) + box->target xy (2) + hand->box xy (2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (19 + 4,), np.float32)

    def _obs(self):
        base = super()._obs()                       # 19
        box = self.data.xpos[self.tool_bid]; hand = self._grasp_point()
        b2t = self.target - box[:2]                  # box -> target (xy)
        h2b = box[:2] - hand[:2]                     # hand -> box (xy)
        return np.concatenate([base, b2t, h2b]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)   # box settled on table, body frozen
        d, m = self.data, self.model
        # RESET-IN-CONTACT: IK-servo the (open) hand onto the box so the episode STARTS at contact. The policy
        # then only has to learn the PUSH, not the reach-and-commit it kept camping on. [push-v2]
        for _ in range(80):
            jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jl, None, self.lpad); mujoco.mj_jacBody(m, d, jr, None, self.rpad)
            Jg = 0.5 * (jl + jr)[:, self.arm_dadr]
            err = np.clip((d.xpos[self.tool_bid] - self._grasp_point()) * 3.0, -0.06, 0.06)
            dq = Jg.T @ np.linalg.solve(Jg @ Jg.T + (0.015 ** 2) * np.eye(3), err)  # damped least squares (no singular stall)
            q = d.qpos[self.arm_qadr] + np.clip(dq, -0.18, 0.18)
            a = np.zeros(8, np.float32); a[:7] = np.clip((q - self.arm_mid) / self.arm_half, -1, 1); a[7] = -1.0
            super().step(a)
        box = d.xpos[self.tool_bid][:2].copy()
        # target only in the FORWARD/lateral arc (+Y = away from robot): the hand reaches the box from
        # the robot side, so it can push the box away/sideways but NOT back toward the robot (unreachable).
        ang = self.np_random.uniform(-np.pi / 2, np.pi / 2)        # 0 = straight away (+Y); +-90 = pure lateral
        dist = self.np_random.uniform(0.08, 0.14)                  # target 8-14cm from the box
        self.target = box + dist * np.array([np.sin(ang), np.cos(ang)])
        self._prev_dist = float(np.linalg.norm(box - self.target))
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, np.float32).copy(); action[7] = -1.0   # gripper open: a straddle pusher
        _, _, _, trunc, _ = super().step(action)    # advance physics + held-DOF freeze (latch is disabled)
        d = self.data
        box = d.xpos[self.tool_bid]; hand = self._grasp_point()
        dist = float(np.linalg.norm(box[:2] - self.target))
        reach = float(np.linalg.norm(box - hand))
        box_v = float(np.linalg.norm(d.qvel[self.tool_dadr:self.tool_dadr + 2]))
        lift = box[2] - self._rest_z

        r = self.wp["reach"] * max(0.0, 1.0 - reach / 0.20)             # STRONG dense pull: hand -> box (=3 on box,
                                                                        # 0 at >=20cm). Old -0.4*reach was too weak ->
                                                                        # the hand never reached the box [push-v1 fix]
        r += self.wp["progress"] * (self._prev_dist - dist)             # DENSE push progress toward target
        r -= self.wp["dist"] * dist                                     # closer is better
        r -= self.wp["lift_pen"] * max(0.0, abs(lift) - 0.02)           # keep it ON the table (push, don't lift/fling)
        r -= self.wp["ctrl"] * float(np.sum(np.clip(action, -1, 1) ** 2))
        self._prev_dist = dist

        success = dist < self.TARGET_TOL
        if success:
            r += self.wp["success"]
        knocked = (box[2] < self._rest_z - 0.06) or (lift > 0.10)       # box off the table / launched
        if knocked:
            r -= 5.0
        self._t += 1
        terminated = bool(success or knocked)
        truncated = self._t >= self.max_steps
        info = {"push_dist": dist, "success": success, "knocked": knocked, "box_v": box_v, "reach": reach}
        return self._obs(), float(r), terminated, truncated, info
