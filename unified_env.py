"""
UnifiedHumanoidEnv: the AMO walking humanoid loaded from g1_amo_gripper.mjb (g1 + Robotiq
gripper on the right wrist), keeping the AMO PD-torque control scheme.

The gripper build APPENDS the 3 wrist + 8 gripper DOFs at the end of qpos, so the 23 AMO
DOFs are NO LONGER the last 23 (that would be qpos[-23:] = the gripper). We therefore read
the AMO DOFs by their explicit, contiguous qpos/dof addresses (identical to plain g1.xml),
and drive ctrl per-actuator:
    ctrl[0:23]  = AMO PD torque (leg+waist+arm)   <- the AMO scheme, unchanged
    ctrl[23:26] = wrist PD-hold torque toward a wrist target (roll/pitch/yaw)
    ctrl[26]    = gripper tendon command (0 open .. 255 closed)
"""
import numpy as np
import mujoco
import torch
from collections import deque

from play_amo import HumanoidEnv, G1_CONFIG, quat_to_euler

AMO_JOINTS = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
]
WRIST_JOINTS = ["right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
# the 4 right shoulder/elbow joints the reset IK servo moves (wrist stays PD-held)
RARM_JOINTS = ["right_shoulder_pitch_joint","right_shoulder_roll_joint",
               "right_shoulder_yaw_joint","right_elbow_joint"]


class UnifiedHumanoidEnv(HumanoidEnv):
    def _load_robot_config(self, robot_type):
        super()._load_robot_config(robot_type)
        self.model_path = "g1_amo_gripper.mjb"

    def _init_simulation(self):
        self.sim_duration = 2000.0
        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_binary_path(self.model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        # explicit AMO DOF addresses (qpos/qvel), contiguous, == plain g1.xml
        self.amo_qadr = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in AMO_JOINTS])
        self.amo_dadr = np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in AMO_JOINTS])
        assert np.all(np.diff(self.amo_qadr) == 1), "AMO qpos addrs not contiguous"

        # wrist addresses + actuator ids
        self.wrist_qadr = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in WRIST_JOINTS])
        self.wrist_dadr = np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in WRIST_JOINTS])
        self.wrist_act = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
                                   for j in WRIST_JOINTS])
        self.grip_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg_fingers_actuator")

        # right-arm (4 shoulder/elbow) qpos/dof addresses + joint ranges, for the reset IK
        # servo that places the gripper ON a target (reset-in-contact). These 4 joints are the
        # ones the button/lever env's arm_reach_bias / RL action drive (wrist stays PD-held).
        self.rarm_qadr = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
        self.rarm_dadr = np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
        self.rarm_range = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        # gripper pad bodies (for grasp-point IK)
        self.lpad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rg_left_pad")
        self.rpad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rg_right_pad")

        # wrist PD gains (hold at target) + torque limit. Kp raised so the wrist can HOLD a
        # tilted pose (needed by reset-in-contact, which orients the wrist up to reach a high
        # button cap) against gravity, not just sag back to level.
        self.wrist_kp = np.array([120.0, 120.0, 120.0])
        self.wrist_kv = np.array([4.0, 4.0, 4.0])
        self.wrist_tlim = np.array([40.0, 40.0, 40.0])
        self.wrist_target = np.zeros(3)   # hold wrist level (rest pose)

    def _extract_state(self):
        # read the 23 AMO DOFs by explicit address (NOT [-23:], which is now the gripper)
        self.dof_pos = self.data.qpos.astype(np.float32)[self.amo_qadr]
        self.dof_vel = self.data.qvel.astype(np.float32)[self.amo_dadr]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def wrist_torque(self):
        wp = self.data.qpos[self.wrist_qadr]
        wv = self.data.qvel[self.wrist_dadr]
        t = (self.wrist_target - wp) * self.wrist_kp - wv * self.wrist_kv
        return np.clip(t, -self.wrist_tlim, self.wrist_tlim)

    def apply_ctrl(self, amo_torque, grip_cmd):
        """Write the full 27-wide ctrl: AMO torque (0:23) + wrist hold (23:26) + gripper (26)."""
        self.data.ctrl[:23] = np.clip(amo_torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.wrist_act] = self.wrist_torque()
        self.data.ctrl[self.grip_act] = grip_cmd

    def gripper_point(self):
        """World position of the gripper 'hand' = pad midpoint (the contact point that
        presses a button / pushes a lever). Replaces right_rubber_hand for press/lever."""
        return 0.5 * (self.data.xpos[self.lpad] + self.data.xpos[self.rpad])

    def _rarm_grasp_jac(self):
        """Translational Jacobian of the gripper pad-midpoint wrt the 4 right shoulder/elbow DOFs."""
        jl = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jl, None, self.lpad)
        mujoco.mj_jacBody(self.model, self.data, jr, None, self.rpad)
        return 0.5 * (jl + jr)[:, self.rarm_dadr]

    def right_arm_ik_step(self, cart_target, damping=0.03, gain=5.0, step_clip=0.10, dq_clip=0.15):
        """One DLS-IK step: return the 4-DOF right-arm qpos target that nudges the gripper
        pad-midpoint toward `cart_target` (like ik_pick.servo_action, but for the 4 AMO
        shoulder/elbow joints only). Caller PD-tracks the returned target."""
        J = self._rarm_grasp_jac()
        err = np.clip((cart_target - self.gripper_point()) * gain, -step_clip, step_clip)
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err), -dq_clip, dq_clip)
        q = self.data.qpos[self.rarm_qadr] + dq
        return np.clip(q, self.rarm_range[:, 0], self.rarm_range[:, 1])

    def right_arm7_ik_step(self, cart_target, damping=0.03, gain=5.0, step_clip=0.10, dq_clip=0.15):
        """7-DOF DLS-IK step over the 4 shoulder/elbow + 3 wrist joints, so the gripper can
        RAISE/ORIENT the pad up to a high target (e.g. a button cap at z=0.90 that the fixed
        downward wrist can't reach). Returns (arm4_target, wrist3_target)."""
        d7 = np.concatenate([self.rarm_dadr, self.wrist_dadr])
        q7 = np.concatenate([self.rarm_qadr, self.wrist_qadr])
        jl = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jl, None, self.lpad)
        mujoco.mj_jacBody(self.model, self.data, jr, None, self.rpad)
        J = 0.5 * (jl + jr)[:, d7]
        err = np.clip((cart_target - self.gripper_point()) * gain, -step_clip, step_clip)
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err), -dq_clip, dq_clip)
        q = self.data.qpos[q7] + dq
        # clamp to ranges
        lo = np.concatenate([self.rarm_range[:, 0],
                             np.array([self.model.jnt_range[mujoco.mj_name2id(
                                 self.model, mujoco.mjtObj.mjOBJ_JOINT, j)][0] for j in WRIST_JOINTS])])
        hi = np.concatenate([self.rarm_range[:, 1],
                             np.array([self.model.jnt_range[mujoco.mj_name2id(
                                 self.model, mujoco.mjtObj.mjOBJ_JOINT, j)][1] for j in WRIST_JOINTS])])
        q = np.clip(q, lo, hi)
        return q[:4], q[4:]

    def solve_right_arm7_ik(self, cart_target, iters=400, tol=0.008, damping=0.02):
        """KINEMATIC 7-DOF IK (no dynamics): iterate mj_forward writing the 4 arm + 3 wrist
        qpos until the gripper pad-midpoint reaches cart_target. Returns (arm4, wrist3, err).
        Restores qpos afterward so it doesn't perturb the live sim state.
        Used by reset-in-contact to get a solid TARGET pose that PD can then settle onto."""
        d7 = np.concatenate([self.rarm_dadr, self.wrist_dadr])
        q7 = np.concatenate([self.rarm_qadr, self.wrist_qadr])
        lo = np.concatenate([self.rarm_range[:, 0],
                             np.array([self.model.jnt_range[mujoco.mj_name2id(
                                 self.model, mujoco.mjtObj.mjOBJ_JOINT, j)][0] for j in WRIST_JOINTS])])
        hi = np.concatenate([self.rarm_range[:, 1],
                             np.array([self.model.jnt_range[mujoco.mj_name2id(
                                 self.model, mujoco.mjtObj.mjOBJ_JOINT, j)][1] for j in WRIST_JOINTS])])
        saved = self.data.qpos.copy()
        for _ in range(iters):
            jl = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jl, None, self.lpad)
            mujoco.mj_jacBody(self.model, self.data, jr, None, self.rpad)
            J = 0.5 * (jl + jr)[:, d7]
            err = cart_target - self.gripper_point()
            if np.linalg.norm(err) < tol:
                break
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err), -0.05, 0.05)
            self.data.qpos[q7] = np.clip(self.data.qpos[q7] + dq, lo, hi)
            mujoco.mj_forward(self.model, self.data)
        sol = self.data.qpos[q7].copy()
        final_err = float(np.linalg.norm(cart_target - self.gripper_point()))
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return sol[:4], sol[4:], final_err
