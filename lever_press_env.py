"""
Gymnasium wrapper for the G1 LEVER task — the full-humanoid twin of ButtonPressEnv.

The robot stands in front of the SAME control panel (legs driven by the AMO policy for
balance) and RL controls the 8 arm-joint offsets to reach the lever handle and ROTATE its
hinge to a target angle. Everything about the stance + AMO leg-control machinery is copied
verbatim from env_wrapper_button.ButtonPressEnv; the only changes are:
  - target joint = lever_handle_joint (read its ANGLE, not a slide displacement)
  - reward = drive the hinge angle to a target (~0.9 rad)
  - obs includes lever angle + angle-to-target + hand->handle vector
  - active arm + reach_bias chosen for the lever's x-position (x=0.6 -> RIGHT arm)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any, Optional

from play_amo import HumanoidEnv, quat_to_euler
from unified_env import UnifiedHumanoidEnv
from reward_fn import LeverPressRewardFunction, LEVER_POSITION

# gripper tendon command: keep the gripper CLOSED so it is a solid pusher for levering.
GRIP_CLOSED = 255.0


class LeverPressEnv(gym.Env):
    """
    Environment for training the lever-rotation task.

    Action space: 8 arm joint position offsets in [-1, 1]
    Observation space: arm state + hand positions + handle position + hand->handle
                       vectors + lever angle + angle-to-target
    """

    LEVER_JOINT = "lever_handle_joint"
    LEVER_BODY = "lever_handle"
    LEVER_GRIP_BODY = "lever_grip"  # the geom/body the hand actually contacts

    def __init__(
        self,
        reward_fn: Optional[LeverPressRewardFunction] = None,
        target_angle: float = 0.9,
        freeze_arm: str = "left",
        max_episode_steps: int = 200,
        headless: bool = True,
        device: str = None,
        unified: bool = True,
        reset_in_contact: bool = True,
        curriculum: bool = False,
    ):
        super().__init__()

        self.target_angle = target_angle
        self.freeze_arm = freeze_arm.lower()
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.unified = unified
        self.reset_in_contact = reset_in_contact
        # REACH CURRICULUM (ported from the button): arm starts interpolated between the
        # contact pose (frac 0) and the rest pose (frac 1); frac_max grows on success via the
        # training callback. Makes the reach itself RL instead of an IK seed.
        self.curriculum = curriculum
        self.curriculum_frac_max = 0.0
        self._cached_a4_c = None; self._cached_w3_c = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lever handle reference position (grip world pos at rest angle)
        self.handle_position = LEVER_POSITION.copy()

        # Initialize reward function. With reset-in-contact, use the contact-mode shaping
        # (dense angle-toward-target + hold, reach/approach/distance terms disabled).
        if reward_fn is None:
            self.reward_fn = LeverPressRewardFunction(
                handle_position=self.handle_position, target_angle=target_angle,
                contact_mode=bool(unified and reset_in_contact),
            )
        else:
            self.reward_fn = reward_fn

        # Load AMO policy for leg control
        self.policy_jit = torch.jit.load("amo_jit.pt", map_location=self.device)

        # Create underlying HumanoidEnv. unified -> gripper-humanoid (g1_amo_gripper.mjb,
        # which still includes the lever) with explicit-address AMO DOF reads.
        EnvCls = UnifiedHumanoidEnv if self.unified else HumanoidEnv
        self.env = EnvCls(
            policy_jit=self.policy_jit,
            robot_type="g1",
            device=self.device,
            headless=headless,
        )

        # Lever joint + body IDs. Read the hinge ANGLE via its qpos address (robust to
        # interactive-object count, unlike the button env's joint-id-as-index shortcut).
        self.lever_joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, self.LEVER_JOINT
        )
        self.lever_qadr = self.env.model.jnt_qposadr[self.lever_joint_id]
        # The CONTACT point the hand reaches is the grip knob at the handle tip. It's a
        # GEOM (lever_grip) on the lever_handle body; use its world geom position so we
        # track the swinging tip, not the hinge body origin.
        self.handle_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.LEVER_BODY
        )
        self.grip_geom_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "lever_grip"
        )

        # Hand body IDs
        self.left_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand'
        )
        self.right_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand'
        )

        print(f"[ENV] Lever joint ID: {self.lever_joint_id}, qadr: {self.lever_qadr}, grip body ID: {self.handle_body_id}")
        print(f"[ENV] Hand IDs - Left: {self.left_hand_id}, Right: {self.right_hand_id}")

        # Arm joint configuration
        self.num_arm_joints = 8
        self.arm_joint_start = 15
        self.arm_joint_end = 23

        # Action space: 8 arm joint offsets
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_arm_joints,), dtype=np.float32,
        )

        # Observation space:
        # - arm joint positions (8)
        # - arm joint velocities (8)
        # - left hand position (3)
        # - right hand position (3)
        # - handle position (3)
        # - hand to handle vectors (6)
        # - lever angle (1)
        # - angle to target (1)
        # Total: 33
        obs_dim = 8 + 8 + 3 + 3 + 3 + 6 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0
        self.action_scale = 0.5

        # Robot positioning. The lever grip protrudes at x=0.6, further OUT laterally than the
        # buttons — reaching it from the button-style x=lever_x-0.08 stance needs shoulder
        # roll/yaw torque beyond the 25 Nm arm limit, so the gripper sags ~15cm short. Spawning
        # the robot DIRECTLY in front of the lever (x=lever_x, shoulder over the grip) makes it
        # a strong forward (shoulder-pitch) reach instead — a dynamic-hold spawn sweep put the
        # gripper within ~5cm of the grip at x=lever_x, vs 10-15cm at lever_x-0.08.
        lever_x = self.handle_position[0]
        self.robot_start_pos = np.array([lever_x, -1.62, 0.793])
        self.robot_start_quat = np.array([0.7071, 0, 0, -0.7071])  # -90deg yaw, facing -Y
        self.target_yaw = float(quat_to_euler(self.robot_start_quat)[2])
        self.torso_lean = 0.0
        self.waist_lean = 0.0

        # Active arm by lever x-position (mirror button logic). Lever at x=0.6 (>0) -> RIGHT arm.
        if lever_x < 0:
            self.freeze_arm = "right"  # use LEFT arm
            print(f"[ENV] Lever at x={lever_x:.2f} -> Using LEFT arm")
        else:
            self.freeze_arm = "left"   # use RIGHT arm
            print(f"[ENV] Lever at x={lever_x:.2f} -> Using RIGHT arm")

        # Reach-ready default arm posture: bias the ACTIVE arm toward the handle so the
        # reaching configuration sits centered in the small action envelope.
        #
        # LEVER-ARM FIX (unified gripper): the old right-arm bias was the LEFT button-reach
        # MIRRORED, which used a huge shoulder_roll (+1.56) and swung the gripper ACROSS the
        # torso to x<torso_center — a contorted, self-colliding pose that also plunged past
        # the protruding lever into the panel (y=-1.84 vs lever tip y=-1.69). Replaced with
        # a bias solved by IK (gripper pad-midpoint -> lever grip [0.60,-1.66,0.80]) that
        # reaches OUT to the lever at x=0.60 without crossing the torso. Uses shoulder_PITCH
        # to extend forward rather than shoulder_ROLL to cross. (build_unified/solve_lever_reach.)
        self.arm_reach_bias = np.zeros(self.num_arm_joints, dtype=np.float32)
        _LEFT_REACH = np.array([0.02, -1.56, -0.80, -0.61], dtype=np.float32)
        _RIGHT_LEVER_REACH = np.array([0.852, 0.065, 1.095, -0.583], dtype=np.float32)
        if self.freeze_arm == "right":   # left arm active (lever on the left, x<0)
            self.arm_reach_bias[:4] = _LEFT_REACH
        else:                            # right arm active (lever at x=0.6) — IK-solved, no torso cross
            self.arm_reach_bias[4:] = _RIGHT_LEVER_REACH

        self._solved_wrist = np.zeros(3)
        # ARM-HOLD stiffness + torque limit (unified reset-in-contact): boost ONLY the 4 right
        # shoulder/elbow gains so the arm can statically HOLD the reach onto the lever grip
        # (legs/waist/left-arm untouched -> AMO balance unaffected). The lever grip at x=0.60 is
        # a further LATERAL reach than the button (x=0.45); the default ±25 Nm arm limit
        # saturates and the pad sags ~2cm off the knob, so we also raise the right-arm torque
        # ceiling to ±60 Nm for the hold (matches build_grasp_model's arm force authority).
        if self.unified:
            self._arm_hold_stiffness = self.env.stiffness.astype(float).copy()
            self._arm_hold_stiffness[19:23] *= 4.0
            self._arm_hold_tlim = self.env.torque_limits.astype(float).copy()
            self._arm_hold_tlim[19:23] = 60.0

        # motion shaping (ported from the button env): LOW-PASS the arm command so the arm
        # physically can't flail (a reward penalty alone gets hacked) + torso-avoidance +
        # base-stability penalties. Kills Jatin's body-rocking/jerk failure mode.
        self.arm_alpha = 0.12
        self._filt_arm = np.zeros(self.num_arm_joints, dtype=np.float32)
        self.torso_w = 12.0; self.base_w = 4.0
        self.right_elbow_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
        self._prev_action = np.zeros(self.num_arm_joints, dtype=np.float32)

    def _amo_arm_step(self, arm_qtarget_right, wrist_target=None, grip_cmd=GRIP_CLOSED):
        """One control step for the reset IK servo: AMO drives the legs (balance) while the 4
        RIGHT shoulder/elbow joints PD-track `arm_qtarget_right` (boosted hold); wrist held
        toward `wrist_target` + gripper CLOSED via apply_ctrl. Mirrors the button env's servo."""
        if wrist_target is not None:
            self.env.wrist_target = np.asarray(wrist_target, dtype=float)
        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean
        self.env._extract_state()
        amo_obs = self.env._compute_observation()
        obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(
                np.array(self.env.extra_history).flatten().copy(), dtype=torch.float
            ).view(1, -1).to(self.device)
            leg_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        leg_action = np.clip(leg_action, -40.0, 40.0)
        scaled_leg = leg_action * self.env.action_scale
        self.env.last_action = np.concatenate([
            leg_action.copy(),
            (self.env.dof_pos[15:] - self.env.default_dof_pos[15:]) / self.env.action_scale])
        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] = scaled_leg + self.env.default_dof_pos[:15]
        pd_target[14] += self.waist_lean
        pd_target[19:23] = arm_qtarget_right   # right shoulder/elbow (AMO dofs 19..22)
        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0)
        if self.env._in_place_stand and np.any(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.25])
        stiff = getattr(self, "_arm_hold_stiffness", self.env.stiffness)
        tlim = getattr(self, "_arm_hold_tlim", self.env.torque_limits)
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * stiff - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -tlim, tlim)
            self.env.apply_ctrl(torque, grip_cmd)
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()

    def _servo_to_contact(self):
        """IK-servo (7-DOF arm+wrist, boosted hold) the CLOSED gripper onto the lever GRIP,
        AMO balancing throughout. The lever grip is a sphere on a shaft that PROTRUDES toward
        the robot (+Y), so a progressive straight-in approach seats the pad on the knob.
        Stops at LIGHT contact (lever just begins to turn) so the full arc is left for RL.
        Returns the solved 4-DOF right-arm qpos to plant into arm_reach_bias."""
        def grip(): return self.env.data.geom_xpos[self.grip_geom_id].copy()

        # Approach the grip knob straight-in from the +Y (robot) side at decreasing standoff,
        # boosted-hold settle each step. The grip is a knob on a soft hinge (stiffness 20) that
        # deflects when touched, so we solve IK fresh to the (current) grip each standoff and
        # keep the SOLVED arm pose; contact is reached when the pad is at the grip or the hinge
        # just begins to move. Slightly BELOW the knob so a later push drives it UP (+angle).
        a4 = self.env.data.qpos[self.env.rarm_qadr].copy()
        w3 = self.env.wrist_target.copy()
        # advance to the knob and then aim slightly INTO it (past center in -Y) so the closed
        # pads seat ON the knob (radius 3.5cm) rather than hovering ~2cm off its front face.
        for standoff in [0.14, 0.10, 0.07, 0.05, 0.03, 0.01, -0.015]:
            tgt = grip() + np.array([0.0, standoff, -0.005])
            a4, w3, err = self.env.solve_right_arm7_ik(tgt)
            for _ in range(60):
                self._amo_arm_step(a4, wrist_target=w3)
            gp = self.env.gripper_point()
            # stop once seated ON the knob (pad within the knob radius) or the hinge turned
            if np.linalg.norm(gp - grip()) < 0.035 or float(self.env.data.qpos[self.lever_qadr]) > 0.04:
                break
        self._solved_wrist = w3.copy()
        return a4

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)

        # reset the wrist hold to level (the reset-in-contact servo may have tilted it last episode)
        if self.unified:
            self.env.wrist_target = np.zeros(3)

        # Position robot in front of the lever — pelvis qadr found dynamically.
        robot_qpos_start = self.env.model.jnt_qposadr[mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
        self.env.data.qpos[robot_qpos_start:robot_qpos_start+3] = self.robot_start_pos
        self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7] = self.robot_start_quat
        if self.curriculum:
            # STANCE ROBUSTNESS (ported from button): randomize arrival stance up to +-4cm/+-5deg
            # so the policy tolerates wherever the walk lands. Noise SCALES with curriculum level
            # (zero at frac 0 — keeps the contact bootstrap easy). Heading setpoint stays nominal.
            k = self.curriculum_frac_max
            self.env.data.qpos[robot_qpos_start:robot_qpos_start+2] += k * np.random.uniform(-0.04, 0.04, 2)
            dyaw = k * np.random.uniform(-0.09, 0.09)
            qz = np.array([np.cos(dyaw/2), 0.0, 0.0, np.sin(dyaw/2)])
            w1,x1,y1,z1 = qz; w2,x2,y2,z2 = self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7].copy()
            self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7] = [
                w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]

        self.env.data.qvel[:] = 0.0

        # Reset lever to closed (angle 0)
        self.env.data.qpos[self.lever_qadr] = 0.0

        mujoco.mj_forward(self.env.model, self.env.data)

        # Reset env state (verbatim from button env)
        self.env._extract_state()
        self.env.last_action = np.zeros(self.env.num_dofs, dtype=np.float32)
        self.env.arm_action = self.env.default_dof_pos[15:].copy()
        self.env.prev_arm_action = self.env.default_dof_pos[15:].copy()
        self.env.arm_blend = 0.0
        self.env._in_place_stand = True
        self.env.gait_cycle = np.array([0.25, 0.25])

        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean

        self.env.proprio_history = deque(maxlen=self.env.history_len)
        self.env.extra_history = deque(maxlen=self.env.extra_history_len)
        for _ in range(self.env.history_len):
            self.env.proprio_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        for _ in range(self.env.extra_history_len):
            self.env.extra_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))

        # Let AMO settle (verbatim from button env)
        for _ in range(10):
            self.env.viewer.commands[:] = 0.0
            self.env.viewer.commands[1] = self.target_yaw
            self.env.viewer.commands[5] = self.torso_lean
            self.env._extract_state()
            amo_obs = self.env._compute_observation()

            obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                extra_hist = torch.tensor(
                    np.array(self.env.extra_history).flatten().copy(),
                    dtype=torch.float,
                ).view(1, -1).to(self.device)
                leg_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

            leg_action = np.clip(leg_action, -40.0, 40.0)
            scaled_leg_action = leg_action * self.env.action_scale

            pd_target = self.env.default_dof_pos.copy()
            pd_target[:15] = scaled_leg_action + self.env.default_dof_pos[:15]
            pd_target[14] += self.waist_lean

            for _ in range(self.env.sim_decimation):
                torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
                torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
                if self.unified:
                    self.env.apply_ctrl(torque, GRIP_CLOSED)
                else:
                    self.env.data.ctrl = torque
                mujoco.mj_step(self.env.model, self.env.data)
                self.env._extract_state()

        # RESET-IN-CONTACT: IK the CLOSED gripper onto the lever grip so the episode STARTS
        # touching it. Plant the solved right-arm pose into arm_reach_bias + the solved wrist
        # into wrist_target so zero-action HOLDS contact; RL only has to push the lever through
        # its arc + hold at target.
        if self.unified and self.reset_in_contact:
            solved_right = self._servo_to_contact()
            self.arm_reach_bias[4:] = (solved_right - self.env.default_dof_pos[19:23]).astype(np.float32)
            self.env.wrist_target = self._solved_wrist.copy()

        elif self.unified and self.curriculum:
            # CURRICULUM reach+turn (ported from button): seat the arm between the CONTACT pose
            # (frac 0) and the REST pose (frac 1); frac ~ U[0, frac_max]. RL learns the reach
            # from progressively farther, ending at the rest pose the end-to-end demo hands it.
            if self._cached_a4_c is None:
                self._cached_a4_c = self._servo_to_contact().copy()
                self._cached_w3_c = self._solved_wrist.copy()
            a4_c = self._cached_a4_c; w3_c = self._cached_w3_c
            a4_rest = self.env.default_dof_pos[19:23].copy()
            frac = float(np.random.uniform(0.0, max(1e-3, self.curriculum_frac_max)))
            a4_start = (1.0 - frac) * a4_c + frac * a4_rest
            self.arm_reach_bias[4:] = (a4_start - self.env.default_dof_pos[19:23]).astype(np.float32)
            self.env.wrist_target = w3_c.copy()
            self._solved_wrist = w3_c.copy()
            for _ in range(60):
                self._amo_arm_step(a4_start, wrist_target=w3_c)
            self._cur_frac = frac

        self.episode_steps = 0
        self.episode_return = 0.0
        self.reward_fn.reset()
        self._prev_action = np.zeros(self.num_arm_joints, dtype=np.float32)
        self._filt_arm = np.zeros(self.num_arm_joints, dtype=np.float32)

        return self._get_obs(), {}

    def set_curriculum_frac(self, f: float):
        """Advance the reach curriculum (called by the training callback via env_method)."""
        self.curriculum_frac_max = float(np.clip(f, 0.0, 1.0))

    def _right_hand_pos(self) -> np.ndarray:
        """Right 'hand' contact point: gripper pad-midpoint (unified) or rubber hand."""
        if self.unified:
            return self.env.gripper_point()
        return self.env.data.xpos[self.right_hand_id]

    def _get_lever_angle(self) -> float:
        """Get the lever hinge angle (rad)."""
        return float(self.env.data.qpos[self.lever_qadr])

    def _get_handle_pos(self) -> np.ndarray:
        """World position of the grip knob (geom) the hand contacts — tracks the
        swinging tip, not the hinge origin."""
        return self.env.data.geom_xpos[self.grip_geom_id]

    def _get_obs(self) -> np.ndarray:
        arm_pos = self.env.dof_pos[self.arm_joint_start:self.arm_joint_end]
        arm_vel = self.env.dof_vel[self.arm_joint_start:self.arm_joint_end]

        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self._right_hand_pos()

        handle_pos = self._get_handle_pos()
        left_to_handle = handle_pos - left_hand_pos
        right_to_handle = handle_pos - right_hand_pos

        lever_angle = self._get_lever_angle()
        angle_to_target = self.target_angle - lever_angle

        obs = np.concatenate([
            arm_pos,                # 8
            arm_vel * 0.1,          # 8
            left_hand_pos,          # 3
            right_hand_pos,         # 3
            handle_pos,             # 3
            left_to_handle,         # 3
            right_to_handle,        # 3
            [lever_angle],          # 1
            [angle_to_target],      # 1
        ]).astype(np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.episode_steps += 1

        action = np.array(action, dtype=np.float32)

        if self.freeze_arm == "left":
            action[:4] = 0.0
        elif self.freeze_arm == "right":
            action[4:] = 0.0

        scaled_arm_action = action * self.action_scale
        if self.unified:                                    # LOW-PASS -> arm eases to the target, cannot flail
            self._filt_arm = (1 - self.arm_alpha) * self._filt_arm + self.arm_alpha * scaled_arm_action
            scaled_arm_action = self._filt_arm

        # === LEG CONTROL FROM AMO (standing still) — verbatim from button env ===
        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean
        self.env._extract_state()
        amo_obs = self.env._compute_observation()

        obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(
                np.array(self.env.extra_history).flatten().copy(),
                dtype=torch.float,
            ).view(1, -1).to(self.device)
            leg_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

        leg_action = np.clip(leg_action, -40.0, 40.0)
        scaled_leg_action = leg_action * self.env.action_scale

        self.env.last_action = np.concatenate([
            leg_action.copy(),
            (self.env.dof_pos[15:] - self.env.default_dof_pos[15:]) / self.env.action_scale
        ])

        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] = scaled_leg_action + self.env.default_dof_pos[:15]
        pd_target[14] += self.waist_lean
        pd_target[15:] = self.env.default_dof_pos[15:] + self.arm_reach_bias + scaled_arm_action

        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0
        )
        if self.env._in_place_stand and np.any(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.25])
        if not self.env._in_place_stand and np.all(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.75])

        # boosted right-arm hold stiffness + torque ceiling for the contact-hold task
        if self.unified and (self.reset_in_contact or self.curriculum):
            stiff = self._arm_hold_stiffness; tlim = self._arm_hold_tlim
        else:
            stiff = self.env.stiffness; tlim = self.env.torque_limits
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * stiff - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -tlim, tlim)
            if self.unified:
                self.env.apply_ctrl(torque, GRIP_CLOSED)
            else:
                self.env.data.ctrl = torque
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()

        # Reward
        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        rpy = np.zeros(3)
        ang_vel = self.env.ang_vel

        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self._right_hand_pos()

        lever_angle = self._get_lever_angle()
        handle_pos = self._get_handle_pos()

        reward, info = self.reward_fn.compute_reward(
            position=robot_pos,
            rpy=rpy,
            ang_vel=ang_vel,
            action=action,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            lever_angle=lever_angle,
            current_handle_pos=handle_pos,
        )

        # motion shaping (ported from button): torso-avoidance + base-stability. The low-pass
        # above already prevents flailing at the control level.
        if self.unified:
            if self.right_elbow_id >= 0:
                eb = self.env.data.xpos[self.right_elbow_id][:2]; pv = self.env.data.xpos[self.env.pelvis_id][:2]
                reward -= self.torso_w * max(0.0, 0.11 - float(np.linalg.norm(eb - pv)))
            reward -= self.base_w * float(np.linalg.norm(self.env.data.qvel[0:2]))
            self._prev_action = np.asarray(action, np.float32).copy()

        self.episode_return += reward

        terminated = False
        truncated = False

        if self.reward_fn.lever_turned:
            info['success'] = True

        if robot_pos[2] < 0.4:
            terminated = True
            info['fell'] = True

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        if terminated or truncated:   # per-episode success for the curriculum callback
            info['is_success'] = bool(self.reward_fn.lever_turned)
            info['curriculum_frac_max'] = self.curriculum_frac_max

        if not self.headless and hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.cam.lookat = robot_pos.astype(np.float32)
            self.env.viewer.render()

        info['episode_return'] = self.episode_return
        info['episode_steps'] = self.episode_steps

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.close()

    def render(self):
        if hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.render()

    def render_frame(self, width: int = 480, height: int = 360) -> np.ndarray:
        if not hasattr(self, '_renderer') or self._renderer is None:
            self._renderer = mujoco.Renderer(self.env.model, height, width)

        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        handle_pos = self._get_handle_pos()

        lookat = np.array([handle_pos[0], handle_pos[1] + 0.2, 0.85])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = 1.2
        cam.azimuth = 0
        cam.elevation = -15

        self._renderer.update_scene(self.env.data, camera=cam)
        return self._renderer.render()
