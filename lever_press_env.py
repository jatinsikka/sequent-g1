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
from reward_fn import LeverPressRewardFunction, LEVER_POSITION


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
    ):
        super().__init__()

        self.target_angle = target_angle
        self.freeze_arm = freeze_arm.lower()
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lever handle reference position (grip world pos at rest angle)
        self.handle_position = LEVER_POSITION.copy()

        if reward_fn is None:
            self.reward_fn = LeverPressRewardFunction(
                handle_position=self.handle_position, target_angle=target_angle
            )
        else:
            self.reward_fn = reward_fn

        # Load AMO policy for leg control
        self.policy_jit = torch.jit.load("amo_jit.pt", map_location=self.device)

        # Create underlying HumanoidEnv (loads g1.xml, which includes the lever)
        self.env = HumanoidEnv(
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

        # Robot positioning — mirror the button stance. The lever grip is at x=0.6, so
        # stand directly in front of it (same -0.08 x nudge the button env uses to absorb
        # reach/recoil residual). Buttons at y=-1.85 spawn the robot at y=-1.62.
        lever_x = self.handle_position[0]
        self.robot_start_pos = np.array([lever_x - 0.08, -1.62, 0.793])
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
        # reaching configuration sits centered in the small action envelope. Same proven
        # left-arm reach offsets the button env uses, mirrored for the active arm.
        self.arm_reach_bias = np.zeros(self.num_arm_joints, dtype=np.float32)
        _LEFT_REACH = np.array([0.02, -1.56, -0.80, -0.61], dtype=np.float32)
        if self.freeze_arm == "right":   # left arm active
            self.arm_reach_bias[:4] = _LEFT_REACH
        else:                            # right arm active (mirror roll & yaw sign)
            self.arm_reach_bias[4:] = _LEFT_REACH * np.array([1, -1, -1, 1], dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)

        # Position robot in front of the lever — pelvis qadr found dynamically.
        robot_qpos_start = self.env.model.jnt_qposadr[mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
        self.env.data.qpos[robot_qpos_start:robot_qpos_start+3] = self.robot_start_pos
        self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7] = self.robot_start_quat

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
                self.env.data.ctrl = torque
                mujoco.mj_step(self.env.model, self.env.data)
                self.env._extract_state()

        self.episode_steps = 0
        self.episode_return = 0.0
        self.reward_fn.reset()

        return self._get_obs(), {}

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
        right_hand_pos = self.env.data.xpos[self.right_hand_id]

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

        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            self.env.data.ctrl = torque
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()

        # Reward
        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        rpy = np.zeros(3)
        ang_vel = self.env.ang_vel

        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self.env.data.xpos[self.right_hand_id]

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
