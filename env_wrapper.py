"""
Gymnasium wrapper for G1 humanoid RL environment.

This module wraps the HumanoidEnv class from play_amo.py into a Gymnasium
(formerly OpenAI Gym) compatible interface, enabling use with Stable Baselines 3.

The action space is continuous and matches the controls in play_amo.py:
  [vx, vy, yaw, height, torso_yaw, torso_pitch, torso_roll, arm_control_flag]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import os
from typing import Tuple, Dict, Any
from collections import deque
import mujoco

from play_amo import HumanoidEnv
from reward_fn import RewardFunction


class G1RLEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for G1 humanoid robot RL training.
    
    Action space:
        - Box(-1, 1, (8,)) representing:
          [vx, vy, yaw, height, torso_yaw, torso_pitch, torso_roll, arm_flag]
    
    Observation space:
        - Box containing proprioceptive state (joint positions, velocities, etc.)
          and exteroceptive features (position, orientation).
    """
    
    def __init__(
        self,
        policy_jit_path: str = "amo_jit.pt",
        robot_type: str = "g1",
        device: str = "cuda",
        action_scale: float = 0.25,
        max_episode_steps: int = 2000,
        reward_fn: RewardFunction = None,
        headless: bool = True,
        min_height: float = 0.4,
        max_roll: float = 0.8,
        max_pitch: float = 0.8,
        goal_distance: float = 0.1,
        max_episode_time: float = 45.0,
        verbose: int = 0,
    ):
        """
        Initialize the G1 RL environment.
        
        Args:
            policy_jit_path: Path to the pre-trained JIT policy.
            robot_type: Robot model type (default "g1").
            device: Device for torch ("cuda" or "cpu").
            action_scale: Scaling factor for actions.
            max_episode_steps: Maximum steps per episode.
            reward_fn: RewardFunction instance; uses default if None.
            headless: If True, disable MuJoCo viewer (faster training, no GUI).
            min_height: Minimum torso height before termination (fall detection).
            max_roll: Maximum absolute roll angle before termination (radians).
            max_pitch: Maximum absolute pitch angle before termination (radians).
            goal_distance: Distance to goal for successful termination (meters).
            max_episode_time: Maximum episode time before termination (seconds).
            verbose: Verbosity level (0 = silent, 1 = info, 2 = debug).
        """
        super().__init__()
        
        self.policy_jit_path = policy_jit_path
        self.robot_type = robot_type
        self.device = device
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.verbose = verbose
        
        # Termination thresholds
        self.min_height = min_height
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.goal_distance = goal_distance
        self.max_episode_time = max_episode_time
        
        # Settling period configuration
        # TODO: figure out if this is best way to let robot settle after spawn
        self.settling_steps = 50  
        self.steps_since_reset = 0
        
        # Disable GLFW/rendering in headless mode by setting env var before import
        if self.headless:
            os.environ['MUJOCO_GL'] = 'osmesa'  # Use off-screen rendering
        
        # Initialize the base humanoid environment
        self.policy_jit = torch.jit.load(policy_jit_path, map_location=device)
        self.env = HumanoidEnv(
            policy_jit=self.policy_jit,
            robot_type=robot_type,
            device=device,
        )
        
        # Handle viewer for headless mode
        if self.headless:
            # If viewer exists, try to close it
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                try:
                    if hasattr(self.env.viewer, 'close'):
                        self.env.viewer.close()
                except Exception as e:
                    print(f"[WARNING] Could not close viewer: {e}")
            
            # Create a minimal mock viewer object with commands array
            # This allows the environment to run without the actual viewer
            class MockViewer:
                def __init__(self):
                    self.commands = np.zeros(8, dtype=np.float32)
                def render(self):
                    pass  # No-op for headless mode
            
            self.env.viewer = MockViewer()
            print("[INFO] Running in headless mode (no GUI)")
        else:
            print("[INFO] Running with MuJoCo viewer GUI")
        
        # Initialize reward function
        self.reward_fn = reward_fn or RewardFunction()
        
        # Action space
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,), 
            dtype=np.float32,
        )
        
        # Observation space: flattened proprioceptive + demo + privileged + history
        # Size determined by HumanoidEnv.get_observation() output
        # obs = obs_prop + obs_demo + obs_priv + obs_hist
        # obs_prop: n_proprio (93)
        # obs_demo: demo_obs_template.shape[0] (17)
        # obs_priv: n_priv (3)
        # obs_hist: history_len * n_proprio (10 * 93 = 930)
        obs_size = (
            self.env.n_proprio +
            self.env.demo_obs_template.shape[0] +
            self.env.n_priv +
            self.env.history_len * self.env.n_proprio
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, seed = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo simulation to initial keyframe (home position)
        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)
        mujoco.mj_step(self.env.model, self.env.data)
        
        # Reset internal state variables
        self.env.last_action = np.zeros(self.env.nj, dtype=np.float32)
        self.env.arm_action = self.env.default_dof_pos[15:].copy()
        self.env.prev_arm_action = self.env.default_dof_pos[15:].copy()
        self.env.arm_blend = 0.0
        self.env.toggle_arm = False
        self.env.target_yaw = 0.0
        self.env._in_place_stand_flag = True
        self.env.gait_cycle = np.array([0.25, 0.25])
        
        # Reset viewer commands to zero
        self.env.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Reset history buffers with correct number of zero entries
        # get_observation() reads history first, then appends new obs
        self.env.proprio_history_buf = deque(maxlen=self.env.history_len)
        self.env.extra_history_buf = deque(maxlen=self.env.extra_history_len)
        for i in range(self.env.history_len):
            self.env.proprio_history_buf.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        for i in range(self.env.extra_history_len):
            self.env.extra_history_buf.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        
        # Extract initial observation after reset
        self.env.extract_data()
        obs = self.env.get_observation()
        
        # Reset settling counter
        self.steps_since_reset = 0
        
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_time = 0.0  # Track elapsed simulation time
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of environment dynamics.
        
        Args:
            action: Action vector in [-1, 1]^8 for [vx, vy, yaw_rate, height, torso_yaw, torso_pitch, torso_roll, arm_control_flag].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Increment step counter
        self.steps_since_reset += 1
        
        # During settling period (first 5 seconds), zero out all velocity commands
        # This allows robot to settle from spawning in air
        # TODO: figure out settling
        if self.steps_since_reset <= self.settling_steps:
            vx_cmd = 0.0
            vy_cmd = 0.0
            yaw_cmd = 0.0
            height_cmd = 0.0
            torso_yaw_cmd = 0.0
            torso_pitch_cmd = 0.0
            torso_roll_cmd = 0.0
            arm_control_flag = False
        else:
            # Denormalize action from [-1, 1] to actual command range
            # Scale factors chosen to match play_amo.py ranges
            vx_cmd = np.clip(action[0] * 0.5, -0.5, 0.5)   # vx: forward/backward velocity
            vy_cmd = np.clip(action[1] * 0.5, -0.5, 0.5)   # vy: lateral velocity  
            yaw_cmd = np.clip(action[2] * 0.5, -0.5, 0.5)  # yaw_rate: rotation
            height_cmd = np.clip(action[3] * 0.2, -0.2, 0.2)  # height offset
            torso_yaw_cmd = np.clip(action[4] * 0.5, -0.5, 0.5)  # torso yaw
            torso_pitch_cmd = np.clip(action[5] * 0.5, -0.5, 0.5)  # torso pitch
            torso_roll_cmd = np.clip(action[6] * 0.5, -0.5, 0.5)  # torso roll
            arm_control_flag = action[7] >= 0.0  # arm control flag
        
        # Apply commands to viewer (input to AMO policy)
        # AMO policy expects commands in specific order
        self.env.viewer.commands[0] = vx_cmd   # vx
        self.env.viewer.commands[1] = yaw_cmd  # yaw
        self.env.viewer.commands[2] = vy_cmd   # vy
        self.env.viewer.commands[3] = height_cmd  # height offset
        self.env.viewer.commands[4] = torso_yaw_cmd  # torso_yaw
        self.env.viewer.commands[5] = torso_pitch_cmd  # torso_pitch
        self.env.viewer.commands[6] = torso_roll_cmd  # torso_roll
        self.env.viewer.commands[7] = arm_control_flag  # arm_flag

        self.env.extract_data()
        obs = self.env.get_observation()  # This updates self.env._in_place_stand_flag inside!
        
        # AMO policy computes low-level joint actions from observations
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            extra_hist = torch.tensor(
                np.array(self.env.extra_history_buf).flatten().copy(),
                dtype=torch.float,
            ).view(1, -1).to(self.device)
            raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        
        raw_action = np.clip(raw_action, -40.0, 40.0)
        self.env.last_action = np.concatenate([
            raw_action.copy(),
            (self.env.dof_pos - self.env.default_dof_pos)[15:] / self.action_scale
        ])
        scaled_actions = raw_action * self.action_scale
        
        pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.env.default_dof_pos
        pd_target[15:] = (
            (1 - self.env.arm_blend) * self.env.prev_arm_action
            + self.env.arm_blend * self.env.arm_action
        )
        self.env.arm_blend = min(1.0, self.env.arm_blend + 0.01)
        
        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0
        )
        
        if self.env._in_place_stand_flag and ((np.abs(self.env.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.env.gait_cycle[1] - 0.25) < 0.05)):
            self.env.gait_cycle = np.array([0.25, 0.25])
        if (not self.env._in_place_stand_flag) and ((np.abs(self.env.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.env.gait_cycle[1] - 0.25) < 0.05)):
            self.env.gait_cycle = np.array([0.25, 0.75])

        for _ in range(self.env.sim_decimation):
            # Compute torques based on current joint state and fixed PD target
            torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            self.env.data.ctrl = torque
            
            # Step simulation
            mujoco.mj_step(self.env.model, self.env.data)
            
            # Update joint states for next torque calculation
            self.env.extract_data()
            
            # Render if viewer is active (not headless mode)
            if not self.headless and hasattr(self.env, 'viewer') and self.env.viewer is not None:
                if hasattr(self.env.viewer, 'render'):
                    self.env.viewer.cam.lookat = self.env.data.qpos.astype(np.float32)[:3]
                    self.env.viewer.render()
        
        # Extract final observation after all simulation steps
        observation = self.env.get_observation()
        
        # Compute reward
        rpy = self._quat_to_euler(self.env.quat)
        reward = self.reward_fn.compute_reward(
            position=self.env.data.qpos[:3],
            rpy=rpy,
            ang_vel=self.env.ang_vel,
            action=action,
        )
        
        self.episode_steps += 1
        self.episode_return += reward
        # Track elapsed simulation time (control_dt = 0.02s at 20Hz)
        self.episode_time += self.env.control_dt
        
        # Check termination conditions
        # Terminate if robot falls (height too low, excessive tilt)
        torso_height = self.env.data.qpos[2]
        roll = rpy[0]
        pitch = rpy[1]
        
        # Fall detection: check multiple conditions
        height_fail = torso_height < self.min_height
        roll_fail = abs(roll) > self.max_roll
        pitch_fail = abs(pitch) > self.max_pitch
        
        # Goal achievement: check distance to target
        current_xy = self.env.data.qpos[:2]  # x, y position
        target_xy = self.reward_fn.target_position
        distance_to_goal = np.linalg.norm(current_xy - target_xy)
        goal_achieved = distance_to_goal < self.goal_distance
        
        # Time limit: check if episode has exceeded max time
        time_exceeded = self.episode_time >= self.max_episode_time
        
        terminated = height_fail or roll_fail or pitch_fail or goal_achieved or time_exceeded
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Provide detailed info about why episode ended
        termination_reason = None
        if terminated:
            if goal_achieved:
                termination_reason = f"goal_achieved(dist={distance_to_goal:.3f}<{self.goal_distance})"
            elif time_exceeded:
                termination_reason = f"time_exceeded({self.episode_time:.1f}s>={self.max_episode_time}s)"
            elif height_fail:
                termination_reason = f"height_too_low({torso_height:.3f}<{self.min_height})"
            elif roll_fail:
                termination_reason = f"excessive_roll({abs(roll):.3f}>{self.max_roll})"
            elif pitch_fail:
                termination_reason = f"excessive_pitch({abs(pitch):.3f}>{self.max_pitch})"
        
        info = {
            "episode_return": self.episode_return,
            "episode_steps": self.episode_steps,
            "episode_time": self.episode_time,
            "torso_height": torso_height,
            "roll": roll,
            "pitch": pitch,
            "distance_to_goal": distance_to_goal,
            "goal_achieved": goal_achieved,
            "terminated": terminated,
            "termination_reason": termination_reason,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            quat: Quaternion [qw, qx, qy, qz].
        
        Returns:
            np.ndarray: Euler angles [roll, pitch, yaw] in radians.
        """
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, 'env') and hasattr(self.env, 'viewer'):
            if self.env.viewer is not None and hasattr(self.env.viewer, 'close'):
                try:
                    self.env.viewer.close()
                except Exception:
                    pass  # Ignore errors when closing
        super().close()
    
    def render(self, mode: str = "human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode (default "human").
        """
        # Render via the MuJoCo viewer if available
        if not self.headless and hasattr(self.env, 'viewer') and self.env.viewer is not None:
            if hasattr(self.env.viewer, 'render'):
                # Update camera to follow robot
                self.env.viewer.cam.lookat = self.env.data.qpos.astype(np.float32)[:3]
                self.env.viewer.render()
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"G1RLEnv(robot={self.robot_type}, "
            f"action_space={self.action_space.shape}, "
            f"obs_space={self.observation_space.shape})"
        )
