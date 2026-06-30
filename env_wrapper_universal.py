"""
Universal Manipulation Environment for G1 Humanoid.

This environment trains a SINGLE policy that can reach/manipulate ANY object
by randomizing the target during training. The target position is included
in the observation, so the policy learns to reach toward any given target.

Supports:
- Table objects (screwdriver, wrench, block, etc.) - grasping
- Buttons (red, green, yellow, blue) - pressing

Usage:
    python train_universal.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any, Optional, List

from play_amo import HumanoidEnv


# All manipulable objects and their properties
MANIPULATION_TARGETS = {
    # Table objects (grasping)
    "screwdriver": {
        "body": "screwdriver",
        "type": "grasp",
        "position": np.array([1.1, 1.35, 0.74]),
    },
    "wrench": {
        "body": "wrench",
        "type": "grasp",
        "position": np.array([1.25, -0.25, 0.74]),
    },
    "block_cube": {
        "body": "block_cube",
        "type": "grasp",
        "position": np.array([1.5, -0.15, 0.75]),
    },
    "small_box": {
        "body": "small_box",
        "type": "grasp",
        "position": np.array([1.3, -0.3, 0.74]),
    },
    "battery_pack": {
        "body": "battery_pack",
        "type": "grasp",
        "position": np.array([1.55, 1.5, 0.74]),
    },
    "purple_object": {
        "body": "right_table_item2",
        "type": "grasp",
        "position": np.array([1.4, 1.45, 0.74]),
    },
    # Buttons (pressing)
    "button_red": {
        "body": "push_button_red",
        "joint": "push_button_red_joint",
        "type": "press",
        "position": np.array([-0.45, -1.85, 1.0]),
    },
    "button_green": {
        "body": "push_button_green",
        "joint": "push_button_green_joint",
        "type": "press",
        "position": np.array([-0.15, -1.85, 1.0]),
    },
    "button_yellow": {
        "body": "push_button_yellow",
        "joint": "push_button_yellow_joint",
        "type": "press",
        "position": np.array([0.15, -1.85, 1.0]),
    },
    "button_blue": {
        "body": "push_button_blue",
        "joint": "push_button_blue_joint",
        "type": "press",
        "position": np.array([0.45, -1.85, 1.0]),
    },
}


class UniversalManipulationEnv(gym.Env):
    """
    Universal environment that trains ONE policy for ALL manipulation tasks.
    
    Each episode randomly selects a target object. The policy receives the
    target position in its observation, learning to reach toward any target.
    
    Observation:
        - arm joint positions (8)
        - arm joint velocities (8)
        - left hand position (3)
        - right hand position (3)
        - target position (3)
        - left hand to target (3)
        - right hand to target (3)
        - task type one-hot (2): [is_grasp, is_press]
        - task-specific state (2): [is_grasped/button_displacement, lift_amount/0]
    Total: 35
    
    Action: 8 arm joint position offsets in [-1, 1]
    """
    
    def __init__(
        self,
        allowed_targets: List[str] = None,  # None = all targets
        freeze_arm: str = "left",
        max_episode_steps: int = 200,
        headless: bool = True,
        device: str = None,
        randomize_robot_position: bool = True,
    ):
        """
        Initialize universal manipulation environment.
        
        Args:
            allowed_targets: List of target names to train on (None = all)
            freeze_arm: Which arm to freeze ("left", "right", "none")
            max_episode_steps: Max steps per episode
            headless: Run without visualization
            device: Torch device
            randomize_robot_position: Position robot near target each episode
        """
        super().__init__()
        
        self.freeze_arm = freeze_arm.lower()
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.randomize_robot_position = randomize_robot_position
        
        # Filter targets
        if allowed_targets is None:
            self.allowed_targets = list(MANIPULATION_TARGETS.keys())
        else:
            self.allowed_targets = [t for t in allowed_targets if t in MANIPULATION_TARGETS]
        
        print(f"[ENV] Training on {len(self.allowed_targets)} targets: {self.allowed_targets}")
        
        # Load AMO policy for legs
        self.policy_jit = torch.jit.load("amo_jit.pt", map_location=self.device)
        
        # Create HumanoidEnv with headless flag
        self.env = HumanoidEnv(
            policy_jit=self.policy_jit,
            robot_type="g1",
            device=self.device,
            headless=headless,
        )
        
        # Cache body/joint IDs
        self._cache_ids()
        
        # Get hand IDs
        self.left_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_palm_link'
        )
        self.right_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_palm_link'
        )
        
        # Arm configuration
        self.num_arm_joints = 8
        self.arm_joint_start = 15
        self.arm_joint_end = 23
        self.action_scale = 0.25
        
        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_arm_joints,),
            dtype=np.float32,
        )
        
        # Observation space: 35 dims
        obs_dim = 8 + 8 + 3 + 3 + 3 + 3 + 3 + 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Episode state
        self.episode_steps = 0
        self.current_target_name = None
        self.current_target_info = None
        self.task_success = False
        
        # Grasp state
        self.object_grasped = False
        self.grasp_distance = 0.08
        
    def _cache_ids(self):
        """Cache MuJoCo body and joint IDs for all targets."""
        self.target_body_ids = {}
        self.target_joint_ids = {}
        
        for name, info in MANIPULATION_TARGETS.items():
            body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, info["body"]
            )
            self.target_body_ids[name] = body_id
            
            if "joint" in info:
                joint_id = mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_JOINT, info["joint"]
                )
                self.target_joint_ids[name] = joint_id
    
    def _get_robot_position_for_target(self, target_name: str) -> np.ndarray:
        """Get appropriate robot starting position for a target."""
        info = MANIPULATION_TARGETS[target_name]
        target_pos = info["position"]
        
        if info["type"] == "press":
            # Buttons: stand 0.4m in front (higher y)
            return np.array([target_pos[0], target_pos[1] + 0.4, 0.793])
        else:
            # Table objects: stand 0.4m back (lower x)
            return np.array([target_pos[0] - 0.4, target_pos[1], 0.793])
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset and select a random target."""
        if seed is not None:
            np.random.seed(seed)
        
        # Select random target
        self.current_target_name = np.random.choice(self.allowed_targets)
        self.current_target_info = MANIPULATION_TARGETS[self.current_target_name]
        
        # Reset MuJoCo
        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)
        
        # Position robot near target
        if self.randomize_robot_position:
            robot_pos = self._get_robot_position_for_target(self.current_target_name)
            robot_qpos_start = self.env.model.jnt_qposadr[mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]  # robust to interactive-obj count
            self.env.data.qpos[robot_qpos_start:robot_qpos_start+3] = robot_pos
            
            # Face toward target
            target_pos = self.current_target_info["position"]
            yaw = np.arctan2(
                target_pos[1] - robot_pos[1],
                target_pos[0] - robot_pos[0]
            )
            # Convert yaw to quaternion (rotation around Z)
            quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7] = quat
        
        mujoco.mj_step(self.env.model, self.env.data)
        
        # Reset env state
        self.env._extract_state()
        self.env.last_action = np.zeros(self.env.num_dofs, dtype=np.float32)
        self.env.arm_action = self.env.default_dof_pos[15:].copy()
        self.env.prev_arm_action = self.env.default_dof_pos[15:].copy()
        self.env.arm_blend = 0.0
        self.env._in_place_stand = True
        self.env.gait_cycle = np.array([0.25, 0.25])
        self.env.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Reset history
        self.env.proprio_history = deque(maxlen=self.env.history_len)
        self.env.extra_history = deque(maxlen=self.env.extra_history_len)
        for _ in range(self.env.history_len):
            self.env.proprio_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        for _ in range(self.env.extra_history_len):
            self.env.extra_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        
        # Reset episode state
        self.episode_steps = 0
        self.task_success = False
        self.object_grasped = False
        
        info = {
            "target": self.current_target_name,
            "task_type": self.current_target_info["type"],
        }
        
        return self._get_obs(), info
    
    def _get_obs(self) -> np.ndarray:
        """Get observation including target information."""
        # Arm state
        arm_pos = self.env.dof_pos[self.arm_joint_start:self.arm_joint_end]
        arm_vel = self.env.dof_vel[self.arm_joint_start:self.arm_joint_end]
        
        # Hand positions
        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self.env.data.xpos[self.right_hand_id]
        
        # Target position (current, not initial)
        target_body_id = self.target_body_ids[self.current_target_name]
        target_pos = self.env.data.xpos[target_body_id]
        
        # Vectors to target
        left_to_target = target_pos - left_hand_pos
        right_to_target = target_pos - right_hand_pos
        
        # Task type one-hot: [is_grasp, is_press]
        is_grasp = 1.0 if self.current_target_info["type"] == "grasp" else 0.0
        is_press = 1.0 if self.current_target_info["type"] == "press" else 0.0
        task_type = np.array([is_grasp, is_press])
        
        # Task-specific state
        if self.current_target_info["type"] == "press":
            joint_id = self.target_joint_ids[self.current_target_name]
            button_disp = self.env.data.qpos[joint_id]
            task_state = np.array([button_disp, 0.0])
        else:
            # Grasp: [is_grasped, lift_amount]
            grasped = 1.0 if self.object_grasped else 0.0
            lift = max(0, target_pos[2] - 0.74) if self.object_grasped else 0.0
            task_state = np.array([grasped, lift])
        
        obs = np.concatenate([
            arm_pos,            # 8
            arm_vel * 0.1,      # 8
            left_hand_pos,      # 3
            right_hand_pos,     # 3
            target_pos,         # 3
            left_to_target,     # 3
            right_to_target,    # 3
            task_type,          # 2
            task_state,         # 2
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self.episode_steps += 1
        
        # Process action
        action = np.array(action, dtype=np.float32)
        if self.freeze_arm == "left":
            action[:4] = 0.0
        elif self.freeze_arm == "right":
            action[4:] = 0.0
        
        scaled_arm_action = action * self.action_scale
        
        # Leg control (stand still)
        self.env.viewer.commands[:] = 0.0
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
        scaled_leg_action = leg_action * self.action_scale
        
        # Combine actions
        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] += scaled_leg_action
        pd_target[15:] += scaled_arm_action
        
        # Step simulation
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            self.env.data.ctrl = torque
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()
        
        # Compute reward
        reward, info = self._compute_reward(action)
        
        # Check termination
        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        terminated = robot_pos[2] < 0.4  # Fell
        truncated = self.episode_steps >= self.max_episode_steps
        
        info["target"] = self.current_target_name
        info["task_type"] = self.current_target_info["type"]
        info["success"] = self.task_success
        
        # Render
        if not self.headless and hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.render()
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Compute reward based on task type."""
        info = {}
        
        # Get positions
        left_hand = self.env.data.xpos[self.left_hand_id]
        right_hand = self.env.data.xpos[self.right_hand_id]
        target_body_id = self.target_body_ids[self.current_target_name]
        target_pos = self.env.data.xpos[target_body_id]
        
        # Distance to target (use closest hand)
        left_dist = np.linalg.norm(left_hand - target_pos)
        right_dist = np.linalg.norm(right_hand - target_pos)
        min_dist = min(left_dist, right_dist)
        info["hand_distance"] = min_dist
        
        # Base reward: proximity
        r_proximity = np.exp(-3.0 * min_dist)
        
        # Task-specific reward
        if self.current_target_info["type"] == "press":
            # Button pressing
            joint_id = self.target_joint_ids[self.current_target_name]
            button_disp = self.env.data.qpos[joint_id]
            info["button_displacement"] = button_disp
            
            r_press = button_disp * 100.0  # Reward displacement
            if button_disp > 0.02:  # Pressed!
                r_press += 50.0
                self.task_success = True
            
            reward = 0.5 + 5.0 * r_proximity + r_press
            
        else:
            # Grasping
            r_grasp = 0.0
            if min_dist < self.grasp_distance and not self.object_grasped:
                self.object_grasped = True
                r_grasp = 20.0
                print(f"    [GRASP] Grasped {self.current_target_name}!")
            
            # Lift reward
            r_lift = 0.0
            if self.object_grasped:
                lift = target_pos[2] - 0.74
                if lift > 0.05:
                    r_lift = 30.0 * lift
                    self.task_success = True
            
            reward = 0.5 + 5.0 * r_proximity + r_grasp + r_lift
        
        # Action penalty
        reward -= 0.001 * np.linalg.norm(action) ** 2
        
        return reward, info
    
    def close(self):
        if hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.close()
