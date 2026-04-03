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
        max_episode_steps: int = 500,
        reward_fn: RewardFunction = None,
        headless: bool = True,
        min_height: float = 0.4,
        max_roll: float = 0.8,
        max_pitch: float = 0.8,
        goal_distance: float = 0.1,
        max_episode_time: float = 10.0,
        verbose: int = 0,
        freeze_arm: str = "none",
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
        # freeze_arm can be 'none', 'left', or 'right' to freeze that arm's joints
        self.freeze_arm = freeze_arm.lower() if isinstance(freeze_arm, str) else "none"
        
        # Termination thresholds
        self.min_height = min_height
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.goal_distance = goal_distance
        self.max_episode_time = max_episode_time
        
        # Print termination settings for debugging
        print(f"[ENV] Termination settings: max_steps={max_episode_steps}, max_time={max_episode_time}s")
        if self.freeze_arm in ("left", "right"):
            print(f"[ENV] Freezing '{self.freeze_arm}' arm - RL will only control the other arm")
        
        # Settling period configuration
        # Reduced to 10 steps (~0.2 seconds) to start moving faster
        self.settling_steps = 10  
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
            headless=self.headless,
        )
        
        # Headless mode is now handled by HumanoidEnv directly
        if self.headless:
            print("[INFO] Running in headless mode (no GUI)")
        else:
            print("[INFO] Running with MuJoCo viewer GUI")
        
        # Cache the pelvis body ID for getting robot position
        self.pelvis_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis'
        )
        
        # Cache hand body IDs for reaching
        self.left_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand'
        )
        self.right_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand'
        )
        
        # Target object position (screwdriver)
        self.target_object_pos = np.array([1.2, 1.35, 0.74])
        
        # Cache table geom IDs for collision detection
        self.table_geom_ids = []
        for i in range(self.env.model.ngeom):
            name = self.env.model.geom(i).name
            if 'table' in name.lower():
                self.table_geom_ids.append(i)
        # Cache object geom IDs (screwdriver etc.) to exclude from self-collision checks
        self.object_geom_ids = []
        for i in range(self.env.model.ngeom):
            name = self.env.model.geom(i).name
            if 'screw' in name.lower() or 'wrench' in name.lower() or 'battery' in name.lower():
                self.object_geom_ids.append(i)
        
        # Arm joint indices and limits
        # Joints: [L_pitch, L_roll, L_yaw, L_elbow, R_pitch, R_roll, R_yaw, R_elbow]
        self.arm_joint_start = 15
        self.arm_joint_end = 23
        self.num_arm_joints = 8
        self.default_arm_pos = self.env.default_dof_pos[self.arm_joint_start:self.arm_joint_end].copy()
        
        # Updated limits based on actual XML ranges
        # Shoulder pitch moves arm forward/back - needs large positive for forward reach
        # Shoulder roll moves arm in/out from body
        # Elbow bends the arm
        self.arm_pos_low = np.array([
            -2.0,  # Left shoulder pitch (back)
            -1.5,  # Left shoulder roll (in toward body)
            -2.0,  # Left shoulder yaw
            -0.5,  # Left elbow (straight-ish)
            -2.0,  # Right shoulder pitch (back)
            -1.5,  # Right shoulder roll (note: inverted in XML, -2.25 to 1.58)
            -2.0,  # Right shoulder yaw  
            -0.5,  # Right elbow (straight-ish)
        ])
        self.arm_pos_high = np.array([
            2.5,   # Left shoulder pitch (forward) - higher to reach forward
            2.0,   # Left shoulder roll (out from body)
            2.0,   # Left shoulder yaw
            2.0,   # Left elbow (bent)
            2.5,   # Right shoulder pitch (forward) - higher to reach forward
            1.5,   # Right shoulder roll
            2.0,   # Right shoulder yaw
            2.0,   # Right elbow (bent)
        ])
        
        # Initialize reward function
        self.reward_fn = reward_fn or RewardFunction()
        
        # Action space: 8 arm joint position offsets in [-1, 1]
        # NO locomotion - only arm control for grasping
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_arm_joints,),
            dtype=np.float32,
        )
        
        # Simplified observation space for grasping:
        # - arm joint positions (8)
        # - arm joint velocities (8)
        # - left hand position (3)
        # - right hand position (3)
        # - target object position (3)
        # - vector from left hand to target (3)
        # - vector from right hand to target (3)
        # - grasp state (2): [is_grasped, lift_amount]
        # Total: 33
        obs_dim = 8 + 8 + 3 + 3 + 3 + 3 + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self.episode_steps = 0
        self.episode_return = 0.0
        self.best_hand_distance = float('inf')
        self.prev_min_dist = None
        
        # === MAGNETIC GRASP SETUP ===
        # Get screwdriver body ID
        self.screwdriver_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'screwdriver'
        )
        
        # Grasp parameters
        self.grasp_distance = 0.08  # 8cm - increased for easier grasp
        self.lift_height = 0.85  # Height above table to consider "lifted"
        self.object_grasped = False
        self.grasping_hand = None  # 'left' or 'right'
        self.grasp_offset = None  # Offset from hand to object when grasped
        
        # Anti-hovering: track time spent close without grasping
        self.time_close_without_grasp = 0.0
        self.best_distance_ever = float('inf')  # Track best distance achieved
        
        # Initial screwdriver position (for detecting lift)
        self.initial_object_height = 0.74

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
        self.env.last_action = np.zeros(self.env.num_dofs, dtype=np.float32)
        self.env.arm_action = self.env.default_dof_pos[15:].copy()
        self.env.prev_arm_action = self.env.default_dof_pos[15:].copy()
        self.env.arm_blend = 0.0
        self.env.toggle_arm = False
        self.env.target_yaw = 0.0
        self.env._in_place_stand = True
        self.env.gait_cycle = np.array([0.25, 0.25])
        
        # Reset viewer commands to zero
        self.env.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Reset history buffers with correct number of zero entries
        # get_observation() reads history first, then appends new obs
        self.env.proprio_history = deque(maxlen=self.env.history_len)
        self.env.extra_history = deque(maxlen=self.env.extra_history_len)
        for i in range(self.env.history_len):
            self.env.proprio_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        for i in range(self.env.extra_history_len):
            self.env.extra_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        
        # Extract initial observation after reset
        self.env._extract_state()
        
        # Reset settling counter
        self.steps_since_reset = 0
        
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_time = 0.0
        self.best_hand_distance = float('inf')
        self.prev_min_dist = None
        
        # Reset grasp state
        self.object_grasped = False
        self.grasping_hand = None
        self.grasp_offset = None
        
        # Reset anti-hovering tracking
        self.time_close_without_grasp = 0.0
        self.best_distance_ever = float('inf')
        
        # Reset target position to actual screwdriver position after simulation reset
        self.target_object_pos = self.env.data.xpos[self.screwdriver_body_id].copy()
        
        # Reset reward function state
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()
        
        return self._get_grasp_obs(), {}
    
    def _get_grasp_obs(self) -> np.ndarray:
        """Get simplified observation for grasping task."""
        # Arm joint positions and velocities
        arm_pos = self.env.dof_pos[self.arm_joint_start:self.arm_joint_end]
        arm_vel = self.env.dof_vel[self.arm_joint_start:self.arm_joint_end]
        
        # Hand positions
        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self.env.data.xpos[self.right_hand_id]
        
        # Vectors to target
        left_to_target = self.target_object_pos - left_hand_pos
        right_to_target = self.target_object_pos - right_hand_pos
        
        # Grasp state: [is_grasped, object_height_above_table]
        grasp_state = np.array([
            1.0 if self.object_grasped else 0.0,
            self.target_object_pos[2] - self.initial_object_height,  # lift amount
        ])
        
        obs = np.concatenate([
            arm_pos,                    # 8
            arm_vel * 0.1,              # 8 (scaled down)
            left_hand_pos,              # 3
            right_hand_pos,             # 3
            self.target_object_pos,     # 3
            left_to_target,             # 3
            right_to_target,            # 3
            grasp_state,                # 2
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of environment dynamics for GRASPING.
        
        Args:
            action: 8 arm joint position offsets in [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.episode_steps += 1
        self.steps_since_reset += 1
        
        # === ARM CONTROL FROM RL ACTION ===
        # Scale action to cover full joint range
        # action in [-1, 1] maps to [arm_pos_low, arm_pos_high]
        arm_range = (self.arm_pos_high - self.arm_pos_low) / 2.0
        arm_center = (self.arm_pos_high + self.arm_pos_low) / 2.0
        target_arm_pos = arm_center + action * arm_range
        target_arm_pos = np.clip(target_arm_pos, self.arm_pos_low, self.arm_pos_high)
        
        # === LEG CONTROL FROM AMO (standing still) ===
        # Zero velocity commands = stand in place
        self.env.viewer.commands[:] = 0.0
        self.env._extract_state()
        amo_obs = self.env._compute_observation()
        
        # AMO policy computes leg joint actions for standing
        obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(
                np.array(self.env.extra_history).flatten().copy(),
                dtype=torch.float,
            ).view(1, -1).to(self.device)
            leg_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        
        leg_action = np.clip(leg_action, -40.0, 40.0)
        scaled_leg_action = leg_action * self.action_scale
        
        # Update last_action buffer (needed by _compute_observation)
        self.env.last_action = np.concatenate([
            leg_action.copy(),
            (self.env.dof_pos[15:] - self.env.default_dof_pos[15:]) / self.action_scale
        ])
        
        # Build full PD target: legs from AMO, arms from RL
        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] = scaled_leg_action + self.env.default_dof_pos[:15]  # Legs
        # Optionally freeze one arm: replace that arm's targets with defaults
        target_arm_pos_mod = target_arm_pos.copy()
        if self.freeze_arm == 'left':
            target_arm_pos_mod[:4] = self.default_arm_pos[:4]
        elif self.freeze_arm == 'right':
            target_arm_pos_mod[4:8] = self.default_arm_pos[4:8]
        pd_target[15:23] = target_arm_pos_mod  # Arms from RL action (possibly modified)
        
        # Update gait cycle (needed for standing balance)
        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0
        )
        if self.env._in_place_stand:
            if np.any(np.abs(self.env.gait_cycle - 0.25) < 0.05):
                self.env.gait_cycle = np.array([0.25, 0.25])
        
        # Simulate physics
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            self.env.data.ctrl = torque
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()
            
            # Render if viewer is active
            if not self.headless and hasattr(self.env.viewer, 'render'):
                self.env.viewer.cam.lookat = self.env.data.xpos[self.pelvis_body_id].astype(np.float32)
                self.env.viewer.render()
        
        # Get positions
        robot_pos = self.env.data.xpos[self.pelvis_body_id]
        left_hand_pos = self.env.data.xpos[self.left_hand_id].copy()
        right_hand_pos = self.env.data.xpos[self.right_hand_id].copy()
        screwdriver_pos = self.env.data.xpos[self.screwdriver_body_id].copy()
        
        # === MAGNETIC GRASP LOGIC ===
        # Only allow grasp if hand is moving slowly (controlled approach)
        arm_velocities = self.env.dof_vel[15:23]  # Arm joint velocities
        arm_speed = np.sqrt(np.sum(arm_velocities ** 2))
        
        # Compute hand velocities (approximate from position change)
        if not hasattr(self, 'prev_left_hand_pos'):
            self.prev_left_hand_pos = left_hand_pos.copy()
            self.prev_right_hand_pos = right_hand_pos.copy()
        
        left_hand_vel = np.linalg.norm(left_hand_pos - self.prev_left_hand_pos) / self.env.control_dt
        right_hand_vel = np.linalg.norm(right_hand_pos - self.prev_right_hand_pos) / self.env.control_dt
        
        # Grasp speed threshold - hand must be moving slowly for magnetic attach
        GRASP_SPEED_THRESHOLD = 0.5  # m/s - must be moving slower than this to grasp
        
        if not self.object_grasped:
            # Check if either hand is close enough to grasp
            left_dist_to_obj = np.linalg.norm(left_hand_pos - screwdriver_pos)
            right_dist_to_obj = np.linalg.norm(right_hand_pos - screwdriver_pos)
            
            # Right hand grasp - must be close AND moving slowly
            if right_dist_to_obj < self.grasp_distance and right_hand_vel < GRASP_SPEED_THRESHOLD:
                self.object_grasped = True
                self.grasping_hand = 'right'
                self.grasp_offset = screwdriver_pos - right_hand_pos
                print(f"[GRASP] Right hand grabbed screwdriver at step {self.episode_steps}! (speed: {right_hand_vel:.2f} m/s)")
            # Left hand grasp - must be close AND moving slowly
            elif left_dist_to_obj < self.grasp_distance and left_hand_vel < GRASP_SPEED_THRESHOLD:
                self.object_grasped = True
                self.grasping_hand = 'left'
                self.grasp_offset = screwdriver_pos - left_hand_pos
                print(f"[GRASP] Left hand grabbed screwdriver at step {self.episode_steps}! (speed: {left_hand_vel:.2f} m/s)")
            # Log if close but too fast
            elif right_dist_to_obj < self.grasp_distance:
                if self.episode_steps % 50 == 0:
                    print(f"[GRASP FAIL] Right hand close ({right_dist_to_obj:.3f}m) but too fast ({right_hand_vel:.2f} m/s > {GRASP_SPEED_THRESHOLD})")
            elif left_dist_to_obj < self.grasp_distance:
                if self.episode_steps % 50 == 0:
                    print(f"[GRASP FAIL] Left hand close ({left_dist_to_obj:.3f}m) but too fast ({left_hand_vel:.2f} m/s > {GRASP_SPEED_THRESHOLD})")
        
        # Update previous hand positions
        self.prev_left_hand_pos = left_hand_pos.copy()
        self.prev_right_hand_pos = right_hand_pos.copy()
        
        # If object is grasped, move it with the hand
        if self.object_grasped:
            if self.grasping_hand == 'right':
                new_obj_pos = right_hand_pos + self.grasp_offset
            else:
                new_obj_pos = left_hand_pos + self.grasp_offset
            
            # Update screwdriver position in simulation
            # Find the screwdriver's qpos index (it's a freejoint)
            screwdriver_qpos_start = self.env.model.jnt_qposadr[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'screwdriver_joint')
            ]
            self.env.data.qpos[screwdriver_qpos_start:screwdriver_qpos_start+3] = new_obj_pos
            # Zero out velocity so it doesn't fly away
            screwdriver_qvel_start = self.env.model.jnt_dofadr[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'screwdriver_joint')
            ]
            self.env.data.qvel[screwdriver_qvel_start:screwdriver_qvel_start+6] = 0
            
            # Update the position for reward calculation
            screwdriver_pos = new_obj_pos.copy()
        
        # Update target position to actual object position
        self.target_object_pos = screwdriver_pos.copy()
        
        # Compute hand distances to target (for reaching reward before grasp)
        left_dist = np.linalg.norm(left_hand_pos - self.target_object_pos)
        right_dist = np.linalg.norm(right_hand_pos - self.target_object_pos)
        min_dist = min(left_dist, right_dist)
        self.best_hand_distance = min(self.best_hand_distance, min_dist)
        
        # Determine which hand is closer
        if right_dist < left_dist:
            active_hand_pos = right_hand_pos
            active_hand_id = self.right_hand_id
        else:
            active_hand_pos = left_hand_pos
            active_hand_id = self.left_hand_id
        
        # === REDESIGNED GRASPING REWARD (Anti-Hovering) ===
        
        # Get hand velocity for speed-based rewards
        if right_dist < left_dist:
            active_hand_vel = right_hand_vel
        else:
            active_hand_vel = left_hand_vel
        
        # Track best distance ever achieved (for one-time bonus)
        new_best = min_dist < self.best_distance_ever
        if new_best:
            self.best_distance_ever = min_dist
        
        # Track time spent close without grasping (anti-hovering)
        if min_dist < 0.15 and not self.object_grasped:
            self.time_close_without_grasp += self.env.control_dt
        
        # === 1. SPARSE MILESTONE REWARDS (one-time bonuses for reaching new distances) ===
        # These only trigger ONCE when a new best distance is achieved
        milestone_bonus = 0.0
        if new_best and not self.object_grasped:
            if min_dist < 0.30:
                milestone_bonus = 5.0
            if min_dist < 0.20:
                milestone_bonus = 10.0
            if min_dist < 0.15:
                milestone_bonus = 20.0
            if min_dist < 0.10:
                milestone_bonus = 40.0
            if min_dist < 0.08:  # Within grasp range!
                milestone_bonus = 80.0
        
        # === 2. SMALL CONTINUOUS PROXIMITY (reduced to prevent hovering) ===
        proximity_reward = 0.0
        if not self.object_grasped:
            # Much smaller continuous reward - just for gradient
            proximity_reward = 1.0 * np.exp(-3.0 * min_dist)  # Reduced from 5.0 * exp(-5.0)
        
        # === 3. PROGRESS REWARD (only for sustained progress, not oscillation) ===
        progress_reward = 0.0
        if not self.object_grasped and self.prev_min_dist is not None:
            delta = self.prev_min_dist - min_dist
            # Only reward progress, not oscillation
            if delta > 0.001:  # Must make real progress (1mm)
                progress_reward = delta * 50.0  # Reduced from 150
        self.prev_min_dist = min_dist
        
        # === 4. HOVERING PENALTY (penalize time spent close without grasping) ===
        hovering_penalty = 0.0
        if min_dist < 0.15 and not self.object_grasped:
            # Increasing penalty the longer you hover
            hovering_penalty = 0.5 * self.time_close_without_grasp  # Grows over time!
            # Extra penalty for being very close but not grasping
            if min_dist < 0.1:
                hovering_penalty += 1.0  # Per-step penalty for hovering in grasp zone
        
        # === 5. SPEED CONTROL (simplified) ===
        speed_reward = 0.0
        speed_penalty = 0.0
        grasp_ready_bonus = 0.0
        
        if not self.object_grasped:
            if min_dist < 0.1:
                # Close: reward being slow and ready to grasp
                if active_hand_vel < 0.5:
                    grasp_ready_bonus = 5.0  # Ready to grasp!
                else:
                    speed_penalty = 2.0 * (active_hand_vel - 0.5)  # Too fast
        
        # === 6. GRASP AND LIFT REWARDS (MASSIVE) ===
        grasp_bonus = 0.0
        lift_bonus = 0.0
        object_height = screwdriver_pos[2]
        lift_amount = object_height - self.initial_object_height
        
        if self.object_grasped:
            # HUGE bonus for successful grasp - this is the PRIMARY goal!
            grasp_bonus = 500.0  # Massively increased! This should dominate
            
            # Bonus for lifting (progressive)
            if lift_amount > 0:
                lift_bonus = 100.0 * lift_amount
            
            # Big bonus for lifting above threshold
            if object_height > self.lift_height:
                lift_bonus += 500.0
        
        # === 7. ACTION PENALTY ===
        action_magnitude = np.sqrt(np.sum(action ** 2))
        action_penalty = 0.001 * action_magnitude ** 2
        
        # === 8. ALIVE BONUS (small) ===
        alive_bonus = 0.05  # Reduced
        
        # === 9. COLLISION PENALTIES ===
        table_collisions = self._count_table_collisions()
        table_collision_penalty = 5.0 * table_collisions
        
        self_collisions = self._count_self_collisions()
        self_collision_penalty = 2.0 * self_collisions

        collision_penalty = table_collision_penalty + self_collision_penalty
        table_collision_termination = table_collisions > 10
        
        # === 10. VELOCITY DAMPING ===
        velocity_penalty = 0.0
        if min_dist < 0.15:
            velocity_penalty = 0.1 * arm_speed
        
        # === CALCULATE TOTAL REWARD ===
        reward = (
            alive_bonus 
            + milestone_bonus      # One-time bonuses for new best distances
            + proximity_reward     # Small continuous gradient
            + progress_reward      # Reward for getting closer
            + speed_reward
            + grasp_ready_bonus
            + grasp_bonus          # HUGE reward for grasping
            + lift_bonus 
            - action_penalty 
            - collision_penalty 
            - velocity_penalty
            - speed_penalty
            - hovering_penalty     # Penalty for hovering without grasping
        )
        
        # Extra penalty for table collision
        if table_collisions > 0:
            reward -= 10.0 * table_collisions
        
        self.episode_return += reward
        self.episode_time += self.env.control_dt
        
        # Check termination
        torso_height = robot_pos[2]
        rpy = self._quat_to_euler(self.env.quat)
        
        # Success = lifted object above threshold
        success = self.object_grasped and object_height > self.lift_height
        
        fell = torso_height < self.min_height or abs(rpy[0]) > self.max_roll or abs(rpy[1]) > self.max_pitch
        time_exceeded = self.episode_time >= self.max_episode_time
        timeout = self.episode_steps >= self.max_episode_steps
        
        # Terminate on table collision (after grace period)
        terminated = fell or success or time_exceeded or table_collision_termination
        truncated = timeout
        
        info = {
            "episode_return": self.episode_return,
            "episode_steps": self.episode_steps,
            "episode_time": self.episode_time,
            "torso_height": torso_height,
            "left_hand_dist": left_dist,
            "right_hand_dist": right_dist,
            "min_hand_dist": min_dist,
            "best_hand_dist": self.best_hand_distance,
            "best_distance_ever": self.best_distance_ever,
            "success": success,
            "fell": fell,
            "time_exceeded": time_exceeded,
            "table_collision_termination": table_collision_termination,
            "roll": rpy[0],
            "pitch": rpy[1],
            "table_collisions": table_collisions,
            "self_collisions": self_collisions,
            # Reward components for debugging
            "milestone_bonus": milestone_bonus,
            "proximity_reward": proximity_reward,
            "progress_reward": progress_reward,
            "hovering_penalty": hovering_penalty,
            "time_close_without_grasp": self.time_close_without_grasp,
            "speed_reward": speed_reward,
            "speed_penalty": speed_penalty,
            "grasp_ready_bonus": grasp_ready_bonus,
            "grasp_bonus": grasp_bonus,
            "table_collision_penalty": table_collision_penalty,
            "velocity_penalty": velocity_penalty,
            "active_hand_vel": active_hand_vel,
            # Grasp-related info
            "object_grasped": self.object_grasped,
            "object_height": object_height,
            "lift_amount": lift_amount,
            "grasping_hand": self.grasping_hand or "none",
        }
        
        return self._get_grasp_obs(), reward, terminated, truncated, info
    
    def _count_table_collisions(self) -> int:
        """Check if robot is touching the table (binary: 0 or 1)."""
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            # Check if either geom is a table AND the other is a robot part (not floor)
            if geom1 in self.table_geom_ids or geom2 in self.table_geom_ids:
                # Check that the other geom is not the floor (geom 0)
                other_geom = geom2 if geom1 in self.table_geom_ids else geom1
                if other_geom > 0 and other_geom not in self.table_geom_ids:
                    return 1  # Binary: touching table
        return 0

    def _count_self_collisions(self) -> int:
        """Self-collision detection - DISABLED for now.
        
        Most 'self-collisions' are actually feet touching ground or normal body contacts.
        Disabling to simplify learning.
        """
        return 0  # Disabled - too noisy
    
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
                # Update camera to follow robot pelvis
                self.env.viewer.cam.lookat = self.env.data.xpos[self.pelvis_body_id].astype(np.float32)
                self.env.viewer.render()
    
    def render_frame(self, width: int = 480, height: int = 360) -> np.ndarray:
        """
        Render a frame as RGB array for video recording.
        
        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            
        Returns:
            np.ndarray: RGB image array of shape (height, width, 3).
        """
        # Create renderer if not exists
        if not hasattr(self, '_renderer') or self._renderer is None:
            self._renderer = mujoco.Renderer(self.env.model, height, width)
        
        # Set up camera to show robot and target
        # Camera looks at a point between robot and screwdriver
        robot_pos = self.env.data.xpos[self.pelvis_body_id]
        target_pos = self.target_object_pos
        
        # Look at midpoint between robot and target, slightly elevated
        lookat = np.array([
            (robot_pos[0] + target_pos[0]) / 2,
            (robot_pos[1] + target_pos[1]) / 2,
            0.8  # Look at roughly chest height
        ])
        
        # Camera position: behind and above the robot, looking toward target
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = 2.5  # Distance from lookat point
        cam.azimuth = 135   # Angle around vertical axis (behind-left of robot)
        cam.elevation = -25  # Angle from horizontal (looking down slightly)
        
        # Update scene with custom camera
        self._renderer.update_scene(self.env.data, camera=cam)
        
        # Render and return RGB array
        pixels = self._renderer.render()
        return pixels
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"G1RLEnv(robot={self.robot_type}, "
            f"action_space={self.action_space.shape}, "
            f"obs_space={self.observation_space.shape})"
        )
