"""
Walk and Grasp Demo

This script combines:
1. PID-controlled locomotion (using AMO policy) to walk to an object
2. Trained PPO arm policy to grasp the object

Usage:
    python walk_and_grasp.py
    python walk_and_grasp.py --target screwdriver
    python walk_and_grasp.py --target wrench --model checkpoints/final_model
"""

import argparse
import torch
import numpy as np
import mujoco
from stable_baselines3 import PPO

from play_amo import HumanoidEnv, quat_to_euler
from pid_controller import LocomotionPIDController
from env_wrapper import G1RLEnv
from reward_fn import RewardFunction


# ============================================
# OBJECT POSITIONS (from interactive_objects.xml)
# ============================================
OBJECT_POSITIONS = {
    # Table objects
    "screwdriver":   np.array([1.1, 1.35, 0.74]),
    "battery_pack":  np.array([1.55, 1.5, 0.74]),
    "wrench":        np.array([1.25, -0.25, 0.74]),
    "block_cube":    np.array([1.5, -0.15, 0.75]),
    "small_box":     np.array([1.3, -0.3, 0.74]),
    "purple_object": np.array([1.4, 1.45, 0.74]),
    
    # Control panel buttons (y=-1.85, z=1.0)
    "button_red":    np.array([-0.45, -1.85, 0.9]),
    "button_green":  np.array([-0.15, -1.85, 0.9]),
    "button_yellow": np.array([0.15, -1.85, 0.9]),
    "button_blue":   np.array([0.45, -1.85, 0.9]),
}


def compute_approach_waypoint(object_pos, approach_distance=0.4):
    """
    Compute a position for the robot to stand at within grasping range.
    For tables (higher x), approach from the front (lower x).
    For control panel buttons (low y), approach from in front (higher y).
    """
    obj_xy = object_pos[:2]
    
    # Control panel buttons (y is very negative)
    if object_pos[1] < -1.0:
        # Approach from in front (robot stands at higher y)
        return np.array([obj_xy[0], obj_xy[1] + approach_distance])
    
    # Table objects (higher x values)
    else:
        # Approach from front of table (lower x)
        return np.array([obj_xy[0] - approach_distance, obj_xy[1]])


class WalkAndGraspDemo:
    """
    Demo combining locomotion and manipulation:
    - Phase 1: Walk to target object using PID + AMO
    - Phase 2: Use trained PPO policy to control arm for grasping
    """
    
    def __init__(
        self,
        target_object: str = "screwdriver",
        model_path: str = "checkpoints/final_model",
        approach_distance: float = 0.85,
        arrival_threshold: float = None,  # Auto-set based on target type
        grasp_duration: float = 5.0,  # seconds to run grasp policy
        freeze_arm: str = "left",
        use_universal: bool = False,
        device: str = None,
    ):
        self.target_object = target_object
        self.model_path = model_path
        self.approach_distance = approach_distance
        self.grasp_duration = grasp_duration
        self.freeze_arm = freeze_arm
        self.use_universal = use_universal
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get target position
        if target_object not in OBJECT_POSITIONS:
            raise ValueError(f"Unknown object: {target_object}. Choose from: {list(OBJECT_POSITIONS.keys())}")
        
        self.target_pos_3d = OBJECT_POSITIONS[target_object]
        
        # Set arrival threshold based on target type
        # Buttons need tighter threshold (0.3m), table objects need larger (0.75m)
        if arrival_threshold is not None:
            self.arrival_threshold = arrival_threshold
        elif target_object.startswith("button_"):
            self.arrival_threshold = 0.3
        else:
            self.arrival_threshold = 0.75
        
        self.approach_waypoint = compute_approach_waypoint(self.target_pos_3d, approach_distance)
        
        print(f"\n=== Walk and Grasp Demo ===")
        print(f"Target object: {target_object}")
        print(f"Object position: ({self.target_pos_3d[0]:.2f}, {self.target_pos_3d[1]:.2f}, {self.target_pos_3d[2]:.2f})")
        print(f"Approach waypoint: ({self.approach_waypoint[0]:.2f}, {self.approach_waypoint[1]:.2f})")
        print(f"Arrival threshold: {self.arrival_threshold}m ({'button' if target_object.startswith('button_') else 'table object'})")
        print(f"Grasp model: {model_path}")
        print(f"Model type: {'Universal (35-dim)' if use_universal else 'Object-specific (33-dim)'}")
        print()
        
    def run(self):
        """Execute the walk and grasp sequence."""
        
        # ========================================
        # PHASE 1: Walk to object
        # ========================================
        print(">>> PHASE 1: Walking to object...")
        
        # Load AMO locomotion policy
        policy_jit = torch.jit.load("amo_jit.pt", map_location=self.device)
        env = HumanoidEnv(policy_jit=policy_jit, robot_type="g1", device=self.device)
        
        # PID controller for navigation
        pid = LocomotionPIDController(
            kp_pos=1.0, ki_pos=0.0, kd_pos=0.1,
            max_vel=0.6,
            min_vel=0.2
        )
        
        pd_target = np.zeros(env.num_dofs)
        pd_target[:15] = env.default_dof_pos[:15]
        pd_target[15:] = env.default_dof_pos[15:]
        
        arrived = False
        print_counter = 0
        min_dist_seen = float('inf')
        
        print(f"    Arrival threshold: {self.arrival_threshold}m")
        print(f"    Approach waypoint: ({self.approach_waypoint[0]:.3f}, {self.approach_waypoint[1]:.3f})")
        
        for i in range(int(env.sim_duration / env.sim_dt)):
            env._extract_state()
            
            if i % env.sim_decimation == 0:
                robot_pos = env.data.xpos[env.pelvis_id][:2]
                robot_quat = env.quat
                robot_rpy = quat_to_euler(robot_quat)
                robot_yaw = robot_rpy[2]
                
                dist = np.linalg.norm(self.approach_waypoint - robot_pos)
                min_dist_seen = min(min_dist_seen, dist)
                
                # Status update every 50 steps (~1 second)
                print_counter += 1
                if print_counter % 50 == 0:
                    print(f"    Dist: {dist:.3f}m (min: {min_dist_seen:.3f}m) | Robot: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | Target: ({self.approach_waypoint[0]:.2f}, {self.approach_waypoint[1]:.2f})")
                
                # Check arrival
                if dist < self.arrival_threshold:
                    print(f">>> Arrived at approach position! Distance: {dist:.3f}m")
                    arrived = True
                    break
                
                # Compute navigation commands
                bearing = np.arctan2(
                    self.approach_waypoint[1] - robot_pos[1],
                    self.approach_waypoint[0] - robot_pos[0]
                )
                
                vx, vy, heading_cmd = pid.compute_action(
                    current_pos=robot_pos,
                    current_yaw=robot_yaw,
                    target_pos=self.approach_waypoint,
                    target_yaw=bearing,
                    dt=env.control_dt
                )
                
                env.viewer.commands[0] = vx
                env.viewer.commands[2] = vy
                env.viewer.commands[1] = heading_cmd
                
                # Run AMO policy for leg control
                obs = env._compute_observation()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(env.device)
                
                with torch.no_grad():
                    extra_hist = torch.tensor(
                        np.array(env.extra_history).flatten().copy(),
                        dtype=torch.float
                    ).view(1, -1).to(env.device)
                    raw_action = env.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                raw_action = np.clip(raw_action, -40., 40.)
                env.last_action = np.concatenate([
                    raw_action.copy(),
                    (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale
                ])
                scaled_actions = raw_action * env.action_scale
                
                pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + env.default_dof_pos
                pd_target[15:] = (1 - env.arm_blend) * env.prev_arm_action + env.arm_blend * env.arm_action
                env.arm_blend = min(1.0, env.arm_blend + 0.01)
                
                # Update gait
                env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
                if env._in_place_stand and np.any(np.abs(env.gait_cycle - 0.25) < 0.05):
                    env.gait_cycle = np.array([0.25, 0.25])
                if not env._in_place_stand and np.all(np.abs(env.gait_cycle - 0.25) < 0.05):
                    env.gait_cycle = np.array([0.25, 0.75])
                
                # Render
                if hasattr(env, 'viewer') and env.viewer is not None:
                    env.viewer.cam.lookat = env.data.xpos[env.pelvis_id].astype(np.float32)
                    env.viewer.render()
            
            # Apply torques
            torque = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
            torque = np.clip(torque, -env.torque_limits, env.torque_limits)
            env.data.ctrl = torque
            mujoco.mj_step(env.model, env.data)
        
        if not arrived:
            print(">>> WARNING: Did not reach approach position in time!")
            env.viewer.close()
            return
        
        # ========================================
        # PHASE 2: Grasp object
        # ========================================
        print(f"\n>>> PHASE 2: Attempting to grasp {self.target_object}...")
        print(f"    Running trained arm policy for {self.grasp_duration}s")
        
        # Load the trained PPO model
        try:
            grasp_model = PPO.load(self.model_path, device=self.device)
            print(f"    Loaded grasp model: {self.model_path}")
        except Exception as e:
            print(f"    ERROR: Could not load model: {e}")
            env.viewer.close()
            return
        
        # Keep robot standing still during grasping
        env.viewer.commands[0] = 0.0  # vx = 0
        env.viewer.commands[1] = 0.0  # yaw = 0
        env.viewer.commands[2] = 0.0  # vy = 0
        
        # Cache hand body IDs and target body ID
        right_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand')
        
        # Map target object name to MuJoCo body name
        body_name_map = {
            "screwdriver": "screwdriver",
            "battery_pack": "battery_pack",
            "wrench": "wrench",
            "block_cube": "block_cube",
            "small_box": "small_box",
            "purple_object": "right_table_item2",
            "button_red": "push_button_red",
            "button_green": "push_button_green",
            "button_yellow": "push_button_yellow",
            "button_blue": "push_button_blue",
        }
        body_name = body_name_map.get(self.target_object, self.target_object)
        target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(f"    Right hand body ID: {right_hand_id}, Target ({body_name}) body ID: {target_body_id}")
        
        # Detect model type based on observation space
        model_obs_dim = grasp_model.observation_space.shape[0]
        is_button_model = (model_obs_dim == 32)
        is_universal_model = (model_obs_dim == 35)
        is_grasp_model = (model_obs_dim == 33)
        print(f"    Model observation dim: {model_obs_dim} -> {'button' if is_button_model else 'universal' if is_universal_model else 'grasp'} model")
        
        # Initialize button tracking if needed
        if is_button_model and self.target_object.startswith("button_"):
            joint_name = f"push_{self.target_object}_joint"
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self._initial_button_disp = env.data.qpos[joint_id]
        
        # Grasping phase - run for grasp_duration seconds
        grasp_steps = int(self.grasp_duration / env.sim_dt)
        print(f"    Running {grasp_steps} simulation steps...")
        
        grasp_print_counter = 0
        min_hand_dist = float('inf')
        
        for step in range(grasp_steps):
            env._extract_state()
            
            if step % env.sim_decimation == 0:
                grasp_print_counter += 1
                
                # Get observation for PPO policy based on model type
                if is_button_model:
                    obs = self._get_button_observation(env)
                elif self.use_universal or is_universal_model:
                    obs = self._get_universal_observation(env)
                else:
                    obs = self._get_grasp_observation(env)
                
                # Get arm action from trained policy
                with torch.no_grad():
                    arm_action, _ = grasp_model.predict(obs, deterministic=True)
                
                # arm_action is 8-dim for both arms, but we may freeze one
                arm_action = np.array(arm_action, dtype=np.float32)
                
                # Apply freeze_arm logic
                if self.freeze_arm == "left":
                    # Freeze left arm (indices 0-3), use PPO output for right arm (4-7)
                    arm_action[:4] = 0.0
                elif self.freeze_arm == "right":
                    # Freeze right arm (indices 4-7), use PPO output for left arm (0-3)
                    arm_action[4:] = 0.0
                
                # Scale arm action
                scaled_arm_action = arm_action * 0.25  # action_scale
                
                # Run AMO policy for legs (standing)
                amo_obs = env._compute_observation()
                obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(env.device)
                
                with torch.no_grad():
                    extra_hist = torch.tensor(
                        np.array(env.extra_history).flatten().copy(),
                        dtype=torch.float
                    ).view(1, -1).to(env.device)
                    leg_action = env.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                leg_action = np.clip(leg_action, -40., 40.)
                env.last_action = np.concatenate([
                    leg_action.copy(),
                    (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale
                ])
                scaled_leg_action = leg_action * env.action_scale
                
                # Combine leg (15 DOFs) + arm (8 DOFs)
                pd_target = env.default_dof_pos.copy()
                pd_target[:15] += scaled_leg_action
                pd_target[15:] += scaled_arm_action
                
                # Update gait for standing
                env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
                if np.any(np.abs(env.gait_cycle - 0.25) < 0.05):
                    env.gait_cycle = np.array([0.25, 0.25])
                
                # Render
                if hasattr(env, 'viewer') and env.viewer is not None:
                    env.viewer.cam.lookat = env.data.xpos[env.pelvis_id].astype(np.float32)
                    env.viewer.render()
                
                # Print distance to object every 25 control steps (~0.5 seconds)
                if grasp_print_counter % 25 == 0:
                    if right_hand_id >= 0 and target_body_id >= 0:
                        hand_pos = env.data.xpos[right_hand_id]
                        obj_pos = env.data.xpos[target_body_id]
                        obj_dist = np.linalg.norm(hand_pos - obj_pos)
                        min_hand_dist = min(min_hand_dist, obj_dist)
                        elapsed = step * env.sim_dt
                        print(f"    [{elapsed:.1f}s] Hand-to-object: {obj_dist:.3f}m (min: {min_hand_dist:.3f}m) | Action: [{arm_action[4]:.2f}, {arm_action[5]:.2f}, {arm_action[6]:.2f}, {arm_action[7]:.2f}]")
            
            # Apply torques
            torque = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
            torque = np.clip(torque, -env.torque_limits, env.torque_limits)
            env.data.ctrl = torque
            mujoco.mj_step(env.model, env.data)
        
        print(f"\n>>> Grasp attempt completed!")
        print(">>> Press ESC to close viewer...")
        
        # Keep rendering until user closes
        while env.viewer.is_alive:
            env._extract_state()
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.render()
        
        env.viewer.close()
    
    def _get_grasp_observation(self, env):
        """
        Construct observation matching G1RLEnv._get_grasp_obs() format.
        
        The PPO policy was trained on observations:
        - arm joint positions (8)
        - arm joint velocities (8) scaled by 0.1
        - left hand position (3)
        - right hand position (3)
        - target object position (3)
        - left hand to target vector (3)
        - right hand to target vector (3)
        - grasp state (2): [is_grasped, lift_amount]
        Total: 33
        """
        # Arm joint positions and velocities (8 DOFs each)
        arm_pos = env.dof_pos[15:23]  # arm joints 15-22
        arm_vel = env.dof_vel[15:23]
        
        # Hand positions
        left_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand')
        right_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand')
        
        if left_hand_id >= 0:
            left_hand_pos = env.data.xpos[left_hand_id]
        else:
            left_hand_pos = np.zeros(3)
        
        if right_hand_id >= 0:
            right_hand_pos = env.data.xpos[right_hand_id]
        else:
            right_hand_pos = np.zeros(3)
        
        # Target position - try to get actual position from simulation
        # Map target object name to MuJoCo body name
        body_name_map = {
            "screwdriver": "screwdriver",
            "battery_pack": "battery_pack",
            "wrench": "wrench",
            "block_cube": "block_cube",
            "small_box": "small_box",
            "purple_object": "right_table_item2",
            "button_red": "push_button_red",
            "button_green": "push_button_green",
            "button_yellow": "push_button_yellow",
            "button_blue": "push_button_blue",
        }
        
        body_name = body_name_map.get(self.target_object, self.target_object)
        target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id >= 0:
            target_pos = env.data.xpos[target_body_id].copy()
        else:
            target_pos = self.target_pos_3d.copy()
        
        # Vectors to target
        left_to_target = target_pos - left_hand_pos
        right_to_target = target_pos - right_hand_pos
        
        # Grasp state: [is_grasped=0, lift_amount=0] (we start fresh)
        initial_height = 0.74
        grasp_state = np.array([0.0, target_pos[2] - initial_height])
        
        obs = np.concatenate([
            arm_pos,                    # 8
            arm_vel * 0.1,              # 8 (scaled down)
            left_hand_pos,              # 3
            right_hand_pos,             # 3
            target_pos,                 # 3
            left_to_target,             # 3
            right_to_target,            # 3
            grasp_state,                # 2
        ]).astype(np.float32)
        
        return obs
    
    def _get_universal_observation(self, env):
        """
        Construct observation for UNIVERSAL model (35 dims).
        
        Format matches UniversalManipulationEnv:
        - arm joint positions (8)
        - arm joint velocities (8)
        - left hand position (3)
        - right hand position (3)
        - target position (3)
        - left hand to target (3)
        - right hand to target (3)
        - task type one-hot (2): [is_grasp, is_press]
        - task state (2): [button_disp or is_grasped, lift_amount]
        """
        arm_pos = env.dof_pos[15:23]
        arm_vel = env.dof_vel[15:23]
        
        left_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand')
        right_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand')
        
        left_hand_pos = env.data.xpos[left_hand_id] if left_hand_id >= 0 else np.zeros(3)
        right_hand_pos = env.data.xpos[right_hand_id] if right_hand_id >= 0 else np.zeros(3)
        
        # Get target position
        body_name_map = {
            "screwdriver": "screwdriver",
            "battery_pack": "battery_pack",
            "wrench": "wrench",
            "block_cube": "block_cube",
            "small_box": "small_box",
            "purple_object": "right_table_item2",
            "button_red": "push_button_red",
            "button_green": "push_button_green",
            "button_yellow": "push_button_yellow",
            "button_blue": "push_button_blue",
        }
        body_name = body_name_map.get(self.target_object, self.target_object)
        target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        target_pos = env.data.xpos[target_body_id].copy() if target_body_id >= 0 else self.target_pos_3d.copy()
        
        left_to_target = target_pos - left_hand_pos
        right_to_target = target_pos - right_hand_pos
        
        # Task type
        is_button = self.target_object.startswith("button_")
        task_type = np.array([0.0 if is_button else 1.0, 1.0 if is_button else 0.0])
        
        # Task state
        if is_button:
            joint_name = f"push_{self.target_object}_joint"
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                button_disp = env.data.qpos[joint_id]
            else:
                button_disp = 0.0
            task_state = np.array([button_disp, 0.0])
        else:
            task_state = np.array([0.0, target_pos[2] - 0.74])
        
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
    
    def _get_button_observation(self, env):
        """
        Construct observation for BUTTON model (32 dims).
        
        Format matches ButtonPressEnv:
        - arm joint positions (8)
        - arm joint velocities (8)
        - left hand position (3)
        - right hand position (3)
        - button position (3)
        - left hand to button (3)
        - right hand to button (3)
        - button displacement (1)
        """
        arm_pos = env.dof_pos[15:23]
        arm_vel = env.dof_vel[15:23]
        
        left_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand')
        right_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand')
        
        left_hand_pos = env.data.xpos[left_hand_id] if left_hand_id >= 0 else np.zeros(3)
        right_hand_pos = env.data.xpos[right_hand_id] if right_hand_id >= 0 else np.zeros(3)
        
        # Get button body name
        button_body_map = {
            "button_red": "push_button_red",
            "button_green": "push_button_green",
            "button_yellow": "push_button_yellow",
            "button_blue": "push_button_blue",
        }
        body_name = button_body_map.get(self.target_object, "push_button_red")
        button_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        button_pos = env.data.xpos[button_body_id].copy() if button_body_id >= 0 else self.target_pos_3d.copy()
        
        left_to_button = button_pos - left_hand_pos
        right_to_button = button_pos - right_hand_pos
        
        # Get button displacement (relative - start is 0)
        joint_name = f"push_{self.target_object}_joint"
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            # Track initial displacement to compute relative
            if not hasattr(self, '_initial_button_disp'):
                self._initial_button_disp = env.data.qpos[joint_id]
            button_disp = env.data.qpos[joint_id] - self._initial_button_disp
        else:
            button_disp = 0.0
        
        obs = np.concatenate([
            arm_pos,                    # 8
            arm_vel * 0.1,              # 8 (scaled)
            left_hand_pos,              # 3
            right_hand_pos,             # 3
            button_pos,                 # 3
            left_to_button,             # 3
            right_to_button,            # 3
            [button_disp],              # 1
        ]).astype(np.float32)
        
        return obs


def main():
    parser = argparse.ArgumentParser(description="Walk and Grasp Demo")
    parser.add_argument("--target", type=str, default="screwdriver",
                        choices=list(OBJECT_POSITIONS.keys()),
                        help="Target object to grasp")
    parser.add_argument("--model", type=str, default="checkpoints/final_model",
                        help="Path to trained grasp model (without .zip)")
    parser.add_argument("--approach_distance", type=float, default=0.4,
                        help="Distance from object to stop for grasping")
    parser.add_argument("--grasp_duration", type=float, default=5.0,
                        help="Seconds to run grasp policy")
    parser.add_argument("--freeze_arm", type=str, default="left",
                        choices=["none", "left", "right"],
                        help="Which arm to freeze during grasping")
    parser.add_argument("--universal", action="store_true",
                        help="Use universal model (35-dim obs) instead of object-specific (33-dim)")
    
    args = parser.parse_args()
    
    demo = WalkAndGraspDemo(
        target_object=args.target,
        model_path=args.model,
        approach_distance=args.approach_distance,
        grasp_duration=args.grasp_duration,
        freeze_arm=args.freeze_arm,
        use_universal=args.universal,
    )
    
    demo.run()


if __name__ == "__main__":
    main()
