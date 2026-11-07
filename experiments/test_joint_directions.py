"""
Diagnostic script to understand arm joint directions.
"""

import numpy as np
import torch
import mujoco
import time

from env_wrapper import G1RLEnv


def test_joint_directions():
    """Test each arm joint individually to understand directions."""
    
    print("=" * 60)
    print("ARM JOINT DIRECTION TEST")
    print("=" * 60)
    
    env = G1RLEnv(
        policy_jit_path="amo_jit.pt",
        robot_type="g1",
        device="cuda",
        max_episode_steps=1000,
        headless=False,
        max_episode_time=60.0,
        freeze_arm=None,  # Control both arms
    )
    
    obs, info = env.reset()
    
    print(f"\nDefault arm positions: {env.default_arm_pos}")
    print(f"Arm pos LOW:  {env.arm_pos_low}")
    print(f"Arm pos HIGH: {env.arm_pos_high}")
    
    # Get initial hand position
    initial_right_hand = env.env.data.xpos[env.right_hand_id].copy()
    screwdriver_pos = env.env.data.xpos[env.screwdriver_body_id].copy()
    
    print(f"\nInitial right hand position: {initial_right_hand}")
    print(f"Screwdriver position: {screwdriver_pos}")
    print(f"Direction to screwdriver: {screwdriver_pos - initial_right_hand}")
    
    # Joint names for reference (right arm is indices 4-7)
    joint_names = [
        "Left shoulder pitch", "Left shoulder roll", "Left shoulder yaw", "Left elbow",
        "Right shoulder pitch", "Right shoulder roll", "Right shoulder yaw", "Right elbow"
    ]
    
    print("\n" + "=" * 60)
    print("Testing each RIGHT ARM joint (indices 4-7)")
    print("Moving from action=0 to action=+1 (maximum position)")
    print("=" * 60)
    
    for joint_idx in range(4, 8):  # Right arm joints
        print(f"\n--- Testing {joint_names[joint_idx]} (index {joint_idx}) ---")
        
        # Reset
        obs, _ = env.reset()
        initial_hand = env.env.data.xpos[env.right_hand_id].copy()
        
        # Apply action to just this joint
        for step in range(100):
            action = np.zeros(8, dtype=np.float32)
            action[joint_idx] = 1.0  # Maximum position for this joint
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        final_hand = env.env.data.xpos[env.right_hand_id].copy()
        movement = final_hand - initial_hand
        
        print(f"  Initial hand pos: {initial_hand}")
        print(f"  Final hand pos:   {final_hand}")
        print(f"  Movement (delta): {movement}")
        print(f"  Movement magnitude: {np.linalg.norm(movement):.3f}m")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("Now testing NEGATIVE direction (action=-1)")
    print("=" * 60)
    
    for joint_idx in range(4, 8):  # Right arm joints
        print(f"\n--- Testing {joint_names[joint_idx]} (index {joint_idx}) NEGATIVE ---")
        
        # Reset
        obs, _ = env.reset()
        initial_hand = env.env.data.xpos[env.right_hand_id].copy()
        
        # Apply negative action to just this joint
        for step in range(100):
            action = np.zeros(8, dtype=np.float32)
            action[joint_idx] = -1.0  # Minimum position for this joint
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        final_hand = env.env.data.xpos[env.right_hand_id].copy()
        movement = final_hand - initial_hand
        
        print(f"  Initial hand pos: {initial_hand}")
        print(f"  Final hand pos:   {final_hand}")
        print(f"  Movement (delta): {movement}")
        print(f"  Movement magnitude: {np.linalg.norm(movement):.3f}m")
        
        time.sleep(0.5)
    
    env.close()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_joint_directions()
