"""
Test script to verify magnetic grasping functionality.
Manually moves the robot's hand close to the screwdriver to test grasp triggering.
"""

import numpy as np
import torch
import mujoco
import time

from env_wrapper import G1RLEnv
from config import get_training_config


def test_magnetic_grasp():
    """Test the magnetic grasp by manually positioning the hand near the screwdriver."""
    
    print("=" * 60)
    print("MAGNETIC GRASP TEST")
    print("=" * 60)
    
    # Create environment
    config = get_training_config()
    env = G1RLEnv(
        policy_jit_path="amo_jit.pt",
        robot_type="g1",
        device="cuda",
        max_episode_steps=1000,
        headless=False,  # Show visualization
        max_episode_time=30.0,
        freeze_arm="left",  # Only use right arm
    )
    
    obs, info = env.reset()
    
    # Print arm configuration
    print(f"\nArm joint configuration:")
    print(f"  Default arm pos: {env.default_arm_pos}")
    print(f"  Arm pos LOW:  {env.arm_pos_low}")
    print(f"  Arm pos HIGH: {env.arm_pos_high}")
    print(f"  Current arm pos: {env.env.dof_pos[15:23]}")
    
    # Get positions
    screwdriver_pos = env.env.data.xpos[env.screwdriver_body_id].copy()
    right_hand_pos = env.env.data.xpos[env.right_hand_id].copy()
    left_hand_pos = env.env.data.xpos[env.left_hand_id].copy()
    
    print(f"\nInitial positions:")
    print(f"  Screwdriver: {screwdriver_pos}")
    print(f"  Right hand:  {right_hand_pos}")
    print(f"  Left hand:   {left_hand_pos}")
    print(f"\nGrasp distance threshold: {env.grasp_distance}m")
    print(f"Grasp speed threshold: 0.5 m/s")
    
    # Calculate distances
    right_dist = np.linalg.norm(right_hand_pos - screwdriver_pos)
    left_dist = np.linalg.norm(left_hand_pos - screwdriver_pos)
    print(f"\nInitial distances:")
    print(f"  Right hand to screwdriver: {right_dist:.3f}m")
    print(f"  Left hand to screwdriver:  {left_dist:.3f}m")
    
    print("\n" + "=" * 60)
    print("Testing action effects...")
    print("=" * 60)
    
    # The action space maps [-1, 1] to [arm_pos_low, arm_pos_high]
    # For right arm (indices 4-7 in full arm, but we're frozen left so it's 0-3 in action):
    # Wait - if left is frozen, we only control right arm, so action is still 8D
    # but left arm indices (0-3) are overwritten with defaults
    
    # Right arm joints (indices 4-7 in full 8D action):
    # Index 4: Right shoulder pitch - positive = forward
    # Index 5: Right shoulder roll 
    # Index 6: Right shoulder yaw
    # Index 7: Right elbow - positive = bent
    
    step = 0
    grasped = False
    prev_action = np.zeros(8, dtype=np.float32)  # For smoothing
    
    try:
        while step < 500 and not grasped:
            # Get current positions
            screwdriver_pos = env.env.data.xpos[env.screwdriver_body_id].copy()
            right_hand_pos = env.env.data.xpos[env.right_hand_id].copy()
            
            dist = np.linalg.norm(right_hand_pos - screwdriver_pos)
            
            # Create action - action is 8D, controls full arm
            # Left arm (0-3) will be overwritten with defaults due to freeze_arm="left"
            # Right arm (4-7) will be used
            
            # SMOOTH approach: interpolate action based on distance
            # Normalize distance to [0, 1] range (0 = at target, 1 = far away)
            dist_normalized = np.clip((dist - 0.05) / (0.4 - 0.05), 0.0, 1.0)
            
            # Define target actions for "far" and "close"
            # FAR: aggressive reach toward screwdriver
            far_action = np.array([0, 0, 0, 0, -0.7, -0.1, -0.4, 0.5], dtype=np.float32)
            # CLOSE: gentle approach for grasp
            close_action = np.array([0, 0, 0, 0, -0.55, 0.0, -0.2, 0.3], dtype=np.float32)
            
            # Interpolate smoothly between far and close actions
            action = close_action + dist_normalized * (far_action - close_action)
            
            # Apply smoothing with previous action to avoid jerks
            smoothing = 0.8  # How much to keep from previous action (higher = smoother)
            action = smoothing * prev_action + (1 - smoothing) * action
            prev_action = action.copy()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check grasp status
            grasped = env.object_grasped
            
            # Print status every 25 steps
            if step % 25 == 0:
                hand_vel = info.get("active_hand_vel", 0)
                current_arm_pos = env.env.dof_pos[15:23]
                
                # Calculate target arm position from action
                arm_range = (env.arm_pos_high - env.arm_pos_low) / 2.0
                arm_center = (env.arm_pos_high + env.arm_pos_low) / 2.0
                target_arm_pos = arm_center + action * arm_range
                
                print(f"\n[Step {step}]")
                print(f"  Distance: {dist:.4f}m (threshold: {env.grasp_distance}m)")
                print(f"  Hand pos: {right_hand_pos}")
                print(f"  Screwdriver: {screwdriver_pos}")
                print(f"  Hand velocity: {hand_vel:.3f} m/s")
                print(f"  Action (right arm): {action[4:8]}")
                print(f"  Target arm pos: {target_arm_pos[4:8]}")
                print(f"  Current arm pos: {current_arm_pos[4:8]}")
                print(f"  Grasped: {grasped}")
                print(f"  Reward: {reward:.3f}")
            
            # Render
            env.render()
            time.sleep(0.02)
            
            step += 1
            
            if terminated or truncated:
                print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
                break
        
        if grasped:
            print("\n" + "=" * 60)
            print("SUCCESS! Object grasped!")
            print(f"  Grasping hand: {env.grasping_hand}")
            print("=" * 60)
            
            # Now try to lift
            print("\nAttempting to lift...")
            for lift_step in range(200):
                # Lift action - move arm up (reduce shoulder pitch)
                action = np.zeros(8, dtype=np.float32)
                action[4] = -0.5  # shoulder pitch back/up
                action[7] = 0.3   # elbow less bent
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                object_height = info.get("object_height", 0)
                lift_amount = info.get("lift_amount", 0)
                
                if lift_step % 25 == 0:
                    print(f"[Lift step {lift_step}] Height: {object_height:.3f}m, Lifted: {lift_amount:.3f}m")
                
                env.render()
                time.sleep(0.02)
                
                if terminated or truncated:
                    break
        else:
            print("\n" + "=" * 60)
            print("FAILED to grasp after 500 steps")
            print(f"  Final distance: {dist:.4f}m")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        env.close()
        print("\nTest complete!")


if __name__ == "__main__":
    test_magnetic_grasp()
