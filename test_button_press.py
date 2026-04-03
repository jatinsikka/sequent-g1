"""
Test script to verify button pressing setup.

This tests:
1. Robot position relative to button
2. Whether arm can reach the button
3. Button joint displacement detection
"""

import numpy as np
import mujoco
import time
import torch
from play_amo import HumanoidEnv
from reward_fn import BUTTON_POSITIONS


def test_button_reach():
    """Test if robot can reach the button with arm extension."""
    
    print("\n=== Button Press Test ===\n")
    
    # Load environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_jit = torch.jit.load("amo_jit.pt", map_location=device)
    
    env = HumanoidEnv(
        policy_jit=policy_jit,
        robot_type="g1",
        device=device,
        headless=False,  # Show visualization
    )
    
    # Reset to keyframe (robot should be at -0.45, -1.50 facing -Y)
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_step(env.model, env.data)
    
    # Get body IDs
    left_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand')
    right_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand')
    button_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'push_button_red')
    button_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'push_button_red_joint')
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    
    print(f"Body IDs - Left hand: {left_hand_id}, Right hand: {right_hand_id}")
    print(f"Button body ID: {button_body_id}, joint ID: {button_joint_id}")
    
    # Check initial positions
    left_hand_pos = env.data.xpos[left_hand_id].copy()
    right_hand_pos = env.data.xpos[right_hand_id].copy()
    button_pos = env.data.xpos[button_body_id].copy()
    pelvis_pos = env.data.xpos[pelvis_id].copy()
    
    print(f"\n=== Initial Positions ===")
    print(f"Robot pelvis: ({pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f})")
    print(f"Left hand:    ({left_hand_pos[0]:.3f}, {left_hand_pos[1]:.3f}, {left_hand_pos[2]:.3f})")
    print(f"Right hand:   ({right_hand_pos[0]:.3f}, {right_hand_pos[1]:.3f}, {right_hand_pos[2]:.3f})")
    print(f"Red button:   ({button_pos[0]:.3f}, {button_pos[1]:.3f}, {button_pos[2]:.3f})")
    
    # Calculate distances
    dist_left = np.linalg.norm(left_hand_pos - button_pos)
    dist_right = np.linalg.norm(right_hand_pos - button_pos)
    print(f"\n=== Distances to Button ===")
    print(f"Left hand to button:  {dist_left:.3f}m")
    print(f"Right hand to button: {dist_right:.3f}m")
    
    # Check which hand is closer and calculate components
    closer_hand = "left" if dist_left < dist_right else "right"
    closer_pos = left_hand_pos if closer_hand == "left" else right_hand_pos
    delta = button_pos - closer_pos
    print(f"\nCloser hand: {closer_hand}")
    print(f"Delta X: {delta[0]:.3f}m, Delta Y: {delta[1]:.3f}m, Delta Z: {delta[2]:.3f}m")
    
    # Now test arm extension toward button
    print("\n=== Testing Arm Extension ===")
    print("Moving left arm toward button...")
    
    # Left arm joints are at dof indices 15-18 (shoulder pitch, roll, yaw, elbow)
    # For reaching forward-down toward button:
    # - shoulder_pitch (15): negative = forward
    # - elbow (18): positive = extend
    
    # Get initial arm position
    initial_arm = env.data.qpos[15:19].copy()
    print(f"Initial arm joints: {initial_arm}")
    
    # Target arm position for reaching toward button
    # We'll slowly move the arm
    steps = 100
    for i in range(steps):
        t = i / steps
        
        # Gradually move arm toward button
        # Shoulder pitch: move forward (negative)
        env.data.qpos[15] = initial_arm[0] - t * 0.8  # Forward rotation
        # Shoulder roll: slight outward
        env.data.qpos[16] = initial_arm[1] + t * 0.2
        # Elbow: extend slightly
        env.data.qpos[18] = initial_arm[3] + t * 0.3
        
        mujoco.mj_step(env.model, env.data)
        
        if i % 20 == 0:
            left_hand_pos = env.data.xpos[left_hand_id].copy()
            button_pos = env.data.xpos[button_body_id].copy()
            dist = np.linalg.norm(left_hand_pos - button_pos)
            print(f"Step {i}: Hand at ({left_hand_pos[0]:.3f}, {left_hand_pos[1]:.3f}, {left_hand_pos[2]:.3f}), dist={dist:.3f}m")
        
        if not env.headless:
            env.viewer.sync()
            time.sleep(0.02)
    
    # Check final distance
    left_hand_pos = env.data.xpos[left_hand_id].copy()
    button_pos = env.data.xpos[button_body_id].copy()
    final_dist = np.linalg.norm(left_hand_pos - button_pos)
    print(f"\n=== Final State ===")
    print(f"Left hand: ({left_hand_pos[0]:.3f}, {left_hand_pos[1]:.3f}, {left_hand_pos[2]:.3f})")
    print(f"Button:    ({button_pos[0]:.3f}, {button_pos[1]:.3f}, {button_pos[2]:.3f})")
    print(f"Distance:  {final_dist:.3f}m")
    
    if final_dist < 0.1:
        print("✓ Hand can reach close to button!")
    else:
        print("✗ Hand is still far from button - may need position adjustment")
    
    # Check button displacement
    button_disp = env.data.qpos[button_joint_id]
    print(f"\nButton displacement: {button_disp:.4f}m")
    
    # Hold for visualization
    print("\nHolding for 5 seconds...")
    for _ in range(250):
        mujoco.mj_step(env.model, env.data)
        if not env.headless:
            env.viewer.sync()
            time.sleep(0.02)
    
    env.close()
    print("\n>>> Test complete!")


if __name__ == "__main__":
    test_button_reach()
