import torch
import numpy as np
import mujoco
import time
from play_amo import HumanoidEnv, quat_to_euler
from pid_controller import LocomotionPIDController

# ============================================
# INTERACTIVE OBJECT POSITIONS
# Robot needs to approach within grasping range (~0.3-0.4m from object)
# ============================================

# Object positions (x, y, z) from the simulation
OBJECT_POSITIONS = {
    # Buttons on control panel (y=-1.85, z=1.0)
    "button_red":    np.array([-0.45, -1.85, 1.0]),
    "button_green":  np.array([-0.15, -1.85, 1.0]),
    "button_yellow": np.array([0.15, -1.85, 1.0]),
    "button_blue":   np.array([0.45, -1.85, 1.0]),
    
    # Objects on left table (y ≈ -0.15 to -0.3)
    "block_cube":  np.array([1.5, -0.15, 0.75]),
    "small_box":   np.array([1.3, -0.3, 0.74]),
    "wrench":      np.array([1.25, -0.25, 0.74]),
    
    # Objects on right table (y ≈ 1.35 to 1.5)
    "screwdriver":   np.array([1.1, 1.35, 0.74]),
    "purple_object": np.array([1.4, 1.45, 0.74]),
    "battery_pack":  np.array([1.55, 1.5, 0.74]),
}

# Robot start position
ROBOT_START = np.array([0.7, 1.6])

def compute_approach_waypoint(object_pos, approach_distance=0.4):
    """
    Compute a position for the robot to stand at to be within grasping range.
    For tables (higher x), approach from the front (lower x).
    For control panel (low y), approach from in front (higher y).
    """
    obj_xy = object_pos[:2]
    
    # Control panel buttons (y is very negative)
    if object_pos[1] < -1.0:
        # Approach from in front (robot stands at higher y)
        return np.array([obj_xy[0], obj_xy[1] + approach_distance])
    
    # Table objects (higher x values)
    else:
        # Approach from the front of the table (lower x)
        return np.array([obj_xy[0] - approach_distance, obj_xy[1]])

def get_navigation_tour():
    """
    Returns a list of waypoints to visit all interactive objects.
    Each waypoint is the approach position for grasping.
    """
    # Define the tour order for the requested subset:
    # (right-table screwdriver, battery), wrench (left table), then control panel buttons
    tour_order = [
        "screwdriver",
        "battery_pack",
        "wrench",
        "button_blue",
        "button_yellow",
        "button_green",
        "button_red",
    ]
    
    waypoints = []
    for obj_name in tour_order:
        obj_pos = OBJECT_POSITIONS[obj_name]
        approach_pos = compute_approach_waypoint(obj_pos)
        waypoints.append((obj_name, approach_pos))
    
    return waypoints

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy_path = "amo_jit.pt"
    
    policy_jit = torch.jit.load(policy_path, map_location=device)
    env = HumanoidEnv(policy_jit=policy_jit, robot_type="g1", device=device)
    
    pid = LocomotionPIDController(
        kp_pos=1.0, ki_pos=0.0, kd_pos=0.1,
        max_vel=0.6,
        min_vel=0.2
    )
    
    # Get the navigation tour through all objects
    tour = get_navigation_tour()
    print("\n=== Object Navigation Tour ===")
    for i, (name, pos) in enumerate(tour):
        obj_pos = OBJECT_POSITIONS[name]
        print(f"  {i+1}. {name}: object at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}), approach at ({pos[0]:.2f}, {pos[1]:.2f})")
    print()
    
    # Extract just the waypoint positions
    waypoints = [pos for _, pos in tour]
    waypoint_names = [name for name, _ in tour]
    
    current_waypoint_idx = 0
    threshold = 0.5# Arrival threshold
    dwell_time = 2.0  # Seconds to pause at each waypoint
    dwell_counter = 0  # Counter for dwell time
    is_dwelling = False  # Flag for whether robot is pausing at waypoint
    
    pd_target = np.zeros(env.num_dofs)
    pd_target[:15] = env.default_dof_pos[:15]
    pd_target[15:] = env.default_dof_pos[15:]
    
    print(f"\n>>> Starting navigation. First target: {waypoint_names[0]}")
    print_counter = 0  # For periodic status updates
    
    for i in range(int(env.sim_duration / env.sim_dt)):
        env._extract_state()
        
        if i % env.sim_decimation == 0:
            # Get robot position from pelvis body (not qpos which has object DOFs first)
            robot_pos = env.data.xpos[env.pelvis_id][:2]
            robot_quat = env.quat
            robot_rpy = quat_to_euler(robot_quat)
            robot_yaw = robot_rpy[2]
            
            target_pos = waypoints[current_waypoint_idx]
            dist = np.linalg.norm(target_pos - robot_pos)
            
            # Print status every 100 control steps (~2 seconds)
            print_counter += 1
            if print_counter % 100 == 0:
                print(f"    Distance to {waypoint_names[current_waypoint_idx]}: {dist:.3f}m | Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
            
            # Check if dwelling at waypoint
            if is_dwelling:
                dwell_counter += 1
                if dwell_counter >= int(dwell_time / env.control_dt):
                    is_dwelling = False
                    dwell_counter = 0
                    current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                    target_pos = waypoints[current_waypoint_idx]
                    print(f">>> Moving to next target: {waypoint_names[current_waypoint_idx]}")
                # During dwell, send zero velocity commands
                vx, vy, heading_cmd = 0.0, 0.0, 0.0
            elif dist < threshold:
                print(f">>> Arrived at {waypoint_names[current_waypoint_idx]}! Distance: {dist:.3f}m")
                print(f"    Pausing for {dwell_time}s to interact...")
                is_dwelling = True
                dwell_counter = 0
                vx, vy, heading_cmd = 0.0, 0.0, 0.0
            else:
                bearing_to_target = np.arctan2(target_pos[1] - robot_pos[1], target_pos[0] - robot_pos[0])
                
                vx, vy, heading_cmd = pid.compute_action(
                    current_pos=robot_pos,
                    current_yaw=robot_yaw,
                    target_pos=target_pos,
                    target_yaw=bearing_to_target,
                    dt=env.control_dt
                )
            
            env.viewer.commands[0] = vx
            env.viewer.commands[2] = vy
            env.viewer.commands[1] = heading_cmd
            

            obs = env._compute_observation()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(env.device)
            
            with torch.no_grad():
                extra_hist = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(env.device)
                raw_action = env.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
            
            raw_action = np.clip(raw_action, -40., 40.)
            env.last_action = np.concatenate([raw_action.copy(), (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale])
            scaled_actions = raw_action * env.action_scale
            
            pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + env.default_dof_pos
            pd_target[15:] = (1 - env.arm_blend) * env.prev_arm_action + env.arm_blend * env.arm_action
            env.arm_blend = min(1.0, env.arm_blend + 0.01)
            
            env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
            if env._in_place_stand and ((np.abs(env.gait_cycle[0] - 0.25) < 0.05) or (np.abs(env.gait_cycle[1] - 0.25) < 0.05)):
                env.gait_cycle = np.array([0.25, 0.25])
            if (not env._in_place_stand) and ((np.abs(env.gait_cycle[0] - 0.25) < 0.05) and (np.abs(env.gait_cycle[1] - 0.25) < 0.05)):
                env.gait_cycle = np.array([0.25, 0.75])
            
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.cam.lookat = env.data.qpos.astype(np.float32)[:3]
                env.viewer.render()
                
        torque = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
        torque = np.clip(torque, -env.torque_limits, env.torque_limits)
        env.data.ctrl = torque
        mujoco.mj_step(env.model, env.data)
        
    env.viewer.close()

if __name__ == "__main__":
    main()