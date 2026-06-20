"""
walk_to skill: PID + AMO locomotion to a named target, headless and verifiable.

This is the validated nav loop from walk_and_grasp.py (proven 2026-06-18 to reach
the table target upright). The executor calls run_walk_to() and treats the
measured outcome (arrived within threshold AND did not fall) as the physics-checked
postcondition for a walk_to step.

Author: Jatin Sikka
"""

from __future__ import annotations

import numpy as np
import torch
import mujoco

from play_amo import HumanoidEnv, quat_to_euler
from pid_controller import LocomotionPIDController
from walk_and_grasp import OBJECT_POSITIONS, compute_approach_waypoint

# Map the planner's free-text target names to a 3D location the robot walks toward.
# Tables/objects resolve to the object; machine/panel/shelf resolve to the button panel.
_ALIASES = {
    "table": "screwdriver", "tool_table": "screwdriver", "workbench": "screwdriver",
    "bench": "screwdriver", "tool": "screwdriver",
    "machine": "button_red", "machine_a": "button_red", "control_panel": "button_red",
    "panel": "button_red", "shelf": "button_red", "station": "button_red",
}


def _resolve_target(target: str) -> np.ndarray:
    """Resolve a target name to its 3D position (object or panel)."""
    key = target.lower().strip()
    if key in OBJECT_POSITIONS:
        return OBJECT_POSITIONS[key]
    if key in _ALIASES:
        return OBJECT_POSITIONS[_ALIASES[key]]
    # default: the tool table
    return OBJECT_POSITIONS["screwdriver"]


def run_walk_to(target: str, max_control_steps: int = 600) -> dict:
    """Walk the G1 to `target` via PID + AMO. Returns measured outcome:
    {arrived, fell, min_dist, steps, waypoint}. arrived+not-fell == verified."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy_jit = torch.jit.load("amo_jit.pt", map_location=dev)
    env = HumanoidEnv(policy_jit=policy_jit, robot_type="g1", device=dev, headless=True)
    pid = LocomotionPIDController(kp_pos=1.0, ki_pos=0.0, kd_pos=0.1, max_vel=0.6, min_vel=0.2)

    tgt3d = _resolve_target(target)
    is_panel = tgt3d[1] < -1.0
    waypoint = compute_approach_waypoint(tgt3d, approach_distance=0.4 if is_panel else 0.85)
    arrival = 0.3 if is_panel else 0.75

    pd_target = env.default_dof_pos.copy()
    arrived = fell = False
    min_dist = float("inf"); cstep = 0

    for i in range(int(env.sim_duration / env.sim_dt)):
        env._extract_state()
        if i % env.sim_decimation == 0:
            cstep += 1
            if cstep > max_control_steps:
                break
            robot_pos = env.data.xpos[env.pelvis_id][:2]
            rpy = quat_to_euler(env.quat)
            h = float(env.data.xpos[env.pelvis_id][2])
            if h < 0.4 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                fell = True
                break
            dist = float(np.linalg.norm(waypoint - robot_pos))
            min_dist = min(min_dist, dist)
            if dist < arrival:
                arrived = True
                break
            bearing = np.arctan2(waypoint[1] - robot_pos[1], waypoint[0] - robot_pos[0])
            vx, vy, heading = pid.compute_action(current_pos=robot_pos, current_yaw=rpy[2],
                                                 target_pos=waypoint, target_yaw=bearing, dt=env.control_dt)
            env.viewer.commands[0] = vx
            env.viewer.commands[2] = vy
            env.viewer.commands[1] = heading
            obs = env._compute_observation()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                eh = torch.tensor(np.array(env.extra_history).flatten().copy(),
                                  dtype=torch.float).view(1, -1).to(dev)
                raw = env.policy_jit(obs_t, eh).cpu().numpy().squeeze()
            raw = np.clip(raw, -40., 40.)
            # CRITICAL: AMO's observation depends on last_action; omitting this makes it fall.
            env.last_action = np.concatenate([raw.copy(), (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale])
            scaled = raw * env.action_scale
            pd_target = np.concatenate([scaled, np.zeros(8)]) + env.default_dof_pos
            pd_target[15:] = (1 - env.arm_blend) * env.prev_arm_action + env.arm_blend * env.arm_action
            env.arm_blend = min(1.0, env.arm_blend + 0.01)
            env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
            if env._in_place_stand and np.any(np.abs(env.gait_cycle - 0.25) < 0.05):
                env.gait_cycle = np.array([0.25, 0.25])
            if not env._in_place_stand and np.all(np.abs(env.gait_cycle - 0.25) < 0.05):
                env.gait_cycle = np.array([0.25, 0.75])
        torque = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
        torque = np.clip(torque, -env.torque_limits, env.torque_limits)
        env.data.ctrl = torque
        mujoco.mj_step(env.model, env.data)

    if hasattr(env, "close"):
        env.close()
    return {"arrived": arrived, "fell": fell, "min_dist": min_dist, "steps": cstep,
            "waypoint": [round(float(x), 2) for x in waypoint]}
