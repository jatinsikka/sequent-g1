"""
Gymnasium wrapper for G1 button pressing environment.

This environment trains the robot to press buttons on the control panel.
The robot is initialized standing in front of the button panel.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any, Optional

from play_amo import HumanoidEnv, quat_to_euler
from unified_env import UnifiedHumanoidEnv
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS

# gripper tendon command: keep the gripper CLOSED so it is a solid pusher for pressing.
GRIP_CLOSED = 255.0


class ButtonPressEnv(gym.Env):
    """
    Environment for training button pressing.
    
    The robot starts positioned in front of the control panel and must
    extend its arm to press the target button.
    
    Action space: 8 arm joint position offsets in [-1, 1]
    Observation space: arm state + hand positions + button position + button state
    """
    
    # Button joint names in MuJoCo
    BUTTON_JOINTS = {
        "button_red": "push_button_red_joint",
        "button_green": "push_button_green_joint",
        "button_yellow": "push_button_yellow_joint",
        "button_blue": "push_button_blue_joint",
    }
    
    BUTTON_BODIES = {
        "button_red": "push_button_red",
        "button_green": "push_button_green",
        "button_yellow": "push_button_yellow",
        "button_blue": "push_button_blue",
    }
    
    def __init__(
        self,
        button_name: str = "button_red",
        reward_fn: Optional[ButtonPressRewardFunction] = None,
        freeze_arm: str = "left",
        max_episode_steps: int = 200,
        headless: bool = True,
        device: str = None,
        unified: bool = True,
        reset_in_contact: bool = True,
    ):
        """
        Initialize button press environment.

        Args:
            button_name: Which button to target (button_red, button_green, etc.)
            reward_fn: Reward function instance
            freeze_arm: Which arm to freeze ("left", "right", "none")
            max_episode_steps: Maximum steps per episode
            headless: Run without visualization
            device: Torch device
            unified: use the UNIFIED gripper-humanoid (g1_amo_gripper.mjb) instead of the
                rubber-hand g1.xml. When True the right "hand" is the CLOSED gripper's
                pad-midpoint and control routes through UnifiedHumanoidEnv.apply_ctrl.
            reset_in_contact: (unified only) after the AMO settle, IK-servo the CLOSED gripper
                ONTO the button cap so the episode STARTS in contact. Kills the RL
                reach-from-distance contact dead-zone: RL only learns push-in + hold.
                Default on (for training).
        """
        super().__init__()

        self.button_name = button_name
        self.freeze_arm = freeze_arm.lower()
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.reset_in_contact = reset_in_contact
        self.unified = unified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate button name
        if button_name not in BUTTON_POSITIONS:
            raise ValueError(f"Unknown button: {button_name}")
        
        self.button_position = BUTTON_POSITIONS[button_name]
        self.button_joint_name = self.BUTTON_JOINTS[button_name]
        self.button_body_name = self.BUTTON_BODIES[button_name]
        
        # Initialize reward function. With reset-in-contact, use the contact-mode shaping
        # (dense press-from-contact + hold, reach/approach/distance terms disabled).
        if reward_fn is None:
            self.reward_fn = ButtonPressRewardFunction(
                button_position=self.button_position,
                contact_mode=bool(unified and reset_in_contact),
            )
        else:
            self.reward_fn = reward_fn
        
        # Load AMO policy for leg control
        self.policy_jit = torch.jit.load("amo_jit.pt", map_location=self.device)
        
        # Create underlying HumanoidEnv with headless flag.
        # unified -> gripper-humanoid (g1_amo_gripper.mjb) with the explicit-address AMO
        # DOF reads (so AMO still balances despite the appended wrist+gripper DOFs).
        EnvCls = UnifiedHumanoidEnv if self.unified else HumanoidEnv
        self.env = EnvCls(
            policy_jit=self.policy_jit,
            robot_type="g1",
            device=self.device,
            headless=headless,
        )
        
        # Get button joint and body IDs
        self.button_joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, self.button_joint_name
        )
        self.button_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.button_body_name
        )
        
        # Get hand body IDs. Left stays the rubber hand; the RIGHT "hand" is the gripper
        # (pad-midpoint) when unified — right_rubber_hand is hidden on the gripper model.
        self.left_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand'
        )
        self.right_hand_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand'
        )
        
        print(f"[ENV] Button joint ID: {self.button_joint_id}, Body ID: {self.button_body_id}")
        print(f"[ENV] Hand IDs - Left: {self.left_hand_id}, Right: {self.right_hand_id}")
        
        # Arm joint configuration
        self.num_arm_joints = 8
        self.arm_joint_start = 15  # Arm joints start at index 15 in qpos
        self.arm_joint_end = 23
        
        # Action space: 8 arm joint offsets
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_arm_joints,),
            dtype=np.float32,
        )
        
        # Observation space:
        # - arm joint positions (8)
        # - arm joint velocities (8)
        # - left hand position (3)
        # - right hand position (3)
        # - button position (3)
        # - hand to button vectors (6)
        # - button displacement (1)
        # Total: 32
        obs_dim = 8 + 8 + 3 + 3 + 3 + 6 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0
        self.action_scale = 0.5  # Increased for larger arm movements
        
        # Robot positioning for button panel
        # Position robot DIRECTLY in front of the target button
        button_x = self.button_position[0]
        # Robot should be ~0.35m in front of buttons (buttons at y=-1.85)
        # Aligned with g1.xml keyframe position
        # Spawn ~0.2m closer than before (was y=-1.50) and nudge x to absorb the
        # forward/lateral reach residual + the ~8.6cm pelvis recoil measured when the
        # arm extends (diagnosed 2026-06-19, see _reachgrid.py / button-press memory).
        self.robot_start_pos = np.array([button_x - 0.08, -1.62, 0.793])
        # -90° yaw rotation (facing -Y direction toward buttons)
        # Quaternion [w, x, y, z] for -90° around Z axis
        self.robot_start_quat = np.array([0.7071, 0, 0, -0.7071])
        # AMO commands[1] is an ABSOLUTE world target-yaw setpoint. Hold the spawn
        # heading so dyaw=0; zeroing it while spawned ~90 deg rotated makes AMO fight
        # a persistent yaw error and topple (confirmed via _diag.py).
        self.target_yaw = float(quat_to_euler(self.robot_start_quat)[2])
        # Forward torso-pitch command (AMO commands[5]); leans the whole body toward the
        # panel so the hand can press the button face straight-on (natural press). AMO's
        # adapter is conditioned on torso pitch (play_amo.py:323), so it balances the lean.
        self.torso_lean = 0.0
        # Direct waist-pitch (dof 14) bias on the control target — a STRONGER forward lean
        # than AMO's command, to close the ~6cm depth gap for a straight-on press.
        self.waist_lean = 0.0
        
        # Determine which arm to use based on button position
        # For buttons on the left (x < 0), use left arm (freeze right)
        # For buttons on the right (x >= 0), use right arm (freeze left)
        if button_x < 0:
            # Left side button - override freeze_arm to use LEFT arm
            self.freeze_arm = "right"  # Freeze right, use left
            print(f"[ENV] Button at x={button_x:.2f} -> Using LEFT arm")
        else:
            # Right side button - use RIGHT arm
            self.freeze_arm = "left"  # Freeze left, use right
            print(f"[ENV] Button at x={button_x:.2f} -> Using RIGHT arm")
        
        # Reach-ready default arm posture: bias the ACTIVE arm toward the button so
        # the pressing configuration sits centered in the small (±0.5rad) action
        # envelope. Diagnosed 2026-06-19: the neutral default pose + ±0.5rad action
        # cannot raise the hand to button height, though FK shows the button IS
        # reachable (0.031m). Offsets are (FK-winning joints − default) for the left
        # arm reaching button_red; the active arm gets the bias, frozen arm stays neutral.
        # Offsets are (natural-reach joints − default) for the LEFT arm reaching
        # button_red at spawn y=-1.56, where "natural" = the reaching config closest
        # to the neutral default (avoids contorted IK branches). Co-solved + dynamically
        # verified to depress the button base-upright (_presssolve.py, 2026-06-19).
        self.arm_reach_bias = np.zeros(self.num_arm_joints, dtype=np.float32)
        _LEFT_REACH = np.array([0.02, -1.56, -0.80, -0.61], dtype=np.float32)
        if self.freeze_arm == "right":   # left arm active (button on left, x<0)
            self.arm_reach_bias[:4] = _LEFT_REACH
        else:                            # right arm active (mirror roll & yaw sign)
            self.arm_reach_bias[4:] = _LEFT_REACH * np.array([1, -1, -1, 1], dtype=np.float32)

        # Track initial button state for proper displacement calculation
        self.initial_button_displacement = None

        # cap-face contact target for reset-in-contact: the button cap top-geom world pos,
        # nudged ~1cm toward the robot (+Y) so the gripper pads seat ON the cap face, not
        # buried in it. Only meaningful for the unified gripper + right-side buttons.
        self.cap_geom_name = f"{self.button_body_name}_top"
        self._solved_wrist = np.zeros(3)

        # ARM-HOLD stiffness (unified reset-in-contact): the AMO arm gains (shoulder_yaw kp=40)
        # are too soft to statically HOLD the raised reach onto a z=0.90 cap — the arm sags ~6cm
        # and never touches. Boost ONLY the 4 right shoulder/elbow gains for the contact-hold
        # (legs/waist/left-arm untouched, so AMO balance is unaffected). Used when unified.
        if self.unified:
            self._arm_hold_stiffness = self.env.stiffness.astype(float).copy()
            self._arm_hold_stiffness[19:23] *= 4.0   # right shoulder p/r/y + elbow

    def _amo_arm_step(self, arm_qtarget_right, wrist_target=None, grip_cmd=GRIP_CLOSED):
        """One control step used by the reset IK servo: AMO drives the legs (balance) while
        the 4 RIGHT shoulder/elbow joints PD-track `arm_qtarget_right`; wrist held toward
        `wrist_target` (if given) + gripper CLOSED via apply_ctrl. Mirrors the step()/settle
        leg-control loop so AMO keeps its state consistent."""
        if wrist_target is not None:
            self.env.wrist_target = np.asarray(wrist_target, dtype=float)
        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean
        self.env._extract_state()
        amo_obs = self.env._compute_observation()
        obs_tensor = torch.from_numpy(amo_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(
                np.array(self.env.extra_history).flatten().copy(), dtype=torch.float
            ).view(1, -1).to(self.device)
            leg_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        leg_action = np.clip(leg_action, -40.0, 40.0)
        scaled_leg = leg_action * self.env.action_scale
        self.env.last_action = np.concatenate([
            leg_action.copy(),
            (self.env.dof_pos[15:] - self.env.default_dof_pos[15:]) / self.env.action_scale])
        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] = scaled_leg + self.env.default_dof_pos[:15]
        pd_target[14] += self.waist_lean
        # right shoulder/elbow are AMO dofs 19..22; PD-track them to the IK target
        pd_target[19:23] = arm_qtarget_right
        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0)
        if self.env._in_place_stand and np.any(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.25])
        stiff = getattr(self, "_arm_hold_stiffness", self.env.stiffness)
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * stiff - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            self.env.apply_ctrl(torque, grip_cmd)
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()

    def _servo_to_contact(self, n_align=200, n_press=120):
        """IK-servo (DLS on the gripper pad-midpoint) the CLOSED gripper onto the blue button
        cap face, AMO balancing throughout. Two stages so the pad-mid seats on the CAP (which
        protrudes +Y past the wide housing base) instead of jamming on the base rim:
          1. ALIGN: drive to a front standoff aligned in x,z with the cap center (offset +Y).
          2. PRESS-SEAT: move in -Y to the cap face until contact.
        Returns the achieved 4-DOF right-arm qpos so the caller plants arm_reach_bias there
        (episode then STARTS in contact)."""
        cap_gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, self.cap_geom_name)

        def cap(): return self.env.data.geom_xpos[cap_gid].copy()

        # PROGRESSIVE STRAIGHT-IN approach (7-DOF arm+wrist KINEMATIC IK + boosted-hold settle):
        # the fixed downward wrist tops out ~6cm below the z=0.90 cap, so the IK tilts the wrist
        # to orient the pad onto the cap face. Approaching straight down the cap axis at
        # decreasing +Y standoff (with the boosted right-arm hold each step) keeps the pad
        # centered in x,z as it seats on the protruding cap — instead of sagging low and
        # catching the wide base rim. Each standoff is solved fresh + settled.
        bj = self.button_joint_id
        a4 = self.env.data.qpos[self.env.rarm_qadr].copy()
        w3 = self.env.wrist_target.copy()
        # approach to a LIGHT-contact standoff: stop as soon as the pad just TOUCHES the cap
        # (button begins to move), so the episode starts AT contact with the full ~2cm press
        # still available for RL — not already bottomed out. Check disp every sub-step and
        # halt the whole approach on first touch.
        touched = False
        for standoff in [0.12, 0.09, 0.06, 0.045, 0.035, 0.028, 0.022]:
            a4, w3, err = self.env.solve_right_arm7_ik(cap() + np.array([0.0, standoff, 0.0]))
            for _ in range(40):
                self._amo_arm_step(a4, wrist_target=w3)
                if float(self.env.data.qpos[bj]) > 0.002:   # cap just began to move -> contact
                    touched = True
                    break
            if touched:
                break
        self._solved_wrist = w3.copy()
        # return the SOLVED arm target (episode PD sags onto contact the same way)
        return a4

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo to initial keyframe
        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)

        # reset the wrist hold to level (the reset-in-contact servo may have tilted it last episode)
        if self.unified:
            self.env.wrist_target = np.zeros(3)

        # Position robot in front of button panel
        # Find pelvis (robot root) in qpos dynamically — the interactive-objects block
        # changed size when the lever joint was added, so the old hardcoded 46 is unsafe.
        robot_qpos_start = self.env.model.jnt_qposadr[mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
        self.env.data.qpos[robot_qpos_start:robot_qpos_start+3] = self.robot_start_pos
        # -90° yaw to face -Y direction (toward buttons)
        # Using quaternion defined in __init__
        self.env.data.qpos[robot_qpos_start+3:robot_qpos_start+7] = self.robot_start_quat
        
        # IMPORTANT: Zero out velocities for stable start
        self.env.data.qvel[:] = 0.0
        
        # Reset button to unpressed position (0 displacement)
        self.env.data.qpos[self.button_joint_id] = 0.0
        
        # Run forward kinematics to update body positions
        mujoco.mj_forward(self.env.model, self.env.data)
        
        # Store initial button position (should be 0 now, but track for safety)
        self.initial_button_displacement = self.env.data.qpos[self.button_joint_id]
        
        # Reset env state
        self.env._extract_state()
        self.env.last_action = np.zeros(self.env.num_dofs, dtype=np.float32)
        self.env.arm_action = self.env.default_dof_pos[15:].copy()
        self.env.prev_arm_action = self.env.default_dof_pos[15:].copy()
        self.env.arm_blend = 0.0
        self.env._in_place_stand = True
        self.env.gait_cycle = np.array([0.25, 0.25])
        
        # Reset viewer commands (stand still), but keep the heading setpoint
        # MockViewer in headless mode has commands attribute
        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean
        
        # Reset history buffers
        self.env.proprio_history = deque(maxlen=self.env.history_len)
        self.env.extra_history = deque(maxlen=self.env.extra_history_len)
        for _ in range(self.env.history_len):
            self.env.proprio_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        for _ in range(self.env.extra_history_len):
            self.env.extra_history.append(np.zeros(self.env.n_proprio, dtype=np.float32))
        
        # Let AMO settle for a few steps to stabilize balance
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
            scaled_leg_action = leg_action * self.env.action_scale  # AMO leg scale (0.25), NOT the wrapper's arm scale (0.5)
            
            pd_target = self.env.default_dof_pos.copy()
            pd_target[:15] = scaled_leg_action + self.env.default_dof_pos[:15]
            pd_target[14] += self.waist_lean  # forward waist lean toward the panel

            for _ in range(self.env.sim_decimation):
                torque = (pd_target - self.env.dof_pos) * self.env.stiffness - self.env.dof_vel * self.env.damping
                torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
                if self.unified:
                    self.env.apply_ctrl(torque, GRIP_CLOSED)  # AMO torque + wrist hold + CLOSED gripper
                else:
                    self.env.data.ctrl = torque
                mujoco.mj_step(self.env.model, self.env.data)
                self.env._extract_state()

        # RESET-IN-CONTACT: IK the CLOSED gripper onto the button cap so the episode STARTS
        # touching it (kills the RL reach-from-distance contact dead-zone). Then plant the
        # achieved right-arm pose into arm_reach_bias so ZERO action HOLDS contact; RL's
        # action only has to push further in + hold.
        if self.unified and self.reset_in_contact:
            solved_right = self._servo_to_contact()
            # arm_reach_bias is (target - default) for the 8 arm dofs; set the right 4 so
            # default + bias == the SOLVED seated-contact pose (episode PD sags onto contact).
            self.arm_reach_bias[4:] = (solved_right - self.env.default_dof_pos[19:23]).astype(np.float32)
            # hold the solved wrist tilt through the episode (apply_ctrl PD-holds wrist_target)
            self.env.wrist_target = self._solved_wrist.copy()
            # re-baseline displacement AFTER the servo (any seating displacement is the new zero)
            self.initial_button_displacement = self.env.data.qpos[self.button_joint_id]

        # Reset episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0

        # Reset reward function
        self.reward_fn.reset()

        return self._get_obs(), {}
    
    def _right_hand_pos(self) -> np.ndarray:
        """Right 'hand' contact point: gripper pad-midpoint (unified) or rubber hand."""
        if self.unified:
            return self.env.gripper_point()
        return self.env.data.xpos[self.right_hand_id]

    def _get_button_displacement(self) -> float:
        """Get the relative button displacement from initial position."""
        current_disp = self.env.data.qpos[self.button_joint_id]
        # Return displacement relative to initial (should start at 0)
        return current_disp - self.initial_button_displacement
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        # Arm joint positions and velocities
        arm_pos = self.env.dof_pos[self.arm_joint_start:self.arm_joint_end]
        arm_vel = self.env.dof_vel[self.arm_joint_start:self.arm_joint_end]
        
        # Hand positions
        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self._right_hand_pos()
        
        # Button position
        button_pos = self.env.data.xpos[self.button_body_id]
        
        # Vectors from hands to button
        left_to_button = button_pos - left_hand_pos
        right_to_button = button_pos - right_hand_pos
        
        # Button displacement (relative to initial - starts at 0)
        button_displacement = self._get_button_displacement()
        
        obs = np.concatenate([
            arm_pos,                    # 8
            arm_vel * 0.1,              # 8 (scaled)
            left_hand_pos,              # 3
            right_hand_pos,             # 3
            button_pos,                 # 3
            left_to_button,             # 3
            right_to_button,            # 3
            [button_displacement],      # 1
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self.episode_steps += 1
        
        # Process arm action
        action = np.array(action, dtype=np.float32)
        
        # Apply freeze_arm logic
        if self.freeze_arm == "left":
            action[:4] = 0.0  # Freeze left arm joints
        elif self.freeze_arm == "right":
            action[4:] = 0.0  # Freeze right arm joints
        
        # Scale action
        scaled_arm_action = action * self.action_scale
        
        # === LEG CONTROL FROM AMO (standing still) ===
        # Zero velocity commands = stand in place (but keep heading setpoint)
        # Use viewer.commands (MockViewer in headless mode has this)
        self.env.viewer.commands[:] = 0.0
        self.env.viewer.commands[1] = self.target_yaw
        self.env.viewer.commands[5] = self.torso_lean
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
        scaled_leg_action = leg_action * self.env.action_scale  # AMO leg scale (0.25), NOT the wrapper's arm scale (0.5)
        
        # Update last_action buffer (needed by _compute_observation for next step)
        self.env.last_action = np.concatenate([
            leg_action.copy(),
            (self.env.dof_pos[15:] - self.env.default_dof_pos[15:]) / self.env.action_scale
        ])
        
        # Combine leg + arm actions
        pd_target = self.env.default_dof_pos.copy()
        pd_target[:15] = scaled_leg_action + self.env.default_dof_pos[:15]  # Legs from AMO
        pd_target[14] += self.waist_lean  # forward waist lean toward the panel
        pd_target[15:] = self.env.default_dof_pos[15:] + self.arm_reach_bias + scaled_arm_action  # reach-ready baseline + RL fine-tune
        
        # Update gait cycle (needed for standing balance)
        self.env.gait_cycle = np.remainder(
            self.env.gait_cycle + self.env.control_dt * self.env.gait_freq, 1.0
        )
        # Gait sync — EXACTLY matches the working walk_and_grasp loop (lines 212-215)
        if self.env._in_place_stand and np.any(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.25])
        if not self.env._in_place_stand and np.all(np.abs(self.env.gait_cycle - 0.25) < 0.05):
            self.env.gait_cycle = np.array([0.25, 0.75])

        # Step simulation. Use boosted right-arm hold stiffness for the contact-hold task.
        stiff = self._arm_hold_stiffness if (self.unified and self.reset_in_contact) else self.env.stiffness
        for _ in range(self.env.sim_decimation):
            torque = (pd_target - self.env.dof_pos) * stiff - self.env.dof_vel * self.env.damping
            torque = np.clip(torque, -self.env.torque_limits, self.env.torque_limits)
            if self.unified:
                self.env.apply_ctrl(torque, GRIP_CLOSED)  # AMO torque + wrist hold + CLOSED gripper
            else:
                self.env.data.ctrl = torque
            mujoco.mj_step(self.env.model, self.env.data)
            self.env._extract_state()
        
        # Get state for reward
        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        rpy = np.zeros(3)  # Simplified
        ang_vel = self.env.ang_vel
        
        left_hand_pos = self.env.data.xpos[self.left_hand_id]
        right_hand_pos = self._right_hand_pos()
        
        # Get button state (relative displacement from initial)
        button_displacement = self._get_button_displacement()
        button_pos = self.env.data.xpos[self.button_body_id]
        
        # Compute reward
        reward, info = self.reward_fn.compute_reward(
            position=robot_pos,
            rpy=rpy,
            ang_vel=ang_vel,
            action=action,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            button_displacement=button_displacement,
            current_button_pos=button_pos,
        )
        
        self.episode_return += reward
        
        # Check termination
        terminated = False
        truncated = False
        
        # Success: button pressed
        if self.reward_fn.button_pressed:
            info['success'] = True
            # Don't terminate immediately - let robot hold the press
        
        # Failure: robot fell
        if robot_pos[2] < 0.4:
            terminated = True
            info['fell'] = True
        
        # Truncation: max steps
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
        
        # (gait is already advanced once per control step above, matching walk_and_grasp;
        #  the previous duplicate gait update here advanced it twice per step and was removed)
        
        # Render if not headless
        if not self.headless and hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.cam.lookat = robot_pos.astype(np.float32)
            self.env.viewer.render()
        
        info['episode_return'] = self.episode_return
        info['episode_steps'] = self.episode_steps
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def close(self):
        """Clean up."""
        if hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.close()
    
    def render(self):
        """Render the environment."""
        if hasattr(self.env, 'viewer') and self.env.viewer is not None:
            self.env.viewer.render()
    
    def render_frame(self, width: int = 480, height: int = 360) -> np.ndarray:
        """
        Render a frame for video recording.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            RGB frame as numpy array (H, W, 3)
        """
        # Create offscreen renderer if needed
        if not hasattr(self, '_renderer') or self._renderer is None:
            self._renderer = mujoco.Renderer(self.env.model, height, width)
        
        # Set camera to view the robot and button
        robot_pos = self.env.data.xpos[self.env.pelvis_id]
        button_pos = self.env.data.xpos[self.button_body_id]
        
        # Look at the button panel (in front of the robot)
        lookat = np.array([
            button_pos[0],      # Center on button X
            button_pos[1] + 0.2, # Slightly in front of button
            0.85                # Button height
        ])
        
        # Create camera object
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = 1.2  # Close view to see hand interaction
        cam.azimuth = 0     # Front view (looking in +Y direction at the buttons)
        cam.elevation = -15  # Slightly above
        
        # Update scene with custom camera
        self._renderer.update_scene(self.env.data, camera=cam)
        
        # Render and return pixels
        return self._renderer.render()
