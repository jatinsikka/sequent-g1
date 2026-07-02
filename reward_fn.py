"""
Reward function for G1 humanoid RL training.

This module provides modular reward functions for:
  1. Grasping objects (RewardFunction) - distance to target, hand proximity
  2. Pressing buttons (ButtonPressRewardFunction) - button displacement, hand proximity

Author: Jatin Sikka
"""

import numpy as np
from typing import Tuple, Optional


class RewardFunction:
    """
    Computes rewards for G1 humanoid training.
    """
    
    def __init__(
        self,
        target_position: Tuple[float, float] = (1.2, 1.35),  # Screwdriver on right table
        target_object_pos: Tuple[float, float, float] = (1.2, 1.35, 0.74),  # 3D position of screwdriver
        desired_height: float = 0.75,
        position_weight: float = 1.0,
        velocity_weight: float = 5.0,  # Reward for moving toward target
        hand_proximity_weight: float = 2.0,  # Reward for getting hands close to object
        action_penalty: float = 0.001,  # Reduced to encourage larger actions
        alive_bonus: float = 0.5,  # Bonus for staying upright
    ):
        """
        Initialize the reward function.
        
        Args:
            target_position: Target (x, y) position for the robot torso (default: screwdriver).
            target_object_pos: 3D position of the object to pick up.
            desired_height: Desired torso height (z-coordinate).
            position_weight: Coefficient for position tracking reward.
            velocity_weight: Coefficient for velocity toward target reward.
            hand_proximity_weight: Coefficient for hand-to-object distance reward.
            action_penalty: Coefficient for action penalty.
            alive_bonus: Constant reward for staying upright.
        """
        self.target_position = np.array(target_position)
        self.target_object_pos = np.array(target_object_pos)
        self.desired_height = desired_height
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.hand_proximity_weight = hand_proximity_weight
        self.action_penalty = action_penalty
        self.alive_bonus = alive_bonus
        
        # Store previous position for velocity calculation
        self.prev_position = None
    
    def compute_reward(
        self,
        position: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
        action: np.ndarray,
        left_hand_pos: Optional[np.ndarray] = None,
        right_hand_pos: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute total reward for the current step.
        
        Args:
            position: Robot torso position [x, y, z].
            rpy: Robot torso orientation in roll-pitch-yaw [rad].
            ang_vel: Robot angular velocity [rad/s].
            action: Action taken (motor commands).
            left_hand_pos: Left hand position [x, y, z] (optional).
            right_hand_pos: Right hand position [x, y, z] (optional).
        
        Returns:
            float: Total reward for this step.
        """
        r_position = self._compute_position_reward(position)
        r_velocity = self._compute_velocity_reward(position)
        r_action = self._compute_action_penalty(action)
        r_hand = self._compute_hand_proximity_reward(left_hand_pos, right_hand_pos)
        
        total_reward = (
            self.alive_bonus
            + self.position_weight * r_position
            + self.velocity_weight * r_velocity
            + self.hand_proximity_weight * r_hand
            - self.action_penalty * r_action
        )
        
        return total_reward
    
    def _compute_velocity_reward(self, position: np.ndarray) -> float:
        """
        Compute reward for moving toward the target.
        
        This provides immediate feedback for motion, unlike position reward
        which only improves once the robot gets very close.
        
        Args:
            position: Current robot torso position [x, y, z].
        
        Returns:
            float: Velocity toward target reward.
        """
        current_xy = position[:2]
        
        if self.prev_position is None:
            self.prev_position = current_xy.copy()
            return 0.0
        
        # Vector from previous position to current position (velocity direction)
        velocity_vec = current_xy - self.prev_position
        
        # Vector from current position to target
        to_target = self.target_position - current_xy
        to_target_norm = np.linalg.norm(to_target)
        
        if to_target_norm < 0.01:  # At target
            self.prev_position = current_xy.copy()
            return 1.0
        
        # Normalize to_target
        to_target_unit = to_target / to_target_norm
        
        # Project velocity onto direction toward target
        velocity_toward_target = np.dot(velocity_vec, to_target_unit)
        
        # Scale the reward (positive = moving toward, negative = moving away)
        # Multiply by 50 to make it significant (0.02m/step at 0.5m/s * 0.02s)
        velocity_reward = velocity_toward_target * 50.0
        
        self.prev_position = current_xy.copy()
        return velocity_reward
    
    def reset(self):
        """Reset the reward function state (call on environment reset)."""
        self.prev_position = None
    
    def _compute_hand_proximity_reward(
        self, 
        left_hand_pos: Optional[np.ndarray], 
        right_hand_pos: Optional[np.ndarray]
    ) -> float:
        """
        Compute reward for getting hands close to the target object.
        
        Args:
            left_hand_pos: Left hand position [x, y, z].
            right_hand_pos: Right hand position [x, y, z].
        
        Returns:
            float: Hand proximity reward (0 if hand positions not provided).
        """
        if left_hand_pos is None and right_hand_pos is None:
            return 0.0
        
        # Compute distance from each hand to target object
        distances = []
        if left_hand_pos is not None:
            left_dist = np.linalg.norm(left_hand_pos - self.target_object_pos)
            distances.append(left_dist)
        if right_hand_pos is not None:
            right_dist = np.linalg.norm(right_hand_pos - self.target_object_pos)
            distances.append(right_dist)
        
        # Use minimum distance (closest hand)
        min_distance = min(distances)
        
        # Exponential reward: high reward when hand is very close
        # exp(-d) gives 1.0 at d=0, ~0.37 at d=1m, ~0.14 at d=2m
        hand_reward = np.exp(-2.0 * min_distance)  # Steeper falloff
        
        return hand_reward
    
    def _compute_position_reward(self, position: np.ndarray) -> float:
        """
        Compute position tracking reward (negative distance to target).
        
        Args:
            position: Current robot torso position [x, y, z].
        
        Returns:
            float: Position reward (negative distance scaled).
        """
        # Extract x, y from position (ignore z)
        current_xy = position[:2]
        
        # Distance to target
        distance = np.linalg.norm(current_xy - self.target_position)
        
        # Exponential decay: reward is higher when robot is closer
        # Maximum reward of 1.0 when at target, decays with distance
        position_reward = np.exp(-distance)
        
        return position_reward
    
    def _compute_action_penalty(self, action: np.ndarray) -> float:
        """
        Compute action penalty to encourage smooth, energy-efficient motion.
        
        Args:
            action: Action vector (typically 8-dim for G1).
        
        Returns:
            float: Action penalty (sum of squared actions).
        """
        # L2 norm of actions to penalize large commands
        action_cost = np.linalg.norm(action) ** 2
        return action_cost
    
    def set_target_position(self, target: Tuple[float, float]):
        """
        Update the target position for tracking.
        
        Args:
            target: New target (x, y) position.
        """
        self.target_position = np.array(target)
    
    def __repr__(self) -> str:
        """Return string representation of reward function."""
        return (
            f"RewardFunction(target={self.target_position}, "
            f"desired_height={self.desired_height}, "
            f"position_w={self.position_weight}, "
            f"hand_prox_w={self.hand_proximity_weight}, "
            f"action_penalty={self.action_penalty})"
        )


class ButtonPressRewardFunction:
    """
    Reward function for button pressing task.
    
    The robot needs to:
    1. Move hand close to the button
    2. Push the button (displace it along its slide joint axis)
    
    Buttons are slide joints that move along Y axis (toward the machine).
    """
    
    def __init__(
        self,
        button_position: Tuple[float, float, float] = (-0.45, -1.85, 1.0),  # Red button default
        press_threshold: float = 0.02,  # Button counts as "pressed" if displaced > 2cm
        target_press_depth: float = 0.05,  # A SOLID press: per-step press reward saturates here
        hand_proximity_weight: float = 10.0,  # Increased to encourage reaching
        press_reward: float = 100.0,  # Per-step reward at full target depth (scales with depth)
        hold_bonus: float = 30.0,  # bp-v2: steep per-step income while past the press threshold
        first_press_bonus: float = 50.0,  # One-time bonus on first crossing the threshold
        balance_weight: float = 3.0,  # Penalty for base drifting off spawn (counter reach-recoil)
        action_penalty: float = 0.0001,  # Reduced to not discourage movement
        alive_bonus: float = 0.1,  # Reduced to make reaching more important
        contact_mode: bool = False,  # RESET-IN-CONTACT reward shaping (see below)
    ):
        """
        Initialize button press reward function.
        
        Args:
            button_position: 3D position of the button.
            press_threshold: How far button must move to count as "pressed".
            hand_proximity_weight: Reward weight for hand approaching button.
            press_reward: Bonus reward when button is successfully pressed.
            action_penalty: Penalty for large actions.
            alive_bonus: Bonus for staying upright.
        """
        self.button_position = np.array(button_position)
        self.press_threshold = press_threshold
        self.target_press_depth = target_press_depth
        # CONTACT MODE (reset-in-contact): the episode STARTS with the gripper on the button,
        # so RL only has to push-in + HOLD. Zero the reach/approach/distance terms (they'd pay
        # the policy to retract and "re-approach", fighting the press), and keep the dense
        # depth-monotonic press + first-press bonus + a per-step HOLD income past the threshold
        # + the balance/upright penalties. This is the clean "IK reach + RL contact" split.
        self.contact_mode = contact_mode
        if contact_mode:
            hand_proximity_weight = 0.0
            hold_bonus = max(hold_bonus, 30.0)   # steep sustained-hold income while pressed
        self.hand_proximity_weight = hand_proximity_weight
        self.press_reward = press_reward
        self.hold_bonus = hold_bonus
        self.first_press_bonus = first_press_bonus
        self.balance_weight = balance_weight
        self.action_penalty = action_penalty
        self.alive_bonus = alive_bonus

        # Track button state
        self.button_pressed = False
        self.max_button_displacement = 0.0
        self.prev_hand_distance = None
        self.init_base_xy = None  # base xy at episode start, for the balance/recoil penalty
    
    def compute_reward(
        self,
        position: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
        action: np.ndarray,
        left_hand_pos: Optional[np.ndarray] = None,
        right_hand_pos: Optional[np.ndarray] = None,
        button_displacement: float = 0.0,
        current_button_pos: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        """
        Compute reward for button pressing.
        
        Args:
            position: Robot torso position [x, y, z].
            rpy: Robot orientation [roll, pitch, yaw].
            ang_vel: Angular velocity.
            action: Action taken.
            left_hand_pos: Left hand position.
            right_hand_pos: Right hand position.
            button_displacement: How far the button has been pushed (joint position).
            current_button_pos: Current 3D position of button body.
        
        Returns:
            Tuple of (total_reward, info_dict).
        """
        info = {}
        
        # Use current button position if available, otherwise use initial
        target_pos = current_button_pos if current_button_pos is not None else self.button_position
        
        # 1. Hand proximity reward
        r_hand = self._compute_hand_proximity_reward(left_hand_pos, right_hand_pos, target_pos)
        info['r_hand_proximity'] = r_hand
        
        # 2. Approach velocity reward (bonus for moving hand toward button)
        r_approach = self._compute_approach_reward(left_hand_pos, right_hand_pos, target_pos)
        info['r_approach'] = r_approach
        
        # 3. Button press reward — DEPTH-MONOTONIC, no flat plateau.
        # Per-step reward grows linearly with how deep the button is held, saturating
        # at target_press_depth. A graze at the 2cm threshold earns far less than a
        # solid 5cm press, so RL is pushed toward a firm, held press rather than a
        # brittle threshold-graze (the failure mode seen on the grasp policy).
        self.max_button_displacement = max(self.max_button_displacement, button_displacement)
        depth_frac = np.clip(button_displacement / self.target_press_depth, 0.0, 1.0)
        r_press = self.press_reward * depth_frac
        # bp-v3: reverted v2's hold_bonus CLIFF (it destabilized value learning -> regression).
        # The depth term above is already a SMOOTH sustained-hold reward: every step the
        # button is held deep accumulates press_reward*depth_frac, no discontinuity. The new
        # lever for bp-v3 is the BC warm-start (train_button_bc.py), not the reward.
        if button_displacement > self.press_threshold and not self.button_pressed:
            r_press += self.first_press_bonus  # one-time: it's doable
            self.button_pressed = True
            print(f"    [REWARD] Button pressed! Displacement: {button_displacement:.4f}m")

        # HOLD income: steady per-step income while the button is held past threshold, scaled by
        # depth — pushes RL toward a firm SUSTAINED press. Was contact_mode-only; now ALWAYS on
        # (the curriculum run never got it, which is why the learned press retracted after pressing).
        if button_displacement > self.press_threshold:
            r_press += self.hold_bonus * depth_frac

        info['r_press'] = r_press
        info['button_displacement'] = button_displacement
        info['button_pressed'] = self.button_pressed
        
        # 4. Action penalty (very small)
        r_action = -self.action_penalty * np.linalg.norm(action) ** 2
        info['r_action'] = r_action
        
        # 5. Distance penalty - penalize the ACTIVE (closest) hand being far from button.
        # (Was hardcoded to right_hand, which is the FROZEN hand for left-side buttons.)
        # In contact mode the episode starts on the button, so a distance penalty would just
        # punish the (fixed) gripper-pad-vs-button-center offset every step — disable it.
        if self.contact_mode:
            r_distance_penalty = 0.0
        else:
            dists = [np.linalg.norm(h - target_pos) for h in (left_hand_pos, right_hand_pos) if h is not None]
            dist_to_button = min(dists) if dists else 1.0
            r_distance_penalty = -dist_to_button * 4.0  # Penalty proportional to distance (stronger pull to contact, anti-park)
        info['r_distance_penalty'] = r_distance_penalty

        # 6. Balance term — penalize the base drifting off its spawn xy. The arm reach
        # recoils the pelvis ~8.6cm backward, which carries the hand short of the button;
        # this nudges RL toward a reach that keeps the base planted.
        if self.init_base_xy is None:
            self.init_base_xy = np.array(position[:2], dtype=float)
        base_drift = float(np.linalg.norm(np.array(position[:2]) - self.init_base_xy))
        r_balance = -self.balance_weight * base_drift
        info['r_balance'] = r_balance
        info['base_drift'] = base_drift

        # Total reward
        total_reward = (
            self.alive_bonus
            + self.hand_proximity_weight * r_hand
            + r_approach
            + r_press
            + r_action
            + r_distance_penalty
            + r_balance
        )

        return total_reward, info
    
    def _compute_hand_proximity_reward(
        self,
        left_hand_pos: Optional[np.ndarray],
        right_hand_pos: Optional[np.ndarray],
        target_pos: np.ndarray,
    ) -> float:
        """Reward for hand being close to button."""
        if left_hand_pos is None and right_hand_pos is None:
            return 0.0
        
        distances = []
        if left_hand_pos is not None:
            distances.append(np.linalg.norm(left_hand_pos - target_pos))
        if right_hand_pos is not None:
            distances.append(np.linalg.norm(right_hand_pos - target_pos))
        
        min_dist = min(distances)

        # CONTACT-GATED proximity (bp-v1): pay NOTHING for hovering back. bp-v0 learned
        # to park ~0.1m out and farm a fat far-field proximity reward instead of pressing.
        # Only reward the hand inside a tight near-contact band, ramping steeply to contact,
        # so committing to the press strictly beats parking. (Approach reward below still
        # gives dense gradient to get the hand into the band.)
        band = 0.05   # NARROW (anti-parking): proximity only near contact, no far plateau to sit on. The
        #               far pull is r_approach (progress) + r_distance_penalty (potential), which PENALIZE a
        #               parked hand (0 proximity + negative distance at 5cm) so RL must commit to the press.
        if min_dist > band:
            return 0.0
        proximity_reward = (band - min_dist) / band  # 0 at band edge -> 1 at contact
        if min_dist < 0.05:
            proximity_reward += np.exp(-20.0 * min_dist)  # steep bonus right at the button
        return proximity_reward
    
    def _compute_approach_reward(
        self,
        left_hand_pos: Optional[np.ndarray],
        right_hand_pos: Optional[np.ndarray],
        target_pos: np.ndarray,
    ) -> float:
        """Reward for moving hand toward button."""
        if left_hand_pos is None and right_hand_pos is None:
            return 0.0
        
        # Get closest hand distance
        distances = []
        if left_hand_pos is not None:
            distances.append(np.linalg.norm(left_hand_pos - target_pos))
        if right_hand_pos is not None:
            distances.append(np.linalg.norm(right_hand_pos - target_pos))
        
        current_dist = min(distances)
        
        if self.prev_hand_distance is None:
            self.prev_hand_distance = current_dist
            return 0.0
        
        # Positive reward for getting closer (scaled up significantly)
        approach_reward = (self.prev_hand_distance - current_dist) * 50.0
        self.prev_hand_distance = current_dist
        
        return approach_reward
    
    def reset(self):
        """Reset reward function state."""
        self.button_pressed = False
        self.max_button_displacement = 0.0
        self.prev_hand_distance = None
        self.init_base_xy = None

    def __repr__(self) -> str:
        return (
            f"ButtonPressRewardFunction(button_pos={self.button_position}, "
            f"press_threshold={self.press_threshold}, "
            f"target_press_depth={self.target_press_depth})"
        )


class LeverPressRewardFunction:
    """
    Reward function for the LEVER task — the twin of ButtonPressRewardFunction.

    Instead of driving a slide-joint displacement, the robot reaches the lever
    handle and rotates its HINGE toward a target angle. Mirrors the button reward
    structure exactly:
      - contact-gated hand->handle proximity (pay nothing for hovering back)
      - dense approach shaping (move hand toward the handle)
      - DEPTH-MONOTONIC angle reward: per-step reward grows with how close the
        hinge angle is to the target (saturating at target), so a firm held turn
        beats a brittle graze.
      - one-time bonus on first crossing the success angle band
      - balance term penalizing base drift off spawn (counter reach-recoil)
    """

    def __init__(
        self,
        handle_position=(0.6, -1.72, 0.87),     # lever grip world pos at angle 0 (default)
        target_angle: float = 0.9,               # rotate hinge to ~0.9 rad (~52 deg)
        angle_tol: float = 0.10,                 # success: within ~0.1 rad of target, held
        hand_proximity_weight: float = 10.0,
        rotate_reward: float = 100.0,            # per-step reward at full target angle
        first_turn_bonus: float = 50.0,          # one-time bonus on first reaching the success band
        hold_bonus: float = 30.0,                # per-step hold income while at/near target angle
        balance_weight: float = 3.0,
        action_penalty: float = 0.0001,
        alive_bonus: float = 0.1,
        contact_mode: bool = False,              # RESET-IN-CONTACT reward shaping (see below)
    ):
        self.handle_position = np.array(handle_position)
        self.target_angle = target_angle
        self.angle_tol = angle_tol
        # CONTACT MODE (reset-in-contact): the episode STARTS with the gripper on the lever
        # grip, so RL only has to drive the hinge through its arc + HOLD at target. Zero the
        # reach/approach/distance terms (they'd pay the policy to retract and "re-approach",
        # fighting the turn), keep the dense angle-toward-target reward + first-turn bonus + a
        # per-step HOLD income near target + the balance/upright penalties.
        self.contact_mode = contact_mode
        if contact_mode:
            hand_proximity_weight = 0.0
            hold_bonus = max(hold_bonus, 30.0)
        self.hold_bonus = hold_bonus
        self.hand_proximity_weight = hand_proximity_weight
        self.rotate_reward = rotate_reward
        self.first_turn_bonus = first_turn_bonus
        self.balance_weight = balance_weight
        self.action_penalty = action_penalty
        self.alive_bonus = alive_bonus

        self.lever_turned = False
        self.max_lever_angle = 0.0
        self.prev_hand_distance = None
        self.init_base_xy = None

    def compute_reward(
        self,
        position: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
        action: np.ndarray,
        left_hand_pos: Optional[np.ndarray] = None,
        right_hand_pos: Optional[np.ndarray] = None,
        lever_angle: float = 0.0,
        current_handle_pos: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        info = {}
        target_pos = current_handle_pos if current_handle_pos is not None else self.handle_position

        # 1. Contact-gated hand proximity to the handle
        r_hand = self._compute_hand_proximity_reward(left_hand_pos, right_hand_pos, target_pos)
        info['r_hand_proximity'] = r_hand

        # 2. Approach velocity reward (move hand toward handle)
        r_approach = self._compute_approach_reward(left_hand_pos, right_hand_pos, target_pos)
        info['r_approach'] = r_approach

        # 3. Lever rotation reward — depth-monotonic toward target angle, no flat plateau.
        self.max_lever_angle = max(self.max_lever_angle, lever_angle)
        angle_frac = float(np.clip(lever_angle / self.target_angle, 0.0, 1.0))
        r_rotate = self.rotate_reward * angle_frac
        # success band: within tol of target
        if abs(lever_angle - self.target_angle) < self.angle_tol and not self.lever_turned:
            r_rotate += self.first_turn_bonus
            self.lever_turned = True
            print(f"    [REWARD] Lever turned! Angle: {lever_angle:.4f} rad (target {self.target_angle:.2f})")
        # HOLD income: steady per-step income while the hinge is held near/past target, scaled
        # by how far it's turned — pushes RL toward a firm SUSTAINED turn. Always on (was
        # contact_mode-only; the button curriculum retracted after pressing for the same reason).
        if lever_angle > (self.target_angle - self.angle_tol):
            r_rotate += self.hold_bonus * angle_frac
        info['r_rotate'] = r_rotate
        info['lever_angle'] = lever_angle
        info['lever_turned'] = self.lever_turned

        # 4. Action penalty
        r_action = -self.action_penalty * np.linalg.norm(action) ** 2
        info['r_action'] = r_action

        # 5. Distance penalty — active (closest) hand far from handle. In contact mode the
        # episode starts on the grip, so a distance penalty would just punish the fixed
        # gripper-pad-vs-grip offset every step — disable it.
        if self.contact_mode:
            r_distance_penalty = 0.0
        else:
            dists = [np.linalg.norm(h - target_pos) for h in (left_hand_pos, right_hand_pos) if h is not None]
            dist_to_handle = min(dists) if dists else 1.0
            r_distance_penalty = -dist_to_handle * 4.0   # stronger pull to contact (anti-park)
        info['r_distance_penalty'] = r_distance_penalty

        # 6. Balance term — penalize base drifting off spawn xy
        if self.init_base_xy is None:
            self.init_base_xy = np.array(position[:2], dtype=float)
        base_drift = float(np.linalg.norm(np.array(position[:2]) - self.init_base_xy))
        r_balance = -self.balance_weight * base_drift
        info['r_balance'] = r_balance
        info['base_drift'] = base_drift

        total_reward = (
            self.alive_bonus
            + self.hand_proximity_weight * r_hand
            + r_approach
            + r_rotate
            + r_action
            + r_distance_penalty
            + r_balance
        )
        return total_reward, info

    def _compute_hand_proximity_reward(self, left_hand_pos, right_hand_pos, target_pos):
        if left_hand_pos is None and right_hand_pos is None:
            return 0.0
        distances = []
        if left_hand_pos is not None:
            distances.append(np.linalg.norm(left_hand_pos - target_pos))
        if right_hand_pos is not None:
            distances.append(np.linalg.norm(right_hand_pos - target_pos))
        min_dist = min(distances)
        band = 0.05   # NARROW (anti-parking, same lesson as the button curriculum): no far
        #               proximity plateau to camp on; the far pull is approach + distance-penalty.
        if min_dist > band:
            return 0.0
        proximity_reward = (band - min_dist) / band
        if min_dist < 0.05:
            proximity_reward += np.exp(-20.0 * min_dist)
        return proximity_reward

    def _compute_approach_reward(self, left_hand_pos, right_hand_pos, target_pos):
        if left_hand_pos is None and right_hand_pos is None:
            return 0.0
        distances = []
        if left_hand_pos is not None:
            distances.append(np.linalg.norm(left_hand_pos - target_pos))
        if right_hand_pos is not None:
            distances.append(np.linalg.norm(right_hand_pos - target_pos))
        current_dist = min(distances)
        if self.prev_hand_distance is None:
            self.prev_hand_distance = current_dist
            return 0.0
        approach_reward = (self.prev_hand_distance - current_dist) * 50.0
        self.prev_hand_distance = current_dist
        return approach_reward

    def reset(self):
        self.lever_turned = False
        self.max_lever_angle = 0.0
        self.prev_hand_distance = None
        self.init_base_xy = None

    def __repr__(self) -> str:
        return (f"LeverPressRewardFunction(handle_pos={self.handle_position}, "
                f"target_angle={self.target_angle}, angle_tol={self.angle_tol})")


# Lever handle position (grip world pos at rest angle), for easy reference
LEVER_POSITION = np.array([0.6, -1.72, 0.87])


# Button positions for easy reference
BUTTON_POSITIONS = {
    "button_red": np.array([-0.45, -1.85, 0.8]),
    "button_green": np.array([-0.15, -1.85, 0.9]),
    "button_yellow": np.array([0.15, -1.85, 0.9]),
    "button_blue": np.array([0.45, -1.85, 0.9]),
}
