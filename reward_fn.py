"""
Reward function for G1 humanoid RL training.

This module provides modular reward functions for:
  1. Grasping objects (RewardFunction) - distance to target, hand proximity
  2. Pressing buttons (ButtonPressRewardFunction) - button displacement, hand proximity
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
        press_threshold: float = 0.02,  # Button pressed if displaced > 2cm
        hand_proximity_weight: float = 10.0,  # Increased to encourage reaching
        press_reward: float = 100.0,  # Big reward for actually pressing
        action_penalty: float = 0.0001,  # Reduced to not discourage movement
        alive_bonus: float = 0.1,  # Reduced to make reaching more important
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
        self.hand_proximity_weight = hand_proximity_weight
        self.press_reward = press_reward
        self.action_penalty = action_penalty
        self.alive_bonus = alive_bonus
        
        # Track button state
        self.button_pressed = False
        self.max_button_displacement = 0.0
        self.prev_hand_distance = None
    
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
        
        # 3. Button press reward
        r_press = 0.0
        self.max_button_displacement = max(self.max_button_displacement, button_displacement)
        
        if button_displacement > self.press_threshold:
            if not self.button_pressed:
                # First time pressing - big bonus!
                r_press = self.press_reward
                self.button_pressed = True
                print(f"    [REWARD] Button pressed! Displacement: {button_displacement:.4f}m")
            else:
                # Continuing to hold - smaller reward
                r_press = 10.0
        else:
            # Incremental reward for any displacement
            r_press = button_displacement * 200.0  # Scale up small displacements
        
        info['r_press'] = r_press
        info['button_displacement'] = button_displacement
        info['button_pressed'] = self.button_pressed
        
        # 4. Action penalty (very small)
        r_action = -self.action_penalty * np.linalg.norm(action) ** 2
        info['r_action'] = r_action
        
        # 5. Distance penalty - penalize being far from button
        if right_hand_pos is not None:
            dist_to_button = np.linalg.norm(right_hand_pos - target_pos)
        elif left_hand_pos is not None:
            dist_to_button = np.linalg.norm(left_hand_pos - target_pos)
        else:
            dist_to_button = 1.0
        r_distance_penalty = -dist_to_button * 2.0  # Penalty proportional to distance
        info['r_distance_penalty'] = r_distance_penalty
        
        # Total reward
        total_reward = (
            self.alive_bonus
            + self.hand_proximity_weight * r_hand
            + r_approach
            + r_press
            + r_action
            + r_distance_penalty
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
        
        # Linear reward that increases as hand gets closer
        # At 1m away: reward = 0, at 0m: reward = 1
        # This gives clear gradient even when far away
        proximity_reward = max(0, 1.0 - min_dist)
        
        # Bonus exponential reward when very close (< 0.2m)
        if min_dist < 0.2:
            proximity_reward += np.exp(-10.0 * min_dist)  # Steep bonus near button
        
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
    
    def __repr__(self) -> str:
        return (
            f"ButtonPressRewardFunction(button_pos={self.button_position}, "
            f"press_threshold={self.press_threshold})"
        )


# Button positions for easy reference
BUTTON_POSITIONS = {
    "button_red": np.array([-0.45, -1.85, 1.0]),
    "button_green": np.array([-0.15, -1.85, 1.0]),
    "button_yellow": np.array([0.15, -1.85, 1.0]),
    "button_blue": np.array([0.45, -1.85, 1.0]),
}
