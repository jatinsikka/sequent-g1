"""
Reward function for G1 humanoid RL training.

This module provides a modular reward function that emphasizes:
  - Position tracking (distance to target position)
  - Action efficiency (penalty for high actions)
"""

import numpy as np
from typing import Tuple


class RewardFunction:
    """
    Computes rewards for G1 humanoid training.
    """
    
    def __init__(
        self,
        target_position: Tuple[float, float] = (5.0, 0.0),
        desired_height: float = 0.75,
        position_weight: float = 10.0,
        action_penalty: float = 0.01,
    ):
        """
        Initialize the reward function. TODO: Need to make goals variable later
        
        Args:
            target_position: Target (x, y) position for the robot torso.
            desired_height: Desired torso height (z-coordinate).
            position_weight: Coefficient for position tracking reward.
            action_penalty: Coefficient for action penalty.
        """
        self.target_position = np.array(target_position)
        self.desired_height = desired_height
        self.position_weight = position_weight
        self.action_penalty = action_penalty
    
    def compute_reward(
        self,
        position: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """
        Compute total reward for the current step.
        
        Args:
            position: Robot torso position [x, y, z].
            rpy: Robot torso orientation in roll-pitch-yaw [rad].
            ang_vel: Robot angular velocity [rad/s].
            action: Action taken (motor commands).
        
        Returns:
            float: Total reward for this step.
        """
        r_position = self._compute_position_reward(position)
        r_action = self._compute_action_penalty(action)
        
        total_reward = (
            self.position_weight * r_position
            - self.action_penalty * r_action
        )
        
        return total_reward
    
    
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
            f"action_penalty={self.action_penalty})"
        )
