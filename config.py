"""
Configuration module for G1 humanoid RL training.

This module centralizes all hyperparameters and settings for training,
evaluation, and inference. Modify these values to adjust algorithm behavior.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    
    # Environment
    robot_type: str = "g1"
    model_path: str = "g1.xml"
    device: str = "cuda"  # "cuda" or "cpu"
    headless: bool = True  # Run without GUI (faster training)
    
    # Action space from play_amo.py controls, not sure if we should use arm or not TODO: 
    # [vx, vy, yaw, height, torso_yaw, torso_pitch, torso_roll, arm_control_flag]
    action_dim: int = 8
    action_scale: float = 0.25
    # Training hyperparameters from chatGPT suggestions
    total_timesteps: int = int(1e6)
    learning_rate: float = 5e-4
    n_envs: int = 2  # Number of parallel environments
    batch_size: int = 64
    n_epochs: int = 4
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5
    
    # Neural network architecture
    # Policy and value network hidden layer sizes (list of integers)
    # Default SB3 is [64, 64]. Larger networks may learn better but train slower.
    net_arch: list = None  # Will default to [256, 256] in __post_init__
    
    # Reward weights
    reward_position_weight: float = 10.0
    reward_action_penalty: float = 0.01
    
    # Temp target goals for now TODO: 
    target_position: tuple = (5.0, 0.0)  # (x, y) in world coordinates
    desired_height: float = 0.75  # Desired torso height
    
    # Episode settings
    max_episode_steps: int = 2048
    sim_dt: float = 0.002
    sim_decimation: int = 10
    control_dt: float = sim_dt * sim_decimation
    
    # Termination conditions for falling, should add others later TODO: 
    min_height: float = 0.4  # Terminate if torso height < this
    max_roll: float = 0.8  # Terminate if |roll| > this
    max_pitch: float = 0.8  # Terminate if |pitch| > this
    goal_distance: float = 0.1  # Terminate successfully if within this distance of goal (meters)
    max_episode_time: float = 30.0  # Terminate if episode exceeds this time (seconds)
    
    # Checkpointing
    save_interval: int = int(1e4)  # Save model every N steps
    checkpoint_dir: str = "checkpoints"
    
    # W&B logging
    use_wandb: bool = True
    wandb_project: str = "g1-humanoid-rl"
    wandb_entity: Optional[str] = None
    wandb_tags: list = None
    wandb_notes: str = ""
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wandb_tags is None:
            self.wandb_tags = ["g1", "humanoid", "ppo"]
        
        # Set default network architecture if not specified
        if self.net_arch is None:
            self.net_arch = [256, 256]
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Hyperparameters for evaluation"""
    
    model_path: str = "checkpoints/best_model"
    num_episodes: int = 5
    deterministic: bool = True
    render: bool = True
    
    # W&B logging
    use_wandb: bool = False
    wandb_run_name: str = "eval_run"


def get_training_config() -> TrainingConfig:
    """
    Retrieve the training configuration.
    
    Returns:
        TrainingConfig: Training hyperparameter configuration.
    """
    return TrainingConfig()


def get_eval_config() -> EvaluationConfig:
    """
    Retrieve the evaluation configuration.
    
    Returns:
        EvaluationConfig: Evaluation hyperparameter configuration.
    """
    return EvaluationConfig()
