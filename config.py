"""
Configuration module for G1 humanoid RL training.

This module centralizes all hyperparameters and settings for training,
evaluation, and inference. Modify these values to adjust algorithm behavior.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Hyperparameters for GRASPING training."""

    # Environment
    robot_type: str = "g1"
    model_path: str = "g1.xml"
    device: str = field(default_factory=_default_device)  # auto: cuda if available, else cpu
    headless: bool = True  # Run without GUI (faster training)
    
    # Action space: 8 arm joint position offsets
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    action_dim: int = 8
    action_scale: float = 0.25
    
    # Training hyperparameters for grasping
    total_timesteps: int = int(1e5)  # 100k steps (grasping is simpler)
    learning_rate: float = 3e-4
    n_envs: int = 1  # Single environment
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.05  # INCREASED entropy for more exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Neural network architecture - smaller for simpler task
    net_arch: list = None  # Will default to [128, 128] in __post_init__
    
    # Reward weights (not used directly now - hardcoded in env_wrapper for grasping)
    reward_position_weight: float = 10.0
    reward_action_penalty: float = 0.01
    
    # Target: Screwdriver on right table (moved closer to edge)
    target_position: tuple = (1.1, 1.35)  # (x, y) - screwdriver location
    desired_height: float = 0.75  # Desired torso height
    
    # Episode settings. 250 steps (5s): measured grasps latch at step ~85-123, so
    # 150-step episodes left <1.3s to perform the lift -- not enough runway.
    max_episode_steps: int = 250  # ~5 seconds per episode
    sim_dt: float = 0.002
    sim_decimation: int = 10
    control_dt: float = sim_dt * sim_decimation
    
    # Termination conditions
    min_height: float = 0.4  # Terminate if robot falls
    max_roll: float = 0.8
    max_pitch: float = 0.8
    goal_distance: float = 0.08  # Success if hand within 8cm of screwdriver (easier)
    max_episode_time: float = 5.0  # 5 seconds max per episode (lift runway)

    # Optionally freeze one arm to simplify the task ('none', 'left', 'right')
    freeze_arm: str = "left"  # Freeze left arm so RL only controls right arm
    
    # Checkpointing
    save_interval: int = int(1e4)
    checkpoint_dir: str = "checkpoints"
    
    # W&B logging
    use_wandb: bool = True
    wandb_project: str = "g1-grasping"
    wandb_entity: Optional[str] = None
    wandb_tags: list = None
    wandb_notes: str = ""
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wandb_tags is None:
            self.wandb_tags = ["g1", "grasping", "ppo"]
        
        # Smaller network for simpler grasping task
        if self.net_arch is None:
            self.net_arch = [128, 128]
        
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
    # Optionally freeze one arm during evaluation
    freeze_arm: str = "none"


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
