"""
Evaluation script for trained G1 humanoid RL agents.

This script loads a trained model and evaluates it on the environment with
optional rendering and Weights & Biases logging.

Usage:
    python eval.py --model_path checkpoints/final_model --num_episodes 5 --render True
"""

import argparse
import os
import numpy as np
import torch
import wandb
from typing import Optional

from stable_baselines3 import PPO
from env_wrapper import G1RLEnv
from reward_fn import RewardFunction
from config import get_eval_config, get_training_config


class EvaluationManager:
    """
    Manager class for evaluating trained G1 humanoid RL agents.
    
    Handles model loading, rollout execution, and metric logging.
    """
    
    def __init__(
        self,
        model_path: str,
        num_episodes: int = 5,
        deterministic: bool = True,
        use_wandb: bool = True,
        render: bool = False,
    ):
        """
        Initialize the evaluation manager.
        
        Args:
            model_path: Path to the trained model (.zip file).
            num_episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic policy.
            use_wandb: Enable Weights & Biases logging.
            render: Enable environment rendering.
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.use_wandb = use_wandb
        self.render = render
        
        self.model = None
        self.env = None
        self.run = None
    
    def setup_wandb(self, run_name: str = "eval_run"):
        """
        Initialize Weights & Biases for evaluation logging.
        
        Args:
            run_name: Name for the W&B run.
        """
        if not self.use_wandb:
            return
        
        config = get_training_config()
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"{run_name}_{np.random.randint(10000)}",
            job_type="evaluation",
            tags=["eval", "inference"],
        )
        
        print(f"[INFO] W&B evaluation run started: {self.run.name}")
    
    def load_model(self) -> PPO:
        """
        Load a trained PPO model from disk.
        
        Returns:
            PPO: Loaded model.
        """
        if not os.path.exists(self.model_path + ".zip"):
            raise FileNotFoundError(f"Model not found: {self.model_path}.zip")
        
        # Determine device from environment
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model = PPO.load(self.model_path, device=device)
        print(f"[INFO] Model loaded from: {self.model_path}")
        
        return model
    
    def create_eval_env(self, render: bool = False) -> G1RLEnv:
        """
        Create evaluation environment with default reward function.
        
        Args:
            render: If True, enable MuJoCo viewer GUI for visualization.
        
        Returns:
            G1RLEnv: Configured evaluation environment.
        """
        config = get_training_config()
        
        reward_fn = RewardFunction(
            target_position=config.target_position,
            desired_height=config.desired_height,
            position_weight=config.reward_position_weight,
            action_penalty=config.reward_action_penalty,
        )
        
        env = G1RLEnv(
            policy_jit_path="amo_jit.pt",
            robot_type=config.robot_type,
            device=config.device,
            action_scale=config.action_scale,
            max_episode_steps=config.max_episode_steps,
            reward_fn=reward_fn,
            headless=not render,  # Enable viewer only if render=True
            min_height=config.min_height,
            max_roll=config.max_roll,
            max_pitch=config.max_pitch,
        )
        
        mode_str = "with GUI" if render else "headless"
        print(f"[INFO] Evaluation environment created ({mode_str})")
        return env
    
    def evaluate(self) -> dict:
        """
        Run evaluation episodes and compute statistics.
        
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        print(f"\n[INFO] Starting evaluation for {self.num_episodes} episodes...")
        
        self.model = self.load_model()
        self.env = self.create_eval_env(render=self.render)
        
        episode_returns = []
        episode_lengths = []
        episode_torso_heights = []
        episode_distances = []
        
        for episode_idx in range(self.num_episodes):
            obs, _ = self.env.reset()
            episode_return = 0.0
            episode_length = 0
            min_torso_height = float('inf')
            max_distance = 0.0
            
            done = False
            while not done:
                # Get action from model
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                )
                
                # Step environment (rendering happens automatically if viewer is active)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_return += reward
                episode_length += 1
                min_torso_height = min(min_torso_height, info.get("torso_height", 0))
                
                # Compute distance to target (access underlying env's data)
                target = self.env.reward_fn.target_position
                current_pos = self.env.env.data.qpos[:2]  # Access wrapped env's data
                distance = np.linalg.norm(current_pos - target)
                max_distance = max(max_distance, distance)
                
                done = terminated or truncated
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_torso_heights.append(min_torso_height)
            episode_distances.append(max_distance)
            
            print(f"  Episode {episode_idx + 1}/{self.num_episodes}: "
                  f"Return={episode_return:.2f}, Length={episode_length}, "
                  f"Min Height={min_torso_height:.3f}, Max Dist={max_distance:.3f}")
        
        # Compute statistics
        metrics = {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "min_return": np.min(episode_returns),
            "max_return": np.max(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "mean_min_torso_height": np.mean(episode_torso_heights),
            "mean_max_distance": np.mean(episode_distances),
        }
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics)
            print(f"\n[INFO] Metrics logged to W&B")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({self.num_episodes} episodes)")
        print(f"{'='*60}")
        for key, value in metrics.items():
            print(f"  {key:.<40} {value:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        if self.use_wandb:
            wandb.finish()


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained G1 humanoid RL agent"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/final_model",
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--deterministic",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--render",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Render the environment during evaluation",
    )
    parser.add_argument(
        "--use_wandb",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable Weights & Biases logging",
    )
    
    args = parser.parse_args()
    
    # Create and run evaluation manager
    manager = EvaluationManager(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        use_wandb=args.use_wandb,
        render=args.render,
    )
    
    try:
        metrics = manager.evaluate()
    finally:
        manager.close()


if __name__ == "__main__":
    main()
