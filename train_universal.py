"""
Train a UNIVERSAL manipulation policy for G1 Humanoid.

This trains ONE model that can manipulate ANY object by:
1. Randomizing the target each episode
2. Including target position in observation

The trained model works for ALL objects without retraining!

Usage:
    # Train on all objects (grasping + buttons)
    python train_universal.py
    
    # Train only on table objects (grasping)
    python train_universal.py --targets screwdriver wrench block_cube
    
    # Train only on buttons
    python train_universal.py --targets button_red button_green button_yellow button_blue
    
    # More timesteps
    python train_universal.py --timesteps 500000
"""

import argparse
import os
import numpy as np
import torch
import wandb
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env_wrapper_universal import UniversalManipulationEnv, MANIPULATION_TARGETS


class UniversalTrainingCallback(BaseCallback):
    """Track success rate across different targets."""
    
    def __init__(self, eval_freq: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_counts = {}
        self.episode_counts = {}
        self.total_episodes = 0
        self.total_successes = 0
    
    def _on_step(self) -> bool:
        # Track successes
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'target' in info:
                target = info['target']
                if target not in self.episode_counts:
                    self.episode_counts[target] = 0
                    self.success_counts[target] = 0
        
        # Log periodically
        if self.n_calls % self.eval_freq == 0 and self.total_episodes > 0:
            overall_rate = self.total_successes / self.total_episodes
            print(f"\n[Step {self.n_calls}] Overall success rate: {overall_rate:.1%} ({self.total_successes}/{self.total_episodes})")
            
            # Per-target breakdown
            for target in sorted(self.episode_counts.keys()):
                count = self.episode_counts[target]
                successes = self.success_counts[target]
                if count > 0:
                    rate = successes / count
                    print(f"    {target}: {rate:.1%} ({successes}/{count})")
            print()
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Count episode completions
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        
        for info, done in zip(infos, dones):
            if done and 'target' in info:
                target = info['target']
                self.total_episodes += 1
                self.episode_counts[target] = self.episode_counts.get(target, 0) + 1
                
                if info.get('success', False):
                    self.total_successes += 1
                    self.success_counts[target] = self.success_counts.get(target, 0) + 1


def train_universal(
    targets: list = None,
    total_timesteps: int = 200000,
    learning_rate: float = 3e-4,
    freeze_arm: str = "left",
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints_universal",
):
    """
    Train universal manipulation policy.
    
    Args:
        targets: List of targets to train on (None = all)
        total_timesteps: Total training steps
        learning_rate: PPO learning rate
        freeze_arm: Which arm to freeze
        use_wandb: Enable W&B logging
        checkpoint_dir: Save directory
    """
    print("\n" + "="*60)
    print("UNIVERSAL MANIPULATION TRAINING")
    print("="*60)
    
    if targets is None:
        targets = list(MANIPULATION_TARGETS.keys())
    
    print(f"Targets: {targets}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Freeze arm: {freeze_arm}")
    print()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # W&B
    run = None
    if use_wandb:
        run = wandb.init(
            project="g1-universal-manipulation",
            name=f"universal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "targets": targets,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "freeze_arm": freeze_arm,
            },
            tags=["universal", "multi-target"],
        )
    
    # Create environment
    env = UniversalManipulationEnv(
        allowed_targets=targets,
        freeze_arm=freeze_arm,
        max_episode_steps=200,
        headless=True,
    )
    
    print(f"[ENV] Observation space: {env.observation_space.shape}")
    print(f"[ENV] Action space: {env.action_space.shape}")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},  # Larger network for multi-task
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=20000,
        save_path=checkpoint_dir,
        name_prefix="ppo_universal",
    )
    
    progress_cb = UniversalTrainingCallback(eval_freq=5000)
    
    # Train
    print("\n>>> Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, progress_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n>>> Training interrupted!")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_universal")
    model.save(final_path)
    print(f"\n>>> Saved final model: {final_path}.zip")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total episodes: {progress_cb.total_episodes}")
    print(f"Total successes: {progress_cb.total_successes}")
    if progress_cb.total_episodes > 0:
        print(f"Overall success rate: {progress_cb.total_successes/progress_cb.total_episodes:.1%}")
    print()
    
    env.close()
    if run:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train universal manipulation policy")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Specific targets to train on (default: all)")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--freeze_arm", type=str, default="left",
                        choices=["none", "left", "right"],
                        help="Which arm to freeze")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    args = parser.parse_args()
    
    train_universal(
        targets=args.targets,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        freeze_arm=args.freeze_arm,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
