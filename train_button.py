"""
Training script for button pressing task.

This trains a policy to press buttons on the control panel.
The robot should already be positioned near the button (use walk_and_grasp.py for navigation).

Usage:
    python train_button.py --button red
    python train_button.py --button green --timesteps 200000
"""

import argparse
import os
import numpy as np
import torch
import wandb
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env_wrapper_button import ButtonPressEnv
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording evaluation videos and logging them to W&B.
    """
    
    def __init__(
        self,
        button_name: str,
        eval_freq: int = 2500,
        video_length: int = 200,
        use_wandb: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.button_name = button_name
        self.eval_freq = eval_freq
        self.video_length = video_length
        self.use_wandb = use_wandb
        self.last_eval_step = 0
        self._eval_env = None
    
    def _on_step(self) -> bool:
        if not self.use_wandb:
            return True
        
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._record_video()
        
        return True
    
    def _record_episode(self, deterministic: bool) -> tuple:
        """Record a single episode and return frames, return, and button pressed."""
        frames = []
        obs, _ = self._eval_env.reset()
        episode_return = 0.0
        button_pressed = False
        min_hand_dist = float('inf')
        
        for step in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self._eval_env.step(action)
            episode_return += reward
            
            if info.get('button_pressed', False):
                button_pressed = True
            if 'min_hand_dist' in info:
                min_hand_dist = min(min_hand_dist, info['min_hand_dist'])
            
            # Capture frame
            try:
                frame = self._eval_env.render_frame(width=480, height=360)
                frames.append(frame)
            except Exception as e:
                print(f"[VideoRecorder] Frame capture error: {e}")
                break
            
            if terminated or truncated:
                break
        
        return frames, episode_return, button_pressed, min_hand_dist
    
    def _record_video(self):
        """Record evaluation episode and log to W&B."""
        print(f"[VideoRecorder] Recording video at step {self.num_timesteps}...")
        
        try:
            # Create eval environment (with rendering enabled)
            if self._eval_env is None:
                # Create a fresh reward function for eval (don't share state with training)
                eval_reward_fn = ButtonPressRewardFunction(
                    button_position=BUTTON_POSITIONS[self.button_name],
                    press_threshold=0.02,
                    hand_proximity_weight=5.0,
                    press_reward=50.0,
                )
                self._eval_env = ButtonPressEnv(
                    button_name=self.button_name,
                    reward_fn=eval_reward_fn,
                    freeze_arm="left",  # Will be auto-selected based on button position
                    max_episode_steps=self.video_length,
                    headless=True,  # Still headless, but can render frames
                )
                print(f"[VideoRecorder] Created eval environment")
            
            # Record deterministic episode
            det_frames, det_return, det_pressed, det_min_dist = self._record_episode(deterministic=True)
            pressed_str = "PRESSED!" if det_pressed else "not pressed"
            print(f"[VideoRecorder] Deterministic: {len(det_frames)} frames, return={det_return:.1f}, {pressed_str}")
            
            # Record stochastic episode
            sto_frames, sto_return, sto_pressed, sto_min_dist = self._record_episode(deterministic=False)
            pressed_str = "PRESSED!" if sto_pressed else "not pressed"
            print(f"[VideoRecorder] Stochastic: {len(sto_frames)} frames, return={sto_return:.1f}, {pressed_str}")
            
            # Log videos to W&B
            log_dict = {}
            
            if len(det_frames) > 10:
                video_array = np.array(det_frames)  # (T, H, W, C)
                video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                log_dict["video/deterministic"] = wandb.Video(
                    video_array, fps=25, format="mp4",
                    caption=f"Deterministic - Step {self.num_timesteps}, Return: {det_return:.1f}, Pressed: {det_pressed}"
                )
                log_dict["video/det_return"] = det_return
                log_dict["video/det_pressed"] = int(det_pressed)
                log_dict["video/det_min_dist"] = det_min_dist if det_min_dist < float('inf') else -1
            
            if len(sto_frames) > 10:
                video_array = np.array(sto_frames)  # (T, H, W, C)
                video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                log_dict["video/stochastic"] = wandb.Video(
                    video_array, fps=25, format="mp4",
                    caption=f"Stochastic - Step {self.num_timesteps}, Return: {sto_return:.1f}, Pressed: {sto_pressed}"
                )
                log_dict["video/sto_return"] = sto_return
                log_dict["video/sto_pressed"] = int(sto_pressed)
                log_dict["video/sto_min_dist"] = sto_min_dist if sto_min_dist < float('inf') else -1
            
            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)
                print(f"[VideoRecorder] Videos logged to W&B!")
            else:
                print(f"[VideoRecorder] Not enough frames, skipping videos")
        
        except Exception as e:
            print(f"[VideoRecorder] Error recording video: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_training_end(self):
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class ButtonPressCallback(BaseCallback):
    """Callback for logging button press training progress."""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_press_rate = 0.0
        self.episode_press_count = 0
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check if button was pressed this step
        if 'button_pressed' in self.locals.get('infos', [{}])[0]:
            if self.locals['infos'][0]['button_pressed']:
                self.episode_press_count += 1
        
        # Log every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            if self.episode_count > 0:
                press_rate = self.episode_press_count / max(1, self.episode_count)
                print(f"[Step {self.n_calls}] Press rate: {press_rate:.2%} ({self.episode_press_count}/{self.episode_count} episodes)")
                
                if press_rate > self.best_press_rate:
                    self.best_press_rate = press_rate
                    print(f"    New best press rate!")
        
        return True
    
    def _on_rollout_end(self) -> None:
        self.episode_count += 1


def train_button_press(
    button: str = "red",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    freeze_arm: str = "left",
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints_button",
):
    """
    Train a policy to press a button.
    
    Args:
        button: Which button to train on ("red", "green", "yellow", "blue")
        total_timesteps: Total training steps
        learning_rate: Learning rate for PPO
        freeze_arm: Which arm to freeze ("left", "right", "none")
        use_wandb: Whether to log to W&B
        checkpoint_dir: Directory to save checkpoints
    """
    button_key = f"button_{button}"
    if button_key not in BUTTON_POSITIONS:
        raise ValueError(f"Unknown button: {button}. Choose from: red, green, yellow, blue")
    
    button_pos = BUTTON_POSITIONS[button_key]
    
    print(f"\n=== Button Press Training ===")
    print(f"Button: {button} at position ({button_pos[0]:.2f}, {button_pos[1]:.2f}, {button_pos[2]:.2f})")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Freeze arm: {freeze_arm}")
    print()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize W&B
    run = None
    if use_wandb:
        run = wandb.init(
            project="g1-button-press",
            name=f"button_{button}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "button": button,
                "button_position": button_pos.tolist(),
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "freeze_arm": freeze_arm,
            },
            tags=["button-press", f"button-{button}"],
        )
        print(f"[W&B] Run started: {run.name}")
    
    # Create reward function
    reward_fn = ButtonPressRewardFunction(
        button_position=button_pos,
        press_threshold=0.02,  # 2cm displacement to count as pressed
        hand_proximity_weight=5.0,
        press_reward=50.0,
    )
    
    # Create environment
    env = ButtonPressEnv(
        button_name=button_key,
        reward_fn=reward_fn,
        freeze_arm=freeze_arm,
        max_episode_steps=200,  # ~4 seconds per episode
        headless=True,
    )
    
    print(f"[ENV] Created ButtonPressEnv")
    print(f"[ENV] Observation space: {env.observation_space.shape}")
    print(f"[ENV] Action space: {env.action_space.shape}")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_button_{button}",
    )
    
    progress_callback = ButtonPressCallback(eval_freq=2000)
    
    video_callback = VideoRecorderCallback(
        button_name=button_key,
        eval_freq=2500,  # Record video every 2500 steps
        video_length=200,  # ~4 seconds per video
        use_wandb=use_wandb,
    )
    
    # Train
    print(f"\n>>> Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback, video_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n>>> Training interrupted!")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, f"final_button_{button}")
    model.save(final_path)
    print(f"\n>>> Saved final model to: {final_path}.zip")
    
    # Cleanup
    env.close()
    if run is not None:
        run.finish()
    
    print(f"\n>>> Training complete!")
    print(f">>> Best press rate: {progress_callback.best_press_rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train button pressing policy")
    parser.add_argument("--button", type=str, default="red",
                        choices=["red", "green", "yellow", "blue"],
                        help="Which button to train on")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--freeze_arm", type=str, default="left",
                        choices=["none", "left", "right"],
                        help="Which arm to freeze")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    args = parser.parse_args()
    
    train_button_press(
        button=args.button,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        freeze_arm=args.freeze_arm,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
