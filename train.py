"""
Training script for G1 humanoid RL using Stable Baselines 3.

This script integrates:
  - Stable Baselines 3 PPO algorithm for training
  - Weights & Biases for experiment tracking and logging
  - Modular environment and reward function setup
  - Checkpoint saving and resuming capability

Usage:
    python train.py --total_timesteps 1000000 --use_wandb True
"""

import argparse
import os
import sys
import numpy as np
import torch
try:
    import wandb
except Exception:  # wandb pulls in pkg_resources, absent on setuptools>=81
    wandb = None
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium.wrappers import TimeLimit

from env_wrapper import G1RLEnv
from reward_fn import RewardFunction
from config import get_training_config


class WandbCallback(BaseCallback):
    """
    Enhanced callback for comprehensive W&B logging of training metrics.
    
    Logs:
    - Episode statistics (return, length, termination)
    - Per-step metrics (torso state, actions, rewards)
    - Aggregated statistics (distributions, moving averages)
    - Termination analysis (failure mode tracking)
    """
    
    def __init__(self, use_wandb: bool = True, verbose: int = 0, log_frequency: int = 1):
        """
        Initialize W&B callback.
        
        Args:
            use_wandb: Whether to actually log to W&B (False = no-op).
            verbose: Verbosity level (0 = silent).
            log_frequency: How often to log per-step metrics (1 = every step).
        """
        super().__init__(verbose)
        self.use_wandb = use_wandb
        self.log_frequency = log_frequency
        
        # Per-episode metrics accumulation
        self.current_episode_heights = []
        self.current_episode_rolls = []
        self.current_episode_pitches = []
        self.current_episode_actions = []
        
        # Episode tracking lists
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_terminations = []
        
        # Termination analysis
        self.termination_counts = {
            "height": 0,
            "roll": 0,
            "pitch": 0,
            "goal_achieved": 0,
            "time_exceeded": 0,
            "timeout": 0,
        }
        
        # Rolling statistics (last 100 episodes)
        self.recent_returns = []
        self.recent_lengths = []
        self.max_recent = 100
        # Clean HEADLINE metrics (rolling over last 50 episodes) — the "is it learning" view
        from collections import deque as _dq
        self._recent_grasps = _dq(maxlen=50)
        self._recent_success = _dq(maxlen=50)
        self._recent_bestdist = _dq(maxlen=50)
        self._recent_epret = _dq(maxlen=50)
        self._total_episodes = 0
    
    def _on_step(self) -> bool:
        """
        Called at every step.
        
        Returns:
            bool: Whether to continue training.
        """
        # Log per-step metrics at specified frequency
        if self.use_wandb and self.n_calls % self.log_frequency == 0:
            self._log_step_metrics()
        
        # Log episode statistics when episodes complete
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self._log_episode_complete(info)

        # HEADLINE metrics from episode ends (uses dones; works without a Monitor wrapper)
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        ended = False
        for i, d in enumerate(dones):
            if d and i < len(infos):
                inf = infos[i]
                self._recent_grasps.append(1.0 if inf.get("object_grasped") else 0.0)
                self._recent_success.append(1.0 if inf.get("success") else 0.0)
                self._recent_bestdist.append(float(inf.get("best_distance_ever", 1.0)))
                self._recent_epret.append(float(inf.get("episode_return", 0.0)))
                self._total_episodes += 1
                ended = True
        if self.use_wandb and ended and len(self._recent_grasps) > 0:
            wandb.log({
                "key/grasp_success_rate": float(np.mean(self._recent_grasps)),
                "key/lift_success_rate": float(np.mean(self._recent_success)),
                "key/mean_best_reach_dist": float(np.mean(self._recent_bestdist)),
                "key/episode_return": float(np.mean(self._recent_epret)),
                "key/total_episodes": self._total_episodes,
            }, step=self.num_timesteps)

        # Print summary every 2500 steps
        if self.num_timesteps % 2500 == 0 and self.num_timesteps > 0:
            self._print_training_summary()
        
        return True
    
    def _print_training_summary(self):
        """Print training summary every 2500 steps."""
        n_episodes = len(self.episode_returns)
        if n_episodes > 0:
            recent_returns = self.recent_returns[-10:] if self.recent_returns else self.episode_returns[-10:]
            avg_return = np.mean(recent_returns)
            best_return = max(self.episode_returns) if self.episode_returns else 0
            print(f"\n{'='*60}")
            print(f"[Step {self.num_timesteps}] Training Summary")
            print(f"  Episodes completed: {n_episodes}")
            print(f"  Avg return (last 10): {avg_return:.2f}")
            print(f"  Best return ever: {best_return:.2f}")
            print(f"{'='*60}\n")
    
    def _log_step_metrics(self):
        """Log per-step metrics to W&B."""
        infos = self.locals.get("infos", [])
        if len(infos) == 0:
            return
        
        # Get info from first environment (for single env training)
        info = infos[0]
        
        # Extract state information if available
        log_dict = {}
        
        if "torso_height" in info:
            log_dict["step/torso_height"] = info["torso_height"]
            self.current_episode_heights.append(info["torso_height"])
        
        if "roll" in info:
            log_dict["step/roll"] = info["roll"]
            log_dict["step/roll_abs"] = abs(info["roll"])
            self.current_episode_rolls.append(info["roll"])
        
        if "pitch" in info:
            log_dict["step/pitch"] = info["pitch"]
            log_dict["step/pitch_abs"] = abs(info["pitch"])
            self.current_episode_pitches.append(info["pitch"])
        
        # Grasping-specific metrics
        if "min_hand_dist" in info:
            log_dict["step/min_hand_dist"] = info["min_hand_dist"]
        
        if "best_distance_ever" in info:
            log_dict["step/best_distance_ever"] = info["best_distance_ever"]
        
        if "left_hand_dist" in info:
            log_dict["step/left_hand_dist"] = info["left_hand_dist"]
            
        if "right_hand_dist" in info:
            log_dict["step/right_hand_dist"] = info["right_hand_dist"]
        
        if "episode_time" in info:
            log_dict["step/episode_time"] = info["episode_time"]
        
        # Reward component logging (for debugging reward shaping)
        if "milestone_bonus" in info:
            log_dict["reward/milestone_bonus"] = info["milestone_bonus"]
        if "proximity_reward" in info:
            log_dict["reward/proximity"] = info["proximity_reward"]
        if "progress_reward" in info:
            log_dict["reward/progress"] = info["progress_reward"]
        if "hovering_penalty" in info:
            log_dict["reward/hovering_penalty"] = info["hovering_penalty"]
        if "time_close_without_grasp" in info:
            log_dict["step/time_close_without_grasp"] = info["time_close_without_grasp"]
        if "grasp_bonus" in info:
            log_dict["reward/grasp_bonus"] = info["grasp_bonus"]
        if "grasp_ready_bonus" in info:
            log_dict["reward/grasp_ready_bonus"] = info["grasp_ready_bonus"]
        if "object_grasped" in info:
            log_dict["step/object_grasped"] = float(info["object_grasped"])
        if "active_hand_vel" in info:
            log_dict["step/active_hand_vel"] = info["active_hand_vel"]
        
        # Log action information if available
        if "actions" in self.locals:
            actions = self.locals["actions"]
            if len(actions.shape) > 1:
                actions = actions[0]  # First env
            
            log_dict["step/action_mean"] = float(np.mean(np.abs(actions)))
            log_dict["step/action_max"] = float(np.max(np.abs(actions)))
            log_dict["step/action_std"] = float(np.std(actions))
            self.current_episode_actions.append(actions.copy())
        
        # Log reward information
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            log_dict["step/reward"] = float(rewards[0] if len(rewards) > 0 else 0)
        
        if log_dict:
            wandb.log(log_dict, step=self.num_timesteps)
    
    def _log_episode_complete(self, info):
        """Log comprehensive episode statistics when episode completes."""
        episode_info = info["episode"]
        
        # Get termination reason if available
        termination_reason = info.get("termination_reason", None)
        terminated = info.get("terminated", False)
        
        # Track episode statistics
        episode_return = episode_info["r"]
        episode_length = episode_info["l"]
        
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.recent_returns.append(episode_return)
        self.recent_lengths.append(episode_length)
        
        # Maintain rolling window
        if len(self.recent_returns) > self.max_recent:
            self.recent_returns.pop(0)
            self.recent_lengths.pop(0)
        
        # Analyze termination type
        if terminated and termination_reason:
            if "goal_achieved" in termination_reason:
                self.termination_counts["goal_achieved"] += 1
                self.episode_terminations.append("goal_achieved")
            elif "time_exceeded" in termination_reason:
                self.termination_counts["time_exceeded"] += 1
                self.episode_terminations.append("time_exceeded")
            elif "height" in termination_reason:
                self.termination_counts["height"] += 1
                self.episode_terminations.append("height")
            elif "roll" in termination_reason:
                self.termination_counts["roll"] += 1
                self.episode_terminations.append("roll")
            elif "pitch" in termination_reason:
                self.termination_counts["pitch"] += 1
                self.episode_terminations.append("pitch")
        else:
            self.termination_counts["timeout"] += 1
            self.episode_terminations.append("timeout")
        
        # Only log to W&B if enabled
        if self.use_wandb:
            log_dict = {
                # Basic episode metrics
                "episode/return": episode_return,
                "episode/length": episode_length,
                "episode/time": episode_info["t"],
                
                # Rolling statistics
                "episode/return_mean_100": np.mean(self.recent_returns),
                "episode/return_std_100": np.std(self.recent_returns),
                "episode/length_mean_100": np.mean(self.recent_lengths),
                
                # Episode-level aggregations
                "episode/height_mean": np.mean(self.current_episode_heights) if self.current_episode_heights else 0,
                "episode/height_min": np.min(self.current_episode_heights) if self.current_episode_heights else 0,
                "episode/height_std": np.std(self.current_episode_heights) if self.current_episode_heights else 0,
                
                "episode/roll_mean_abs": np.mean(np.abs(self.current_episode_rolls)) if self.current_episode_rolls else 0,
                "episode/roll_max_abs": np.max(np.abs(self.current_episode_rolls)) if self.current_episode_rolls else 0,
                
                "episode/pitch_mean_abs": np.mean(np.abs(self.current_episode_pitches)) if self.current_episode_pitches else 0,
                "episode/pitch_max_abs": np.max(np.abs(self.current_episode_pitches)) if self.current_episode_pitches else 0,
                
                "episode/action_mean": np.mean(np.abs(self.current_episode_actions)) if self.current_episode_actions else 0,
                "episode/action_max": np.max(np.abs(self.current_episode_actions)) if self.current_episode_actions else 0,
                
                # Termination statistics
                "termination/total_height": self.termination_counts["height"],
                "termination/total_roll": self.termination_counts["roll"],
                "termination/total_pitch": self.termination_counts["pitch"],
                "termination/total_goal_achieved": self.termination_counts["goal_achieved"],
                "termination/total_time_exceeded": self.termination_counts["time_exceeded"],
                "termination/total_timeout": self.termination_counts["timeout"],
                
                # Termination ratios
                "termination/ratio_height": self.termination_counts["height"] / max(1, sum(self.termination_counts.values())),
                "termination/ratio_roll": self.termination_counts["roll"] / max(1, sum(self.termination_counts.values())),
                "termination/ratio_pitch": self.termination_counts["pitch"] / max(1, sum(self.termination_counts.values())),
                "termination/ratio_goal_achieved": self.termination_counts["goal_achieved"] / max(1, sum(self.termination_counts.values())),
                "termination/ratio_time_exceeded": self.termination_counts["time_exceeded"] / max(1, sum(self.termination_counts.values())),
                "termination/ratio_timeout": self.termination_counts["timeout"] / max(1, sum(self.termination_counts.values())),
            }
            
            if termination_reason:
                log_dict["episode/termination"] = termination_reason
            
            # Add histograms every N episodes
            if len(self.episode_returns) % 10 == 0:
                if self.current_episode_heights:
                    log_dict["histogram/heights"] = wandb.Histogram(self.current_episode_heights)
                if self.current_episode_rolls:
                    log_dict["histogram/rolls"] = wandb.Histogram(self.current_episode_rolls)
                if self.current_episode_pitches:
                    log_dict["histogram/pitches"] = wandb.Histogram(self.current_episode_pitches)
                if self.current_episode_actions:
                    actions_flat = np.array(self.current_episode_actions).flatten()
                    log_dict["histogram/actions"] = wandb.Histogram(actions_flat)
            
            wandb.log(log_dict, step=self.num_timesteps)
        
        # Reset per-episode accumulators
        self.current_episode_heights = []
        self.current_episode_rolls = []
        self.current_episode_pitches = []
        self.current_episode_actions = []
        
        # Always print episode info with running average
        n_episodes = len(self.episode_returns)
        avg_return = np.mean(self.recent_returns) if self.recent_returns else episode_return
        term_str = f", term={termination_reason}" if termination_reason else ""
        print(f"[Ep {n_episodes}] Return={episode_return:.2f}, Avg(100)={avg_return:.2f}, Len={episode_length}{term_str}")
        
        return True


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording evaluation videos and logging them to W&B.
    """
    
    def __init__(
        self,
        eval_freq: int = 10000,
        video_length: int = 10000,
        use_wandb: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
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
        """Record a single episode and return frames, return, and min distance."""
        frames = []
        obs, _ = self._eval_env.reset()
        episode_return = 0.0
        min_hand_dist = float('inf')
        
        for step in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self._eval_env.step(action)
            episode_return += reward
            
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
        
        return frames, episode_return, min_hand_dist
    
    def _record_video(self):
        """Record both deterministic and stochastic evaluation episodes and log to W&B."""
        print(f"[VideoRecorder] Recording videos at step {self.num_timesteps}...")
        
        try:
            # Create eval environment
            if self._eval_env is None:
                from config import get_training_config
                config = get_training_config()
                self._eval_env = G1RLEnv(
                    policy_jit_path="amo_jit.pt",
                    robot_type=config.robot_type,
                    device="cuda",  # Must use CUDA for AMO policy
                    max_episode_steps=config.max_episode_steps,
                    headless=True,
                    max_episode_time=config.max_episode_time,
                    freeze_arm=config.freeze_arm,
                )
                print(f"[VideoRecorder] Created eval environment")
            
            # Record deterministic episode
            det_frames, det_return, det_min_dist = self._record_episode(deterministic=True)
            print(f"[VideoRecorder] Deterministic: {len(det_frames)} frames, return={det_return:.1f}, min_dist={det_min_dist:.3f}")
            
            # Record stochastic episode
            sto_frames, sto_return, sto_min_dist = self._record_episode(deterministic=False)
            print(f"[VideoRecorder] Stochastic: {len(sto_frames)} frames, return={sto_return:.1f}, min_dist={sto_min_dist:.3f}")
            
            # Log videos to W&B
            log_dict = {}
            
            if len(det_frames) > 10:
                video_array = np.array(det_frames)  # (T, H, W, C)
                video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                log_dict["video/deterministic"] = wandb.Video(
                    video_array, fps=25, format="mp4",
                    caption=f"Deterministic - Step {self.num_timesteps}, Return: {det_return:.1f}, MinDist: {det_min_dist:.3f}"
                )
                log_dict["video/det_return"] = det_return
                log_dict["video/det_min_dist"] = det_min_dist if det_min_dist < float('inf') else -1
            
            if len(sto_frames) > 10:
                video_array = np.array(sto_frames)  # (T, H, W, C)
                video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                log_dict["video/stochastic"] = wandb.Video(
                    video_array, fps=25, format="mp4",
                    caption=f"Stochastic - Step {self.num_timesteps}, Return: {sto_return:.1f}, MinDist: {sto_min_dist:.3f}"
                )
                log_dict["video/sto_return"] = sto_return
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


class TrainingManager:
    """
    Manager class for G1 humanoid RL training.
    
    Handles environment setup, agent creation, training loop, and logging.
    """
    
    def __init__(self, config=None):
        """
        Initialize the training manager.
        
        Args:
            config: TrainingConfig instance; uses default if None.
        """
        self.config = config or get_training_config()
        self.env = None
        self.model = None
        self.run = None
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.config.use_wandb:
            return
        
        # Readable, sortable run name: "<label>-<MMDD-HHMM>", e.g. v5.2-reachgrasp-0609-2230.
        # Pass --run_name to label the iteration; otherwise it defaults to "reachgrasp".
        from datetime import datetime
        label = getattr(self.config, "run_name", None) or "reachgrasp"
        run_name = f"{label}-{datetime.now().strftime('%m%d-%H%M')}"

        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "n_epochs": self.config.n_epochs,
                "n_envs": self.config.n_envs,
                "total_timesteps": self.config.total_timesteps,
                "reward_position_weight": self.config.reward_position_weight,
                "reward_action_penalty": self.config.reward_action_penalty,
                "goal_distance": self.config.goal_distance,
                "max_episode_time": self.config.max_episode_time,
            },
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
        )
        
        print(f"[INFO] W&B run started: {self.run.name}")
    
    def create_environment(self):
        """
        Create and configure the training environment(s).
        
        Returns:
            VecEnv or single env: The wrapped humanoid environment(s).
        """
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        
        # Initialize reward function with config values
        # target_object_pos is the 3D position of the screwdriver
        target_object_pos = (self.config.target_position[0], self.config.target_position[1], 0.74)  # screwdriver height
        
        reward_fn = RewardFunction(
            target_position=self.config.target_position,
            target_object_pos=target_object_pos,
            desired_height=self.config.desired_height,
            position_weight=1.0,  # Bonus for being close
            velocity_weight=5.0,  # Strong reward for moving toward target
            hand_proximity_weight=2.0,  # Reward for reaching
            action_penalty=0.003,  # Low penalty to encourage larger actions
            alive_bonus=0.5,  # Constant reward for staying upright
        )
        
        def make_env(rank: int):
            """
            Create a single environment instance.
            
            Args:
                rank: Environment ID for seeding.
            """
            def _init():
                env = G1RLEnv(
                    policy_jit_path="amo_jit.pt",
                    robot_type=self.config.robot_type,
                    device=self.config.device,
                    action_scale=self.config.action_scale,
                    max_episode_steps=self.config.max_episode_steps,
                    reward_fn=reward_fn,
                    headless=self.config.headless,
                    min_height=self.config.min_height,
                    max_roll=self.config.max_roll,
                    max_pitch=self.config.max_pitch,
                    goal_distance=self.config.goal_distance,
                    max_episode_time=self.config.max_episode_time,
                    freeze_arm=self.config.freeze_arm,
                    verbose=0,
                    curriculum=getattr(self.config, "curriculum", False),
                )
                # Wrap with TimeLimit to enforce max episode steps
                env = TimeLimit(env, max_episode_steps=self.config.max_episode_steps)
                return env
            return _init
        
        # Create vectorized environment
        # SubprocVecEnv (true parallelism) is safe on Linux with CPU inference.
        # Keep DummyVecEnv on Windows or CUDA: subprocess workers can't share the CUDA JIT context.
        if self.config.n_envs > 1:
            use_subproc = sys.platform != "win32" and self.config.device == "cpu"
            vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
            kw = {"start_method": "spawn"} if use_subproc else {}
            env = vec_cls([make_env(i) for i in range(self.config.n_envs)], **kw)
            mode_str = "headless" if self.config.headless else "with GUI"
            par_str = "parallel" if use_subproc else "sequential"
            print(f"[INFO] Created {self.config.n_envs} {par_str} environments ({mode_str})")
        else:
            # Single environment
            env = DummyVecEnv([make_env(0)])
            mode_str = "headless" if self.config.headless else "with GUI"
            print(f"[INFO] Created single environment ({mode_str})")
        
        print(f"[INFO] Termination conditions:")
        print(f"  - Fall: height<{self.config.min_height}m, |roll|>{self.config.max_roll}rad, |pitch|>{self.config.max_pitch}rad")
        print(f"  - Success: distance to goal <{self.config.goal_distance}m")
        print(f"  - Time limit: >{self.config.max_episode_time}s")
        
        return env
    
    def create_agent(self) -> PPO:
        """
        Create a Stable Baselines 3 PPO agent.
        
        Returns:
            PPO: Configured PPO agent.
        """
        # Configure policy network architecture
        policy_kwargs = dict(
            net_arch=self.config.net_arch  # Shared architecture for policy and value networks
        )
        
        # n_steps should be at least as large as batch_size for proper updates
        # Use 2048 as standard for PPO (will collect multiple episodes)
        n_steps = 2048
        
        agent = PPO(
            policy="MlpPolicy",
            env=self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=self.config.learning_rate,
            n_steps=n_steps,  # Steps per rollout buffer
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=1,
            device=self.config.device,
        )
        
        print(f"[INFO] PPO agent created with learning rate {self.config.learning_rate}")
        print(f"[INFO] Rollout buffer: n_steps={n_steps}, batch_size={self.config.batch_size}")
        print(f"[INFO] Network architecture: {self.config.net_arch}")
        return agent
    
    def train(self, resume_from: Optional[str] = None):
        """
        Execute the training loop.
        
        Args:
            resume_from: Path to a saved model to resume training from.
        """
        print("[INFO] Starting training setup...")
        
        # Initialize W&B
        self.setup_wandb()
        
        # Create environment and agent
        self.env = self.create_environment()
        
        if resume_from and os.path.exists(resume_from):
            print(f"[INFO] Resuming from checkpoint: {resume_from}")
            self.model = PPO.load(resume_from, env=self.env)
        else:
            self.model = self.create_agent()
        
        # Set up callbacks for checkpointing
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_interval,
            save_path=self.config.checkpoint_dir,
            name_prefix="ppo_g1",
            save_replay_buffer=False,
        )
        
        # Enhanced W&B callback with comprehensive logging
        wandb_callback = WandbCallback(
            use_wandb=self.config.use_wandb,
            verbose=1,
            log_frequency=10
        )
        
        # Video recording callback
        # NOTE: each call spins up an eval env and encodes TWO mp4s via moviepy,
        # which blocks the trainer. At eval_freq=2500 that was the dominant cost
        # (~40fps, mostly stalled in encode). 25000 keeps a live dashboard cadence
        # (~one video per ~10min of training) without throttling throughput.
        video_callback = VideoRecorderCallback(
            eval_freq=25000,  # Record every 25k steps (was 2500 -- killed fps)
            video_length=200,
            use_wandb=self.config.use_wandb,
            verbose=1,
        )
        
        print(f"\n[INFO] Training for {self.config.total_timesteps} timesteps...")
        print(f"[INFO] Checkpoints saved to: {self.config.checkpoint_dir}")
        if self.config.use_wandb:
            print(f"[INFO] W&B logging enabled with video recording every 25000 steps")
        print()
        
        try:
            # Train the model
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=[checkpoint_callback, wandb_callback, video_callback],
                progress_bar=True,
            )
            
            # Save final model
            final_model_path = os.path.join(self.config.checkpoint_dir, "final_model")
            self.model.save(final_model_path)
            print(f"\n[INFO] Training complete. Final model saved to: {final_model_path}")
            
            # Log final model to W&B
            if self.config.use_wandb:
                wandb.save(f"{final_model_path}.zip")
                print(f"[INFO] Final model uploaded to W&B")
        
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user. Saving checkpoint...")
            interrupt_path = os.path.join(self.config.checkpoint_dir, "interrupted_model")
            self.model.save(interrupt_path)
            print(f"[INFO] Checkpoint saved to: {interrupt_path}")
        
        finally:
            # Cleanup
            self.env.close()
            if self.config.use_wandb:
                wandb.finish()


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train G1 humanoid RL agent with Stable Baselines 3"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=int(1e6),
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="128,128",
        help="Network architecture as comma-separated layer sizes (e.g., '256,256,128')",
    )
    parser.add_argument(
        "--use_wandb",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="g1-humanoid-rl",
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Label for the W&B run, e.g. 'v5.2-reachgrasp'. A timestamp is appended automatically.",
    )
    parser.add_argument(
        "--curriculum",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Adaptive spawn curriculum: robot starts closer when struggling, anneals to full task.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda or cpu",
    )
    parser.add_argument(
        "--headless",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run without GUI (True for faster training, False to show graphics)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments for training",
    )
    
    args = parser.parse_args()
    
    # Create config with command-line overrides
    config = get_training_config()
    config.total_timesteps = args.total_timesteps
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    
    # Parse network architecture from command line
    if args.net_arch:
        config.net_arch = [int(x.strip()) for x in args.net_arch.split(',')]
    
    config.n_envs = args.n_envs
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.run_name = args.run_name
    config.curriculum = args.curriculum
    config.device = args.device
    config.headless = args.headless
    if not config.headless:
        config.n_envs = 1  # Force single env if not headless
    
    # Create and run training manager
    manager = TrainingManager(config)
    manager.train(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
