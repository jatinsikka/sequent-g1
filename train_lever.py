"""
Training script for the LEVER task — the full-humanoid twin of train_button.py.

Trains a policy for the G1 (legs balanced by AMO) to reach the lever handle on the
control panel and rotate its hinge to a target angle. Mirrors train_button.py, swapping
ButtonPressEnv -> LeverPressEnv.

NOTE: this replaces the old fixed-base train_lever.py (which used the wrong table-based
LeverEnv). The full-humanoid rig is the press_button twin requested for this skill.

Usage:
    python train_lever.py --smoke                 # quick sanity run (2000 steps, no W&B)
    python train_lever.py --timesteps 200000
    python train_lever.py --n_envs 32             # Azure F32 CPU box
"""

import argparse
import os
import sys
import numpy as np
import torch
try:
    import wandb
except Exception:
    wandb = None
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from lever_press_env import LeverPressEnv
from reward_fn import LeverPressRewardFunction, LEVER_POSITION
from train_button import CurriculumCallback   # success-gated reach curriculum (shared with button)


class VideoRecorderCallback(BaseCallback):
    """Record eval videos and log them to W&B."""

    def __init__(self, target_angle: float, eval_freq: int = 2500,
                 video_length: int = 200, use_wandb: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.target_angle = target_angle
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

    def _record_episode(self, deterministic: bool):
        frames = []
        obs, _ = self._eval_env.reset()
        episode_return = 0.0
        turned = False
        max_angle = 0.0

        for step in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self._eval_env.step(action)
            episode_return += reward
            if info.get('lever_turned', False):
                turned = True
            max_angle = max(max_angle, info.get('lever_angle', 0.0))
            try:
                frames.append(self._eval_env.render_frame(width=480, height=360))
            except Exception as e:
                print(f"[VideoRecorder] Frame capture error: {e}")
                break
            if terminated or truncated:
                break
        return frames, episode_return, turned, max_angle

    def _record_video(self):
        print(f"[VideoRecorder] Recording video at step {self.num_timesteps}...")
        try:
            if self._eval_env is None:
                eval_reward_fn = LeverPressRewardFunction(
                    handle_position=LEVER_POSITION, target_angle=self.target_angle,
                )
                self._eval_env = LeverPressEnv(
                    reward_fn=eval_reward_fn,
                    target_angle=self.target_angle,
                    max_episode_steps=self.video_length,
                    headless=True,
                )
                print(f"[VideoRecorder] Created eval environment")

            det_frames, det_return, det_turned, det_max = self._record_episode(deterministic=True)
            print(f"[VideoRecorder] Deterministic: {len(det_frames)} frames, return={det_return:.1f}, "
                  f"max_angle={det_max:.3f}, {'TURNED' if det_turned else 'not turned'}")

            sto_frames, sto_return, sto_turned, sto_max = self._record_episode(deterministic=False)
            print(f"[VideoRecorder] Stochastic: {len(sto_frames)} frames, return={sto_return:.1f}, "
                  f"max_angle={sto_max:.3f}, {'TURNED' if sto_turned else 'not turned'}")

            log_dict = {}
            if len(det_frames) > 10:
                va = np.transpose(np.array(det_frames), (0, 3, 1, 2))
                log_dict["video/deterministic"] = wandb.Video(
                    va, fps=25, format="mp4",
                    caption=f"Det - Step {self.num_timesteps}, R={det_return:.1f}, angle={det_max:.2f}, turned={det_turned}")
                log_dict["video/det_return"] = det_return
                log_dict["video/det_turned"] = int(det_turned)
                log_dict["video/det_max_angle"] = det_max
            if len(sto_frames) > 10:
                va = np.transpose(np.array(sto_frames), (0, 3, 1, 2))
                log_dict["video/stochastic"] = wandb.Video(
                    va, fps=25, format="mp4",
                    caption=f"Sto - Step {self.num_timesteps}, R={sto_return:.1f}, angle={sto_max:.2f}, turned={sto_turned}")
                log_dict["video/sto_return"] = sto_return
                log_dict["video/sto_turned"] = int(sto_turned)
                log_dict["video/sto_max_angle"] = sto_max
            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)
                print(f"[VideoRecorder] Videos logged to W&B!")
        except Exception as e:
            print(f"[VideoRecorder] Error recording video: {e}")
            import traceback
            traceback.print_exc()

    def _on_training_end(self):
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class LeverCallback(BaseCallback):
    """Log lever-turn training progress."""

    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_turn_rate = 0.0
        self.episode_turn_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        if infos and infos[0].get('lever_turned', False):
            self.episode_turn_count += 1
        if self.n_calls % self.eval_freq == 0 and self.episode_count > 0:
            rate = self.episode_turn_count / max(1, self.episode_count)
            print(f"[Step {self.n_calls}] Turn rate: {rate:.2%} ({self.episode_turn_count}/{self.episode_count})")
            if rate > self.best_turn_rate:
                self.best_turn_rate = rate
                print(f"    New best turn rate!")
        return True

    def _on_rollout_end(self) -> None:
        self.episode_count += 1


def train_lever(
    target_angle: float = 0.9,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints_lever",
    n_envs: int = 1,
    curriculum: bool = False,
):
    print(f"\n=== Lever Training ===")
    print(f"Lever handle at ({LEVER_POSITION[0]:.2f}, {LEVER_POSITION[1]:.2f}, {LEVER_POSITION[2]:.2f})")
    print(f"Target angle: {target_angle:.2f} rad")
    print(f"Total timesteps: {total_timesteps}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)

    run = None
    if use_wandb and wandb is not None:
        run = wandb.init(
            project="g1-lever",
            name=f"lever_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "lever_position": LEVER_POSITION.tolist(),
                "target_angle": target_angle,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
            },
            tags=["lever"],
        )
        print(f"[W&B] Run started: {run.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_env(rank: int):
        def _init():
            rfn = LeverPressRewardFunction(
                handle_position=LEVER_POSITION, target_angle=target_angle,
            )
            return LeverPressEnv(
                reward_fn=rfn,
                target_angle=target_angle,
                max_episode_steps=200,
                headless=True,
                device=device,
                reset_in_contact=not curriculum,   # curriculum uses its own seat-at-distance reset
                curriculum=curriculum,
            )
        return _init

    use_subproc = n_envs > 1 and sys.platform != "win32" and device == "cpu"
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    env = vec_cls([make_env(i) for i in range(n_envs)])
    if use_subproc:
        # workers are capped at 1 thread via OMP_NUM_THREADS (right — they'd thrash); but that
        # also caps the LEARNER's gradient update to 1 core. Raise it for the main process only
        # (workers already forked with their own 1-thread limit).
        torch.set_num_threads(16)
    print(f"[ENV] Created {n_envs} env(s) via {vec_cls.__name__} on {device}")
    print(f"[ENV] Observation space: {env.observation_space.shape}")
    print(f"[ENV] Action space: {env.action_space.shape}")

    batch_size = 1024 if n_envs > 1 else 64

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // max(n_envs, 1), 1),
        save_path=checkpoint_dir,
        name_prefix="ppo_lever",
    )
    progress_callback = LeverCallback(eval_freq=2000)
    curriculum_callback = CurriculumCallback() if curriculum else None
    video_callback = VideoRecorderCallback(
        target_angle=target_angle, eval_freq=2500, video_length=200, use_wandb=use_wandb,
    )

    print(f"\n>>> Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[c for c in (checkpoint_callback, progress_callback, video_callback, curriculum_callback) if c is not None],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n>>> Training interrupted!")

    final_path = os.path.join(checkpoint_dir, "final_lever")
    model.save(final_path)
    print(f"\n>>> Saved final model to: {final_path}.zip")

    env.close()
    if run is not None:
        run.finish()

    print(f"\n>>> Training complete!")
    print(f">>> Best turn rate: {progress_callback.best_turn_rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train lever rotation policy")
    parser.add_argument("--target_angle", type=float, default=0.9,
                        help="Target hinge angle in rad")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity run: 2000 steps, no W&B")
    parser.add_argument("--curriculum", action="store_true",
                        help="Reach curriculum: RL learns to reach from progressively farther (not IK-seeded)")
    args = parser.parse_args()

    if args.smoke:
        train_lever(target_angle=args.target_angle, total_timesteps=2000,
                    use_wandb=False, n_envs=1)
    else:
        train_lever(
            target_angle=args.target_angle,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            use_wandb=not args.no_wandb,
            n_envs=args.n_envs,
            curriculum=args.curriculum,
        )


if __name__ == "__main__":
    main()
