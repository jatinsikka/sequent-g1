"""bp-v5: resume bp-v4 (held 22/25, frac_held still climbing) and train more PPO so the hold
crosses the 25-step bar. Self-contained (no cross-import) for clean spawn behavior."""
import sys, argparse, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure
from env_wrapper_button import ButtonPressEnv
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS

ACTION_SCALE = 2.5
BPOS = BUTTON_POSITIONS["button_red"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PressMetric(BaseCallback):
    def __init__(self):
        super().__init__(); self.d = []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            v = info.get("button_displacement")
            if v is not None: self.d.append(v)
        return True
    def _on_rollout_end(self):
        if self.d:
            arr = np.array(self.d)
            self.logger.record("press/mean_disp_cm", float(arr.mean() * 100))
            self.logger.record("press/max_disp_cm", float(arr.max() * 100))
            self.logger.record("press/frac_held_2cm", float((arr > 0.02).mean()))
            self.d = []


def make_env(i):
    def _init():
        torch.set_num_threads(1)
        e = ButtonPressEnv(button_name="button_red",
                           reward_fn=ButtonPressRewardFunction(button_position=BPOS),
                           freeze_arm="left", max_episode_steps=200, headless=True, device=DEVICE)
        e.action_scale = ACTION_SCALE
        return e
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_button/final_bcv4_red.zip")
    ap.add_argument("--timesteps", type=int, default=4000000)
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--logdir", default="bp_v5_logs")
    ap.add_argument("--out", default="checkpoints_button/final_bcv5_red")
    a = ap.parse_args()
    use_subproc = a.n_envs > 1 and sys.platform != "win32" and DEVICE == "cpu"
    vec = (SubprocVecEnv if use_subproc else DummyVecEnv)(
        [make_env(i) for i in range(a.n_envs)], **({"start_method": "spawn"} if use_subproc else {}))
    print(f"[ENV] {a.n_envs} envs on {DEVICE}; resuming from {a.ckpt}", flush=True)
    model = PPO.load(a.ckpt, env=vec, device=DEVICE)
    model.set_logger(configure(a.logdir, ["stdout", "csv"]))
    cb = CheckpointCallback(save_freq=max(20000 // a.n_envs, 1), save_path="checkpoints_button",
                            name_prefix="ppo_bcv5_red")
    model.learn(total_timesteps=a.timesteps, callback=[cb, PressMetric()],
                progress_bar=False, reset_num_timesteps=True)
    model.save(a.out)
    print(">>> Training complete!", flush=True)


if __name__ == "__main__":
    main()
