"""PPO training for the non-prehensile PUSH task (PushEnv). Mirrors train_grasp.py.

Run:  python train_push.py --tag push-v0 --timesteps 1500000 --n_envs 56
Smoke: python train_push.py --smoke
"""
from __future__ import annotations
import os, argparse, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from push_env import PushEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(rank: int, seed: int = 0):
    def _f():
        env = PushEnv(); env.reset(seed=seed + rank); return env
    return _f


class RateLogger(BaseCallback):
    def __init__(self): super().__init__(); self.reset_acc()
    def reset_acc(self): self.n = 0; self.succ = 0; self.dist_sum = 0.0; self.knock = 0
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.n += 1
            self.succ += int(info.get("success", False))
            self.knock += int(info.get("knocked", False))
            self.dist_sum += float(info.get("push_dist", 0.0))
        return True
    def _on_rollout_end(self) -> None:
        if self.n:
            self.logger.record("push/success_frac", self.succ / self.n)
            self.logger.record("push/knocked_frac", self.knock / self.n)
            self.logger.record("push/mean_dist_cm", 100 * self.dist_sum / self.n)
        self.reset_acc()


def evaluate(model, n_episodes=20):
    env = PushEnv(); succ = 0; final_d = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=20_000 + ep); s = False; done = False; d = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            s = s or info["success"]; d = info["push_dist"]; done = term or trunc
        succ += int(s); final_d.append(d)
    env.close()
    return {"success": succ / n_episodes, "mean_final_dist_cm": 100 * float(np.mean(final_d))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="push-v0")
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--n_envs", type=int, default=56)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--no_subproc", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.timesteps = 8000; a.n_envs = 4; a.tag = "push-smoke"

    run_dir = os.path.join(HERE, "runs", a.tag)
    os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)
    vec_cls = DummyVecEnv if a.no_subproc else SubprocVecEnv
    env = vec_cls([make_env(i) for i in range(a.n_envs)])
    env = VecMonitor(env, os.path.join(run_dir, "monitor"))
    logger = configure(run_dir, ["stdout", "csv"])

    if a.resume and os.path.exists(a.resume):
        print(f"[resume] {a.resume}"); model = PPO.load(a.resume, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=0, learning_rate=a.lr, n_steps=a.n_steps,
                    batch_size=a.batch_size, n_epochs=a.n_epochs, ent_coef=a.ent_coef, gamma=0.99,
                    gae_lambda=0.95, policy_kwargs=dict(net_arch=[256, 256]))
    model.set_logger(logger)
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1),
                              save_path=os.path.join(run_dir, "ckpts"), name_prefix=a.tag)
    print(f"[train] tag={a.tag} steps={a.timesteps} n_envs={a.n_envs}")
    model.learn(total_timesteps=a.timesteps, callback=[RateLogger(), ckpt], progress_bar=False)
    model.save(os.path.join(run_dir, f"{a.tag}_final"))
    print("[eval] deterministic, 20 episodes ...")
    print("[RESULT]", {k: round(v, 3) for k, v in evaluate(model, 20).items()})
    env.close()


if __name__ == "__main__":
    main()
