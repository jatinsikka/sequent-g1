"""
PPO training for the Robotiq-gripper grasp rebuild (GraspFixedEnv).

Self-contained (does NOT use the old config-driven train.py). The env has no AMO/
TorchScript dependency, so SubprocVecEnv true parallelism works even on Windows.

Reward is grip+lift gated on two-sided pad contact (see grasp_fixed_env.py) -- this is
a clean test of whether the rebuilt env learns a real grasp WITHOUT a distance-to-pose
reward (which the v5.x history showed gets hover-hacked).

wandb is broken in this env -> we log SB3's CSV (runs/<tag>/progress.csv) + a custom
grasp/lift/success aggregator. Judge the DETERMINISTIC policy at the end.

Run:  python train_grasp.py --tag gr-v0-0623 --timesteps 1000000 --n_envs 8
Smoke: python train_grasp.py --smoke
"""
from __future__ import annotations
import os, argparse, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from grasp_fixed_env import GraspFixedEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(rank: int, seed: int = 0, robust: bool = False):
    def _f():
        env = GraspFixedEnv(robust=robust)
        env.reset(seed=seed + rank)
        return env
    return _f


class RateLogger(BaseCallback):
    """Aggregate grasp/lift/success/knock rates from step infos and log per rollout."""
    def __init__(self):
        super().__init__()
        self.reset_acc()

    def reset_acc(self):
        self.n = 0; self.grasped = 0; self.success = 0; self.knocked = 0; self.lift_sum = 0.0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.n += 1
            self.grasped += int(info.get("grasped", False))
            self.success += int(info.get("success", False))
            self.knocked += int(info.get("knocked", False))
            self.lift_sum += float(info.get("lift", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self.n:
            self.logger.record("grasp/grasped_frac", self.grasped / self.n)
            self.logger.record("grasp/success_frac", self.success / self.n)
            self.logger.record("grasp/knocked_frac", self.knocked / self.n)
            self.logger.record("grasp/mean_lift_cm", 100 * self.lift_sum / self.n)
        self.reset_acc()


def evaluate(model, n_episodes=20, robust=False):
    """Deterministic eval -> grasp/lift/success rates (the number that counts)."""
    env = GraspFixedEnv(robust=robust)
    grasps = lifts = succ = 0; best_lifts = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        g = False; bl = 0.0; s = False
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            g = g or info["grasped"]; bl = max(bl, info["lift"]); s = s or info["success"]
            done = term or trunc
        grasps += int(g); lifts += int(bl > 0.03); succ += int(s); best_lifts.append(bl)
    env.close()
    return {"grasp": grasps / n_episodes, "lift>3cm": lifts / n_episodes,
            "success": succ / n_episodes, "mean_best_lift_cm": 100 * np.mean(best_lifts)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="gr-v0-0623")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=512)      # per env; buffer = n_steps*n_envs
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--no_subproc", action="store_true")
    p.add_argument("--robust", action="store_true", help="heavier 250g tool + realistic friction")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    if a.smoke:
        a.timesteps = 8000; a.n_envs = 4; a.tag = "gr-smoke"

    run_dir = os.path.join(HERE, "runs", a.tag)
    os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)

    vec_cls = DummyVecEnv if a.no_subproc else SubprocVecEnv
    env = vec_cls([make_env(i, robust=a.robust) for i in range(a.n_envs)])
    env = VecMonitor(env, os.path.join(run_dir, "monitor"))

    logger = configure(run_dir, ["stdout", "csv"])

    if a.resume and os.path.exists(a.resume):
        print(f"[resume] {a.resume}")
        model = PPO.load(a.resume, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                    learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size,
                    n_epochs=a.n_epochs, ent_coef=a.ent_coef, gamma=0.99, gae_lambda=0.95,
                    policy_kwargs=dict(net_arch=[256, 256]))
    model.set_logger(logger)

    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1),
                              save_path=os.path.join(run_dir, "ckpts"), name_prefix=a.tag)
    print(f"[train] tag={a.tag} steps={a.timesteps} n_envs={a.n_envs} "
          f"buffer={a.n_steps * a.n_envs} subproc={not a.no_subproc}")
    model.learn(total_timesteps=a.timesteps, callback=[RateLogger(), ckpt], progress_bar=False)

    final = os.path.join(run_dir, f"{a.tag}_final")
    model.save(final)
    print(f"[saved] {final}.zip")

    print("[eval] deterministic, 20 episodes ...")
    res = evaluate(model, 20, robust=a.robust)
    print("[RESULT]", {k: round(v, 3) for k, v in res.items()})
    env.close()


if __name__ == "__main__":
    main()
