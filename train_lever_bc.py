"""BC warm-start for the LEVER skill (lever twin of train_button_bc.py's bc_pretrain).

Loads scripted pull-down demos (collect_lever_demos.py -> lever_demos.npz), builds the
SAME SB3 PPO model as train_lever.py (net_arch [128,128], log_std_init -1.5), behavior-
clones the policy MLP on the (obs, action) pairs (MSE on the deterministic action), and
saves checkpoints_lever_bc/bc_lever.zip in SB3 format so
    python train_lever.py --curriculum --resume checkpoints_lever_bc/bc_lever.zip
fine-tunes it with PPO. Ends with a deterministic verify rollout (2 eps, far starts).

Run: python train_lever_bc.py
"""
import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from lever_press_env import LeverPressEnv


def bc_pretrain(model, demo_obs, demo_act, epochs=50, lr=1e-3, batch=256):
    """Behavior-clone: MSE between the policy's deterministic (mean) action and the
    demo action. Value head / log_std get no gradient from this loss — PPO fine-tune
    starts them fresh, which is what --resume expects."""
    obs_t, _ = model.policy.obs_to_tensor(demo_obs)
    act_t = torch.as_tensor(demo_act, device=model.device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(demo_obs)
    for e in range(epochs):
        perm = torch.randperm(n, device=model.device)
        losses = []
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            dist = model.policy.get_distribution(obs_t[idx])
            mean = dist.distribution.mean
            loss = torch.nn.functional.mse_loss(mean, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if e % 5 == 0 or e == epochs - 1:
            print(f"[BC] epoch {e:03d}  mse={np.mean(losses):.5f}", flush=True)


def verify(model, env, n_eps=2):
    """Deterministic rollouts at far starts; report final lever angles
    (< 0.4 = clearly pulling, < 0.25 = success)."""
    finals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        a = env._get_lever_angle()
        finals.append(a)
        print(f"[VERIFY ep {ep}] frac={getattr(env, '_cur_frac', -1):.2f} "
              f"final_lever_angle={a:.3f}", flush=True)
    return finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="lever_demos.npz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckptdir", default="checkpoints_lever_bc")
    ap.add_argument("--verify_eps", type=int, default=2)
    ap.add_argument("--min_frac", type=float, default=0.0,
                    help="keep only demo episodes with curriculum frac >= this")
    args = ap.parse_args()

    d = np.load(args.demos)
    demo_obs = d["obs"].astype(np.float32)
    demo_act = d["act"].astype(np.float32)
    if args.min_frac > 0.0 and "ep_len" in d:
        keep = np.zeros(len(demo_obs), dtype=bool)
        i = 0
        for n, f in zip(d["ep_len"], d["ep_frac"]):
            keep[i:i + n] = f >= args.min_frac
            i += n
        demo_obs, demo_act = demo_obs[keep], demo_act[keep]
        print(f"[BC] frac filter >= {args.min_frac}: kept {len(demo_obs)} transitions", flush=True)
    print(f"[BC] loaded {len(demo_obs)} demo transitions from {args.demos}", flush=True)

    # Env matches train_lever.py --curriculum (reset_in_contact off, curriculum reset).
    # Verify at far starts: frac in [0.95, 1.0].
    env = LeverPressEnv(unified=True, reset_in_contact=False, curriculum=True,
                        headless=True, max_episode_steps=200)
    env.set_curriculum_frac(1.0)
    env.curriculum_frac_min = 0.95

    # Same PPO model as train_lever.py (fresh-model branch) so --resume just works.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.5},
        verbose=0, device=device,
    )

    bc_pretrain(model, demo_obs, demo_act, epochs=args.epochs, lr=args.lr, batch=args.batch)

    os.makedirs(args.ckptdir, exist_ok=True)
    out = os.path.join(args.ckptdir, "bc_lever")
    model.save(out)
    print(f"[BC] saved {out}.zip", flush=True)

    if args.verify_eps > 0:
        verify(model, env, n_eps=args.verify_eps)
    env.close()


if __name__ == "__main__":
    main()
