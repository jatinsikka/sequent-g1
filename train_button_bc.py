"""bp-v3: BC warm-start + PPO finetune for button_press, with CSV logging.

The cold-start failure of bp-v0..v2 was that PPO's mean policy never *experienced* a
press, so it parked (hover) or jabbed. Here we first behavior-clone the policy on the
scripted deep-press demo so it STARTS pressing, then PPO-finetune with the smooth
sustained-hold reward (reward accrues every step the button is held deep). The policy
sees button_displacement in its obs, so it can learn the closed-loop hold a script can't.

CSV logs (reward + press-depth + frac-held curves) land in --logdir for the training graph.
Run: python train_button_bc.py --button red --n_envs 32 --timesteps 2000000 --logdir bp_v3_logs
"""
import argparse, os, sys, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure
from env_wrapper_button import ButtonPressEnv
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS

import mujoco
ARM_JOINTS = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
              'left_shoulder_yaw_joint', 'left_elbow_joint']
ACTION_SCALE = 2.5  # arm action scale used by both the IK demo and training (uncaps the reach)


def ik_press_action(env, m, mod, capg, hand, dofadr, default, jacp):
    """One closed-loop IK step: servo the hand toward a point just inside the cap, return
    the 8-d arm action that drives there. This is the holding-press controller we clone."""
    cap = m.geom_xpos[capg].copy(); target = cap.copy(); target[1] -= 0.06
    mujoco.mj_jacBody(mod, m, jacp, None, hand); J = jacp[:, dofadr]
    dq = np.linalg.lstsq(J, target - m.xpos[hand].copy(), rcond=None)[0]
    tgt = m.qpos[[68, 69, 70, 71]] + np.clip(dq, -0.8, 0.8)
    act = np.zeros(8, dtype=np.float32)
    act[:4] = np.clip((tgt - default[15:19]) / ACTION_SCALE, -1, 1)
    return act


def collect_demos(button_key, freeze_arm, n_eps=8, horizon=170):
    """Run the closed-loop IK press (a real reach+press+hold) and collect (obs, action) to BC.
    Small action noise on later episodes for state coverage."""
    obs_l, act_l = [], []
    capname = button_key.replace('button_', 'push_button_') + '_top'
    for ep in range(n_eps):
        env = ButtonPressEnv(button_name=button_key,
                             reward_fn=ButtonPressRewardFunction(button_position=BUTTON_POSITIONS[button_key]),
                             freeze_arm=freeze_arm, max_episode_steps=horizon, headless=True)
        env.action_scale = ACTION_SCALE
        o, _ = env.reset()
        m, mod = env.env.data, env.env.model
        capg = mujoco.mj_name2id(mod, mujoco.mjtObj.mjOBJ_GEOM, capname)
        hand = env.left_hand_id
        dofadr = [mod.jnt_dofadr[mujoco.mj_name2id(mod, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
        default = env.env.default_dof_pos.copy(); jacp = np.zeros((3, mod.nv))
        for t in range(horizon):
            act = ik_press_action(env, m, mod, capg, hand, dofadr, default, jacp)
            if ep > 0:
                act[:4] = np.clip(act[:4] + np.random.uniform(-0.06, 0.06, 4), -1, 1)
            obs_l.append(o.copy()); act_l.append(act.copy())
            o, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        env.close()
    return np.array(obs_l, dtype=np.float32), np.array(act_l, dtype=np.float32)


def bc_pretrain(model, demo_obs, demo_act, epochs=400, lr=1e-3, batch=512):
    """Behavior-clone: maximize log-prob of expert actions under the policy."""
    obs_t, _ = model.policy.obs_to_tensor(demo_obs)
    act_t = torch.as_tensor(demo_act, device=model.device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(demo_obs)
    for e in range(epochs):
        idx = torch.randperm(n, device=model.device)[:batch]
        dist = model.policy.get_distribution(obs_t[idx])
        loss = -dist.log_prob(act_t[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if e % 100 == 0:
            print(f"[BC] epoch {e}  nll={loss.item():.3f}", flush=True)
    print(f"[BC] done  final_nll={loss.item():.3f}", flush=True)


class PressMetric(BaseCallback):
    """Log per-rollout press metrics so the training graph has a real task-success curve."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--button", default="red")
    ap.add_argument("--timesteps", type=int, default=2000000)
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--logdir", default="bp_v3_logs")
    ap.add_argument("--ckptdir", default="checkpoints_button")
    args = ap.parse_args()

    button_key = f"button_{args.button}"
    bpos = BUTTON_POSITIONS[button_key]
    freeze = "left"  # env auto-flips to use the correct arm for the button side
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckptdir, exist_ok=True)

    def make_env(i):
        def _init():
            torch.set_num_threads(1)
            e = ButtonPressEnv(button_name=button_key,
                               reward_fn=ButtonPressRewardFunction(button_position=bpos),
                               freeze_arm=freeze, max_episode_steps=200, headless=True, device=device)
            e.action_scale = ACTION_SCALE  # match the IK demo (uncapped reach)
            return e
        return _init

    use_subproc = args.n_envs > 1 and sys.platform != "win32" and device == "cpu"
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    kw = {"start_method": "spawn"} if use_subproc else {}
    env = vec_cls([make_env(i) for i in range(args.n_envs)], **kw)
    print(f"[ENV] {args.n_envs} env(s) via {vec_cls.__name__} on {device}", flush=True)

    batch_size = 1024 if args.n_envs > 1 else 64
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=batch_size,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                vf_coef=0.5, max_grad_norm=0.5, policy_kwargs={"net_arch": [128, 128]},
                verbose=0, device=device)
    model.set_logger(configure(args.logdir, ["stdout", "csv"]))

    # --- BC warm-start ---
    demo_obs, demo_act = collect_demos(button_key, freeze)
    print(f"[BC] collected {len(demo_obs)} demo transitions; pretraining...", flush=True)
    bc_pretrain(model, demo_obs, demo_act)

    # --- PPO finetune ---
    ckpt = CheckpointCallback(save_freq=max(20000 // args.n_envs, 1), save_path=args.ckptdir,
                              name_prefix=f"ppo_bcv3_{args.button}")
    model.learn(total_timesteps=args.timesteps, callback=[ckpt, PressMetric()], progress_bar=False)
    model.save(os.path.join(args.ckptdir, f"final_bcv3_{args.button}"))
    print(">>> Training complete!", flush=True)


if __name__ == "__main__":
    main()
