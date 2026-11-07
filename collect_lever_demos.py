"""Collect scripted lever pull-down demos for BC warm-start (lever twin of the
button's collect_demos in train_button_bc.py).

Per episode: curriculum reset (frac 0.6-1.0 -> mid/far arm starts, lever partially up),
then a two-phase script:
  1. REACH: command the curriculum's CACHED physically-servoed contact pose
     (env._a4c_by_bucket) as a constant action. Analytic arc-waypoint IK approaches were
     tried first and FAILED: solve_right_arm7_ik redistributes reach into the 3 wrist
     joints (held fixed during episodes -> executed pose lands ~15cm off) and the 4-DOF
     right_arm_ik_step plateaus far below high targets, so the gripper swept through the
     knob from BELOW and flung the lever UP past its latch. The cached pose is the one
     arm pose PROVEN to sit on the crown, and it is inside the action envelope
     (max |a4_c - a4_start| ~ 1.1 < action_scale 1.2).
  2. PULL: per-step closed-loop 4-DOF IK (right_arm_ik_step, the button demo's
     ik_press_action pattern) toward the ACTUAL knob position + a small press-in offset,
     so the pad rides the crown with constant downward force through the whole arc.

(obs, action) recorded at every step; only episodes ending with lever angle < 0.25 are
kept. Saved to lever_demos.npz.

Run: python collect_lever_demos.py
"""
import argparse
import numpy as np
from lever_press_env import LeverPressEnv

TARGET_ANGLE = 0.12                          # script slightly past the 0.15 task target
SUCCESS_ANGLE = 0.25
PRESS_OFFSET = np.array([0.0, 0.0, 0.005])   # just above knob center -> presses the crown


def a4_to_action(env: LeverPressEnv, a4: np.ndarray) -> np.ndarray:
    """Invert step()'s arm command: the action whose (converged) filtered command PDs the
    right shoulder/elbow to a4. Left arm zeros (frozen)."""
    act = np.zeros(8, dtype=np.float32)
    act[4:] = np.clip(
        (a4 - env.env.default_dof_pos[19:23] - env.arm_reach_bias[4:]) / env.action_scale,
        -1.0, 1.0)
    return act


def run_episode(env, seed, noise=0.0, reach_steps=45, pull_steps=190, hold_steps=15):
    """noise > 0 injects uniform action noise (executed AND recorded — the button demo's
    DART-style trick): the next-step closed-loop IK corrects the deviation, so BC gets
    RECOVERY data around the nominal path instead of a razor-thin state tube."""
    obs, _ = env.reset(seed=seed)
    obs_l, act_l = [], []
    frac = getattr(env, "_cur_frac", -1.0)
    start_angle = env._get_lever_angle()

    def record_step(act, nz):
        nonlocal obs
        if nz > 0:
            act = act.copy()
            act[4:] = np.clip(act[4:] + np.random.uniform(-nz, nz, 4), -1.0, 1.0)
        obs_l.append(obs.copy())
        act_l.append(act.copy())
        obs, r, term, trunc, info = env.step(act)
        return term or trunc

    # --- phase 1: reach the cached contact pose (constant action; low-pass converges) ---
    bucket = round(frac * 5) / 5.0
    a4_c, _w3_c = env._a4c_by_bucket[bucket]
    reach_act = a4_to_action(env, a4_c)
    for _ in range(reach_steps):
        if record_step(reach_act, noise * 0.6):
            return obs_l, act_l, frac, start_angle, env._get_lever_angle()

    # --- phase 2: pull — servo the pad into the actual knob crown down the arc ---
    for _ in range(pull_steps):
        if env._get_lever_angle() <= TARGET_ANGLE + 0.01:
            break
        a4 = env.env.right_arm_ik_step(env._get_handle_pos().copy() + PRESS_OFFSET)
        if record_step(a4_to_action(env, a4), noise):
            return obs_l, act_l, frac, start_angle, env._get_lever_angle()

    # --- phase 3: hold at target so BC learns to STAY, not overshoot ---
    for _ in range(hold_steps):
        a4 = env.env.right_arm_ik_step(env._get_handle_pos().copy() + PRESS_OFFSET)
        if record_step(a4_to_action(env, a4), 0.0):
            break
    return obs_l, act_l, frac, start_angle, env._get_lever_angle()


def run_dagger_episode(env, policy, seed):
    """Roll out the CURRENT BC policy; label ONLY near-contact (pull-phase) states with
    the per-step IK expert. Reach-phase states are skipped: their expert label is the
    bias-dependent constant action (arm_reach_bias varies per episode and is not in the
    obs), so labeling them re-poisons BC — a full-phase DAgger round regressed verify
    0.68->0.89. The stalls all happen near contact, which is exactly what this covers."""
    obs, _ = env.reset(seed=seed)
    obs_l, act_l = [], []
    frac = getattr(env, "_cur_frac", -1.0)
    start_angle = env._get_lever_angle()
    for _ in range(env.max_episode_steps):
        tgt = env._get_handle_pos().copy() + PRESS_OFFSET
        if np.linalg.norm(env.env.gripper_point() - tgt) < 0.07:
            obs_l.append(obs.copy())
            act_l.append(a4_to_action(env, env.env.right_arm_ik_step(tgt)))
        act, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        if term or trunc:
            break
    return obs_l, act_l, frac, start_angle, env._get_lever_angle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eps", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="action noise on 2/3 of episodes (recovery-data coverage)")
    ap.add_argument("--out", default="lever_demos.npz")
    ap.add_argument("--frac_min", type=float, default=0.6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--dagger", default=None,
                    help="path to a BC policy .zip: roll IT out and label with the expert")
    args = ap.parse_args()

    env = LeverPressEnv(unified=True, reset_in_contact=False, curriculum=True,
                        headless=True, max_episode_steps=260)
    env.set_curriculum_frac(1.0)
    env.curriculum_frac_min = args.frac_min

    policy = None
    if args.dagger:
        from stable_baselines3 import PPO
        policy = PPO.load(args.dagger, device="cpu")

    all_obs, all_act, ep_len, ep_frac, n_ok = [], [], [], [], 0
    for ep in range(args.n_eps):
        if policy is not None:
            obs_l, act_l, frac, a_start, a_end = run_dagger_episode(env, policy, seed=args.seed0 + ep)
            ok = True                        # DAgger keeps every episode
        else:
            nz = args.noise if ep % 3 else 0.0   # 1/3 clean, 2/3 noisy
            obs_l, act_l, frac, a_start, a_end = run_episode(env, seed=args.seed0 + ep, noise=nz)
            ok = a_end < SUCCESS_ANGLE
        print(f"[EP {ep:02d}] frac={frac:.2f} start={a_start:.3f} "
              f"final={a_end:.3f} steps={len(obs_l)} {'SUCCESS' if ok else 'fail'}", flush=True)
        if ok:
            all_obs.extend(obs_l)
            all_act.extend(act_l)
            ep_len.append(len(obs_l))
            ep_frac.append(frac)
            n_ok += 1
    env.close()

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_act, dtype=np.float32)
    np.savez(args.out, obs=obs_arr, act=act_arr,
             ep_len=np.array(ep_len, dtype=np.int64),
             ep_frac=np.array(ep_frac, dtype=np.float32))
    print(f"\n[DONE] {n_ok}/{args.n_eps} successful episodes, "
          f"{len(obs_arr)} transitions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
