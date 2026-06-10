"""Deterministic evaluation harness for reach-and-grasp checkpoints.

Runs N deterministic episodes over random spawns and reports the metrics that
actually decide iteration choices:
  - grasp rate / lift rate (deterministic = what the deployed skill does)
  - mean/best min hand-object distance
  - smoothness: mean per-step action change and mean arm joint speed
  - renders a GIF of the best episode for visual judgment

Usage:
  python _eval_policy.py --model checkpoints/final_model.zip --episodes 20 --gif out.gif
"""
import argparse
import numpy as np

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

from env_wrapper import G1RLEnv
from config import get_training_config


def make_env(config, max_steps=None, max_time=None):
    env = G1RLEnv(
        policy_jit_path="amo_jit.pt",
        robot_type=config.robot_type,
        device=config.device,
        action_scale=config.action_scale,
        max_episode_steps=max_steps or config.max_episode_steps,
        headless=True,
        min_height=config.min_height,
        max_roll=config.max_roll,
        max_pitch=config.max_pitch,
        goal_distance=config.goal_distance,
        max_episode_time=max_time or config.max_episode_time,
        freeze_arm=config.freeze_arm,
        verbose=0,
    )
    return TimeLimit(env, max_episode_steps=max_steps or config.max_episode_steps)


def run_episode(model, env, deterministic=True, record=False):
    obs, _ = env.reset()
    frames = []
    ep = {
        "return": 0.0, "min_dist": float("inf"), "grasped": False, "lifted": False,
        "fell": False, "steps": 0, "action_rate": [], "arm_speed": [],
    }
    prev_action = None
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        ep["return"] += float(reward)
        ep["steps"] += 1
        ep["min_dist"] = min(ep["min_dist"], info.get("min_hand_dist", float("inf")))
        ep["grasped"] = ep["grasped"] or bool(info.get("object_grasped", False))
        ep["lifted"] = ep["lifted"] or bool(info.get("success", False))
        ep["fell"] = ep["fell"] or bool(info.get("fell", False))
        if prev_action is not None:
            ep["action_rate"].append(float(np.linalg.norm(action - prev_action)))
        prev_action = action.copy()
        if record:
            try:
                frames.append(env.unwrapped.render_frame(width=480, height=360))
            except Exception:
                record = False
        if term or trunc:
            break
    ep["mean_action_rate"] = float(np.mean(ep["action_rate"])) if ep["action_rate"] else 0.0
    return ep, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--gif", default=None, help="render best episode to this GIF path")
    p.add_argument("--stochastic", action="store_true", help="also run stochastic for comparison")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--max_time", type=float, default=None)
    args = p.parse_args()

    config = get_training_config()
    env = make_env(config, args.max_steps, args.max_time)
    model = PPO.load(args.model, device=config.device)
    print(f"[EVAL] {args.model} | {args.episodes} deterministic episodes")

    results = []
    for i in range(args.episodes):
        ep, _ = run_episode(model, env, deterministic=True)
        results.append(ep)
        print(f"  ep{i:02d}: dist={ep['min_dist']:.3f} grasp={int(ep['grasped'])} "
              f"lift={int(ep['lifted'])} ret={ep['return']:.0f} act_rate={ep['mean_action_rate']:.3f}"
              f"{' FELL' if ep['fell'] else ''}")

    n = len(results)
    grasp = sum(r["grasped"] for r in results) / n
    lift = sum(r["lifted"] for r in results) / n
    fell = sum(r["fell"] for r in results) / n
    print("\n===== DETERMINISTIC SUMMARY =====")
    print(f"grasp rate:      {grasp:.0%}")
    print(f"lift rate:       {lift:.0%}")
    print(f"fall rate:       {fell:.0%}")
    print(f"mean min_dist:   {np.mean([r['min_dist'] for r in results]):.3f} m")
    print(f"best min_dist:   {np.min([r['min_dist'] for r in results]):.3f} m")
    print(f"mean return:     {np.mean([r['return'] for r in results]):.0f}")
    print(f"mean act_rate:   {np.mean([r['mean_action_rate'] for r in results]):.3f}  (smoothness; lower=calmer)")

    if args.stochastic:
        sto = [run_episode(model, env, deterministic=False)[0] for _ in range(args.episodes)]
        print("\n===== STOCHASTIC (exploration) =====")
        print(f"grasp rate: {sum(r['grasped'] for r in sto)/n:.0%} | lift rate: {sum(r['lifted'] for r in sto)/n:.0%}")

    if args.gif:
        # re-run and record: prefer a lifting episode, then a grasping one, else closest
        print("\n[EVAL] rendering best-of-12 episode to GIF...")
        best_frames, best_key = None, (-1, -1, float("inf"))
        for _ in range(12):
            ep, frames = run_episode(model, env, deterministic=True, record=True)
            key = (int(ep["lifted"]), int(ep["grasped"]), -ep["min_dist"])
            if frames and key > best_key:
                best_key, best_frames = key, frames
        if best_frames:
            import imageio
            imageio.mimsave(args.gif, best_frames, fps=25)
            print(f"[EVAL] wrote {args.gif} ({len(best_frames)} frames) "
                  f"lift={best_key[0]} grasp={best_key[1]}")


if __name__ == "__main__":
    main()
