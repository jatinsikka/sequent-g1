"""
Run a trained policy under the verification layer.

For each deterministic episode: check the grasp contract's preconditions at
spawn, run the policy, and judge the episode by measured postconditions
(latch + sustained lift + stability) instead of the policy's own claim.

The point: "claimed" (contact latch fired) vs "verified" (it actually did
the job). v5.6 claims 45% — the verifier should award it 0%.

Usage:
    python _verify_policy.py --model checkpoints/ppo_g1_9920000_steps.zip --episodes 20
"""

import argparse

import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

from config import get_training_config
from env_wrapper import G1RLEnv
from reward_fn import RewardFunction
from verifier import grasp_contract, LiftMonitor, verdict


def make_env(config):
    reward_fn = RewardFunction(
        target_position=config.target_position,
        target_object_pos=(config.target_position[0], config.target_position[1], 0.74),
        desired_height=config.desired_height,
    )
    env = G1RLEnv(
        policy_jit_path="amo_jit.pt",
        robot_type=config.robot_type,
        device=config.device,
        action_scale=config.action_scale,
        max_episode_steps=config.max_episode_steps,
        reward_fn=reward_fn,
        headless=True,
        min_height=config.min_height,
        max_roll=config.max_roll,
        max_pitch=config.max_pitch,
        goal_distance=config.goal_distance,
        max_episode_time=config.max_episode_time,
        freeze_arm=config.freeze_arm,
        verbose=0,
    )
    return TimeLimit(env, max_episode_steps=config.max_episode_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--min_lift", type=float, default=0.05)
    p.add_argument("--sustain_steps", type=int, default=25)
    args = p.parse_args()

    config = get_training_config()
    env = make_env(config)
    model = PPO.load(args.model, device=config.device)
    contract = grasp_contract()

    claimed = verified = pre_failures = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()

        pre = contract.check_pre(env)
        if not verdict(pre):
            pre_failures += 1
            failed = [r for r in pre if not r.ok]
            print(f"ep{ep:02d}: PRECONDITION FAIL — skill refused: "
                  + "; ".join(str(r) for r in failed))
            continue

        lift = LiftMonitor(min_lift=args.min_lift, sustain_steps=args.sustain_steps)
        lift.reset(env)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            lift.update(env)
            done = terminated or truncated

        post = contract.check_post(env, monitors=[lift])
        ep_claimed = env.unwrapped.object_grasped
        ep_verified = verdict(post)
        claimed += int(ep_claimed)
        verified += int(ep_verified)

        failed = [r for r in post if not r.ok]
        status = "VERIFIED" if ep_verified else ("CLAIMED-ONLY" if ep_claimed else "no grasp")
        detail = "" if ep_verified else " | " + "; ".join(str(r) for r in failed) if failed else ""
        print(f"ep{ep:02d}: {status}{detail}")

    n = args.episodes
    print("\n===== VERIFICATION SUMMARY =====")
    print(f"episodes:            {n}")
    print(f"preconditions held:  {n - pre_failures}/{n}")
    print(f"policy-claimed:      {claimed}/{n}  ({100 * claimed / n:.0f}%)   <- contact latch fired")
    print(f"verified complete:   {verified}/{n}  ({100 * verified / n:.0f}%)   <- latch + lift >= "
          f"{args.min_lift * 100:.0f}cm sustained {args.sustain_steps} steps + stable")
    if claimed and not verified:
        print("\nEvery claimed success failed measured postconditions — "
              "the score lied; the physics did not.")

    env.close()


if __name__ == "__main__":
    main()
