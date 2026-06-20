"""press_button skill: run the trained RL press policy and report the MEASURED outcome,
so the executor can verify it against physics (button displacement + base upright), never
the policy's own claim. Mirrors skills_locomotion.run_walk_to.

Author: Jatin Sikka
"""
from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS
from env_wrapper_button import ButtonPressEnv

PRESS_THRESHOLD = 0.02   # 2cm displacement = pressed
SUSTAIN_STEPS = 25       # must be held >=25 consecutive control steps to count
ACTION_SCALE = 2.5       # must match training


def run_press_button(button: str = "button_red",
                     policy_path: str = "checkpoints_button/final_bcv5_red.zip",
                     horizon: int = 200) -> dict:
    """Run the deterministic press policy; return measured outcome:
    {pressed, max_disp, held_steps, fell, min_pelvis_z}. pressed = held >= SUSTAIN_STEPS
    past threshold AND never fell."""
    if button not in BUTTON_POSITIONS:
        button = "button_red"
    env = ButtonPressEnv(button_name=button,
                         reward_fn=ButtonPressRewardFunction(button_position=BUTTON_POSITIONS[button]),
                         freeze_arm="left", max_episode_steps=horizon, headless=True)
    env.action_scale = ACTION_SCALE
    model = PPO.load(policy_path, device="cpu")
    obs, _ = env.reset()
    max_disp = 0.0; streak = 0; held = 0; fell = False; min_z = 9.0
    for _ in range(horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        d = env._get_button_displacement()
        max_disp = max(max_disp, d)
        z = float(env.env.data.qpos[48]); min_z = min(min_z, z)
        streak = streak + 1 if d > PRESS_THRESHOLD else 0
        held = max(held, streak)
        if term:
            fell = True; break
    env.close()
    pressed = (held >= SUSTAIN_STEPS) and not fell
    return {"pressed": bool(pressed), "max_disp": float(max_disp), "held_steps": int(held),
            "fell": bool(fell), "min_pelvis_z": float(min_z)}


def run_grasp(policy_path: str = "checkpoints/v55_final.zip", horizon: int = 250,
              min_lift: float = 0.05, sustain: int = 20) -> dict:
    """Run the grasp RL policy; verified if the object is latched AND lifted >= min_lift
    sustained for `sustain` steps (measured object height, not the policy's claim)."""
    from env_wrapper import G1RLEnv
    env = G1RLEnv(policy_jit_path="amo_jit.pt", robot_type="g1", device="cpu",
                  reward_fn=None, headless=True)
    if hasattr(env, "no_early_success_stop"):
        env.no_early_success_stop = True
    model = PPO.load(policy_path, device="cpu")
    obs, _ = env.reset()
    z0 = float(env.env.data.xpos[env.screwdriver_body_id][2])
    max_lift = 0.0; streak = 0; best = 0; grasped = False
    for _ in range(horizon):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        if getattr(env, "object_grasped", False): grasped = True
        lift = float(env.env.data.xpos[env.screwdriver_body_id][2]) - z0
        max_lift = max(max_lift, lift)
        streak = streak + 1 if lift >= min_lift else 0
        best = max(best, streak)
        if term or trunc: break
    verified = grasped and best >= sustain
    return {"grasped": bool(verified), "latched": bool(grasped),
            "max_lift": float(max_lift), "held_steps": int(best)}
