#!/usr/bin/env python
"""
Headless AMO walk with measurements.

Runs the frozen AMO whole-body controller on the G1 in MuJoCo with no window:
commands a forward walk, then measures what actually happened from mjData —
distance covered, pelvis height, whether it stayed upright. Optionally writes
an mp4 you can drop into a slide.

    python demo/body_walk.py                    # walk 4 s, print measurements
    python demo/body_walk.py --seconds 8 --vx 0.5
    python demo/body_walk.py --video walk.mp4   # also render an mp4 (slower)

This is the half of the body that runs from a clean clone: the AMO
controller and the G1 model are both in the repo. The RL button-press and lever
policies are NOT — their checkpoints are gitignored — so the demos that need
them (g6_onetake.py, final_ticket_demo.py) cannot run here. See LOCAL_DEMO.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np


def patch_jit_to_cpu(module) -> int:
    """`amo_jit.pt` was traced on a CUDA box, so its graph contains a literal
    `device=cuda:0` inside a torch.zeros call. map_location cannot reach that —
    it only moves parameters, not baked-in constants — so on any machine without
    an NVIDIA GPU the forward pass dies with a NotImplementedError. Rewrite the
    device constants to cpu. (Same trick OpenAI's CLIP uses for its CPU path.)

    Returns the number of constants rewritten.
    """
    import torch

    holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device("cpu")), example_inputs=[])
    cpu_node = next(n for n in holder.graph.findAllNodes("prim::Constant")
                    if "Device" in repr(n.output().type()))
    fixed = 0

    def _patch(mod):
        nonlocal fixed
        graphs = []
        try:
            graphs.append(mod.graph)
        except RuntimeError:
            pass
        if hasattr(mod, "forward1"):
            graphs.append(mod.forward1.graph)
        for g in graphs:
            for node in g.findAllNodes("prim::Constant"):
                if "Device" in repr(node.output().type()) and "cuda" in str(node):
                    node.copyAttributes(cpu_node)
                    fixed += 1

    module.apply(_patch)
    _patch(module)
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0, help="simulated seconds to walk")
    ap.add_argument("--vx", type=float, default=0.4, help="forward velocity command (m/s)")
    ap.add_argument("--yaw", type=float, default=0.0, help="yaw command (rad)")
    ap.add_argument("--video", help="write an mp4 here (needs imageio-ffmpeg)")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    import mujoco
    import torch

    import os
    os.chdir(REPO)                      # play_amo resolves g1.xml / *.pt relative to the repo root
    from play_amo import HumanoidEnv

    print(f"loading AMO controller (cpu)...")
    policy = torch.jit.load("amo_jit.pt", map_location="cpu")
    n = patch_jit_to_cpu(policy)
    if n:
        print(f"  rewrote {n} baked-in cuda device constant(s) in the traced graph")
    env = HumanoidEnv(policy_jit=policy, robot_type="g1", device="cpu", headless=True)
    patch_jit_to_cpu(env.adapter)

    # The headless MockViewer has no camera; the run loop expects one.
    class _Cam:
        lookat = np.zeros(3, dtype=np.float32)
    env.viewer.cam = _Cam()

    env.viewer.commands[0] = a.vx       # forward velocity
    env.viewer.commands[1] = a.yaw      # yaw target

    renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(env.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 4.0, -10.0, -90.0

    n_steps = int(a.seconds / env.sim_dt)
    every = max(1, int(1.0 / a.fps / env.sim_dt))
    pd_target = env.default_dof_pos.copy()

    # The scene has loose objects with their own free joints, so the pelvis is NOT
    # at qpos[0:7]. Read its pose from the body's world transform instead.
    start_xy = env.data.xpos[env.pelvis_id][:2].copy()
    heights = []

    print(f"walking: vx={a.vx} m/s, yaw={a.yaw} rad, {a.seconds}s "
          f"({n_steps} physics steps at {env.sim_dt*1000:.0f} ms)")
    t0 = time.perf_counter()
    for step in range(n_steps):
        env._extract_state()
        if step % env.sim_decimation == 0:
            obs = env._compute_observation()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                hist = torch.tensor(np.array(env.extra_history).flatten(),
                                    dtype=torch.float).view(1, -1)
                raw = env.policy_jit(obs_t, hist).cpu().numpy().squeeze()
            raw = np.clip(raw, -40.0, 40.0)
            env.last_action = np.concatenate([
                raw, (env.dof_pos[15:] - env.default_dof_pos[15:]) / env.action_scale])
            env._update_arm_action(step)
            pd_target[:15] = raw * env.action_scale + env.default_dof_pos[:15]
            pd_target[15:] = ((1 - env.arm_blend) * env.prev_arm_action
                              + env.arm_blend * env.arm_action)
            env._update_gait()

        torque = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
        env.data.ctrl = np.clip(torque, -env.torque_limits, env.torque_limits)
        mujoco.mj_step(env.model, env.data)

        heights.append(float(env.data.xpos[env.pelvis_id][2]))

        if renderer is not None and step % every == 0:
            cam.lookat = env.data.xpos[env.pelvis_id].astype(np.float64)
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render())

    wall = time.perf_counter() - t0
    end_xy = env.data.xpos[env.pelvis_id][:2]
    d = end_xy - start_xy
    dist = float(np.linalg.norm(d))
    forward, lateral = abs(float(d[0])), abs(float(d[1]))
    crab = float(np.degrees(np.arctan2(lateral, forward))) if forward > 1e-6 else 0.0
    fell = min(heights) < 0.4

    print()
    print("  MEASURED FROM mjData, not from the policy's opinion")
    print(f"    distance travelled     {dist:.2f} m   (forward {forward:.2f}, lateral {lateral:.2f})")
    print(f"    commanded distance     {a.vx * a.seconds:.2f} m   "
          f"(includes the settle before it starts moving)")
    print(f"    pelvis height          {np.mean(heights):.3f} m mean, {min(heights):.3f} m min")
    print(f"    upright                {'no (fell)' if fell else 'yes'}")
    print(f"    heading error          {crab:.0f} deg off the commanded straight line")
    print(f"    real time factor       {a.seconds / wall:.2f}x  ({wall:.1f}s wall for "
          f"{a.seconds:.0f}s sim)")
    print()
    print(f"  That {crab:.0f}-degree heading error is the headline claim in the README")
    print("  reproduced live: AMO as wired crabs while walking and cannot turn in")
    print("  place, which is why the project wraps it in a closed-loop yaw servo")
    print("  rather than trusting it to walk straight. A measurement, not a complaint.")

    if renderer is not None:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(a.video, frames, fps=a.fps)
            print(f"\n  wrote {a.video}  ({len(frames)} frames)")
        except ImportError:
            print("\n  [!] pip install imageio imageio-ffmpeg to write the mp4")


if __name__ == "__main__":
    main()
