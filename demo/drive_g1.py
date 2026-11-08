#!/usr/bin/env python
"""
SEQUENT — DRIVE THE ROBOT YOURSELF.

Opens the live MuJoCo viewer with the frozen AMO whole-body controller running,
and hands the keyboard to whoever is standing in front of the screen. This is
`play_amo.py` with one fix applied so it runs on a Mac (see below).

    python demo/drive_g1.py

    W / S    walk forward / back          U / J    torso yaw
    A / D    turn left / right            I / K    torso pitch
    Q / E    strafe left / right          O / L    torso roll
    Z / X    stand taller / crouch        T        random arm motion on/off
    ESC      quit

WHY THIS WRAPPER EXISTS
    amo_jit.pt was traced on a CUDA machine, and the trace baked a literal
    `device=cuda:0` into a torch.zeros call inside the graph. map_location only
    relocates parameters, not constants, so `python play_amo.py` dies with
    NotImplementedError on any machine without an NVIDIA GPU. This script loads
    the same file, rewrites those device constants to cpu, and then runs the
    unmodified play_amo environment. Nothing about the controller changes.

WHAT TO SAY WHILE IT RUNS
    Hold W. The robot walks, and it does not walk straight — it crabs. Now try
    A or D to turn: the heading barely moves. That is the frozen AMO controller
    as published, and it is the reason the project wraps it in a closed-loop yaw
    servo instead of commanding it directly. You are watching the limitation
    that shaped the architecture.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    os.chdir(REPO)                       # play_amo resolves g1.xml / *.pt relative to the repo root
    import torch
    from demo.body_walk import patch_jit_to_cpu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    policy = torch.jit.load("amo_jit.pt", map_location=device)
    if device == "cpu":
        n = patch_jit_to_cpu(policy)
        print(f"rewrote {n} baked-in cuda device constant(s) so the trace runs on cpu")

    from play_amo import HumanoidEnv
    env = HumanoidEnv(policy_jit=policy, robot_type="g1", device=device)
    if device == "cpu":
        patch_jit_to_cpu(env.adapter)

    print("\nviewer is open — click the window, then hold W to walk. ESC to quit.\n")
    env.run()


if __name__ == "__main__":
    main()
