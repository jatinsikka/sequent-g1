# Experiments

One-off scripts from development: probes, superseded environments, and the experiment runs the paper mentions. The demos and training entry points do not use them. Some are out of date with the current environment code.

They import modules from the repo root, so run them from there:

```bash
PYTHONPATH=. python experiments/<script>.py
```

| File | What it was for |
|---|---|
| `train_button_bc.py` | Press: behavior-cloning warm start, then PPO fine-tune (bp-v3) |
| `resume_press.py` | Press: resumed PPO run that pushed the hold past 25 steps (bp-v5) |
| `walk_pick_demo.py` | Gate 4: walk to the table and pick the box in one simulation |
| `walk_and_press.py`, `walk_and_grasp_cont.py` | Early one-simulation walk-then-press / walk-then-grasp runs |
| `sop_demo.py`, `sop_demo_trace.txt` | Early end-to-end demo (ticket → SOP → plan → verified steps) and its output |
| `teleop_pick.py` | Teleoperated picks for recording BC demos |
| `lever_env.py` | First lever environment (replaced by `lever_press_env.py`) |
| `push_env.py`, `train_push.py` | Non-prehensile push task, not used in the final system |
| `env_wrapper_universal.py`, `train_universal.py` | Early attempt at one policy for all manipulation tasks |
| `eval.py`, `utils.py` | Early evaluation script and helpers for the first RL setup |
| `plot_bp.py` | Plots an SB3 `progress.csv` for the press runs |
| `export_scene.py` | Exports a rollout for a browser 3D viewer |
| `view_grasp.py` | Static viewer for the G1 + Robotiq gripper model |
| `test_*.py` | Early probes: joint directions, PID, magnetic grasp, button reachability |
| `g1_grasp.xml`, `g1_gripper.xml`, `purple_object.xml` | Older scene files not loaded by the current builders |
