# Sequent

**SOP-driven, verified loco-manipulation on a humanoid.** Tell it what's wrong; it retrieves the right procedure, plans the steps, executes them on a Unitree G1 in simulation, and — the part nobody else does — *verifies each step against the physics* instead of assuming it worked.

<p align="center">
  <img src="demo_reach_press.gif" alt="Unitree G1: RL curriculum policy reaches from the rest pose and presses the panel button (no IK seed)" width="640">
</p>

> Repo: `sequent-g1`. Part of [Sequent Robotics](https://sequent-robotics.vercel.app).

```
 "Machine A pressure is low"
            │
            ▼
   ┌──────────────────┐   retrieves over a 100-SOP library
   │  BRAIN (brain/)   │   emits a typed, step-by-step plan
   │  retrieval + LLM  │   (walk_to · grasp · press_button · …)
   │  planner          │
   └────────┬─────────┘
            │  JSON plan: ordered skills + pre/postconditions
            ▼
   ┌──────────────────┐   gates every step: precondition → run →
   │  VERIFIER         │   postcondition measured against mjData →
   │  (verifier.py)    │   retry / halt with a structured report
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐   RL skills on the AMO whole-body controller
   │  BODY (skills)    │   G1, 29 DoF, MuJoCo
   └──────────────────┘
```

## Why this exists

Modern policy pipelines fail **silently** — NVIDIA's own G1 locomanipulation demos report 75–85% success, with nothing in the loop that notices the other 15–25%. Capability is commoditizing fast (Isaac Lab Mimic, GRAIL); *knowing whether the robot actually did the thing* is not. Sequent is the verification layer: the skills are commodity and swappable, the trust layer above them is the contribution.

The verifier earns this on day one — pointed at our own best grasp policy, it caught a reward-hacked checkpoint that **claimed** 15–55% grasp success but **verified** at 0% (grabs the tool, never lifts it). See `TRAINING_LOG.md`.

## Repository layout

| Path | What |
|---|---|
| `brain/` | SOP retrieval (100-SOP library) + LLM planner → typed JSON plans. From the Fall-2025 DL project; LLM layer being refreshed to a frontier model, execution being de-stubbed. |
| `verifier.py` | Skill contracts: physics-checked pre/postconditions vs. `mjData`. |
| `verify_policy.py` | Claimed-vs-verified evaluation of a trained policy. |
| `train.py`, `env_wrapper*.py`, `reward_fn.py` | RL skills (grasp, button-press) on the AMO controller. |
| `g1.xml`, `meshes/`, `*.xml` | G1 model + scene. |
| `amo_jit.pt`, `adapter_jit.pt` | Frozen AMO whole-body controller (Ze et al., RSS 2025). CPU-patched variants for cloud training. |
| `TRAINING_LOG.md`, `AZURE_COST_LOG.md` | Per-run results and cloud spend. |

## Status

- ✅ **One unified robot**: walking AMO humanoid + Robotiq 2F-85 grafted on the right wrist — every skill runs on the same embodiment (`build_unified_model.py`, `unified_env.py`)
- ✅ **RL reach-and-press via curriculum** (`train_button.py --curriculum`): the arm starts between the contact pose and the true rest pose, and the start distance grows with success — the deterministic policy reaches ~19 cm from rest and presses 28–31 mm, **no IK seed**. Key fixes: reach reward targets the button *cap face* (not the body origin) + an anti-parking proximity band. Smoothness is forced at the control level (low-pass on the arm command; jerk 0.83 → 0.001) — a reward-only anti-jerk penalty got hacked.
- ✅ **Pick from the real table** — a controller, not RL: DLS-IK reach + force-gated latch, held lift, block stays on the tabletop
- ✅ Verifier + no-early-stop eval; **verifying executor** (`executor.py`, `run_task.py`): command → plan → verified step-by-step execution, halts with a report on failure
- ✅ **SOP-driven spine** (`brain_bridge.py`) + **LLM planner** (`llm_planner.py`): incident → SOP retrieval → typed plan → verified execution
- 🚧 Press hold + stance-robustness retrain (seamless walk→press handoff), lever on the same curriculum, one-take end-to-end demo

## Roadmap

- **v0 (Aug 31, 2026):** typed command → retrieved SOP → verified end-to-end execution on one task. Ugly is fine.
- **v1 (Dec 15, 2026):** robust; 90-second demo video, clean repo, short writeup.

## Credits

AMO controller: Ze et al., RSS 2025. Built by Jatin Sikka. Apache 2.0.
