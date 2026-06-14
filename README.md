# Sequent

**SOP-driven, verified loco-manipulation on a humanoid.** Tell it what's wrong; it retrieves the right procedure, plans the steps, executes them on a Unitree G1 in simulation, and — the part nobody else does — *verifies each step against the physics* instead of assuming it worked.

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

- ✅ Whole-body control + walking (AMO), grasp skill (RL, **v5.5 = 55% verified**), cloud training pipeline
- ✅ Verifier + no-early-stop eval; **verifying executor** (`executor.py`, `run_task.py`): command → plan → verified step-by-step execution, halts with a report on failure
- 🚧 Real brain planner wired into `run_task` (currently a keyword stub), more real skills (walk_to, press_button), the demo video

## Roadmap

- **v0 (Aug 31, 2026):** typed command → retrieved SOP → verified end-to-end execution on one task. Ugly is fine.
- **v1 (Dec 15, 2026):** robust; 90-second demo video, clean repo, short writeup.

## Credits

AMO controller: Ze et al., RSS 2025. Built by Jatin Sikka. Apache 2.0.
