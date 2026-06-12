# Language → Loco-Manipulation

**A humanoid robot that takes a typed instruction, plans it, verifies it, and executes it — and tells you when it can't.**

> *DRAFT — this README is written ahead of the system per deliverables-first scoping. Sections marked 🚧 describe components not yet built. When all 🚧 markers are gone, v0 is done.*

![demo](docs/demo.gif)
*🚧 90-second demo: "pick up the screwdriver and put it on the shelf" → planned, verified, executed in MuJoCo.*

---

## What this is

Type a command:

```
> pick up the screwdriver from the table and place it on the shelf
```

An LLM planner decomposes it into a sequence of skills. Each skill carries **machine-checkable pre- and postconditions** grounded in the physics state. A runtime verifier gates every transition: a skill only starts if its preconditions hold, and only counts as done if its postconditions are *measured* true — object actually displaced, grasp actually in contact, robot actually stable. On violation the executor retries or halts with a structured failure report instead of plowing ahead.

The robot is a Unitree G1 (29-DoF) in MuJoCo, balanced and steered by the AMO whole-body controller (RSS 2025), with task policies trained via PPO on Azure.

## Why the verification layer is the point

Modern policy pipelines fail **silently**. NVIDIA's own Isaac Lab G1 locomanipulation demo reports 75–85% success — meaning 1 in 5 runs fails, with nothing in the loop that notices. We trained our own grasp policies from scratch and [documented every pathology](TRAINING_LOG.md): policies that hover next to the object farming proximity reward, grasps that latch across visible gaps, motion that goes timid under penalties — all invisible in the reward curve. Capability is commoditizing; **knowing whether the robot actually did the thing is not.**

## Architecture

```
 typed command
      │
      ▼
 ┌────────────┐   JSON plan (3–5 skills)   ┌──────────────────────────┐
 │ LLM planner │ ─────────────────────────▶ │ Verifying executor        │
 └────────────┘                            │  pre-check → run → post-  │
                                           │  check → retry / halt     │
                                           └────────────┬─────────────┘
                                                        │ skill = goal for policy
                                                        ▼
                              ┌─────────────────────────────────────┐
                              │ Skills: walk_to · grasp · carry_to · │
                              │ place  (PPO policies / parametrized) │
                              └────────────────────┬────────────────┘
                                                   │ arm + base targets, 50 Hz
                                                   ▼
                              ┌─────────────────────────────────────┐
                              │ AMO whole-body controller (frozen)   │
                              │ legs + torso balance, vel. tracking  │
                              └────────────────────┬────────────────┘
                                                   ▼
                                          MuJoCo (G1, 29 DoF)
```

## Skills and their contracts

| Skill | Preconditions (gated before start) | Postconditions (measured after) |
|---|---|---|
| `walk_to(x, y, yaw)` | robot standing (height, roll/pitch in bounds) | base within tolerance of target; stable for 1 s |
| `grasp(object)` | object within reach envelope; hand free; base stable | hand–object contact latched; object lifted ≥ 5 cm; no fall |
| `carry_to(x, y)` | object held; payload within controller limits | base at target; object **still held** (continuous check) |
| `place(object, target)` | at target; object held | object at goal pose ± tol; hand free; robot stable |

🚧 Verifier module (`verifier.py`): condition spec + runtime evaluation against `mjData`.
🚧 Planner module (`planner.py`): LLM → JSON plan, schema-validated, rejected if any skill's static preconditions can't chain.
🚧 Executor (`executor.py`): state machine with retry budget and structured failure reports.

## Quickstart

```bash
git clone https://github.com/jatinsikka/g1-loco-manipulation.git && cd g1-loco-manipulation
pip install -r requirements.txt
python run_task.py --command "pick up the screwdriver and place it on the shelf"   # 🚧
```

## What exists today

- ✅ AMO whole-body control + velocity tracking in MuJoCo (walk_to substrate)
- ✅ Grasp policy: PPO, 40% → improving deterministic grasp ([v5.x log](TRAINING_LOG.md)); v5.6 trained at 1,500 fps on 32-core Azure (28× local throughput)
- ✅ Cloud training pipeline + [cost ledger](AZURE_COST_LOG.md)
- 🚧 Planner, verifier, executor, carry/place skills, demo video

## Roadmap

- **v0 — Aug 31, 2026:** end-to-end typed command → completed pick-carry-place. Ugly is fine.
- **v1 — Dec 15, 2026:** robust; demo video, this README fully de-🚧'd, 4-page workshop writeup (CoRL workshop target).

## Non-goals

Benchmark SOTA, multiple tasks/robots/scenes, real hardware, novelty for its own sake. One task, finished and verifiable, packaged well.

## License / credits

Apache 2.0. AMO controller from [Ze et al., RSS 2025]. Built by Jatin Sikka.
