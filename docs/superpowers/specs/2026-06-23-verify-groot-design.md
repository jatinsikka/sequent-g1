# Sequent v1 — "Verify GR00T": the trust layer on a SOTA humanoid policy

**Status:** design / awaiting final review · **Date:** 2026-06-23
**Author:** Jatin Sikka (with Claude) · **Repo:** `sequent-g1`

---

## TL;DR

Make GR00T N1.5 the most impressive possible *swappable skill* under the
executor→verifier already built for Sequent, and show the verifier catching
where the SOTA foundation model *silently fails*. The contribution stays the
same — **trust, not capability** — but now it sits on top of NVIDIA's open
humanoid foundation model instead of a hand-rolled RL policy.

This is the **v1 headline (Sept–Dec 2026), NOT a v0 change.** v0 (Aug 31)
ships with the existing RL skills, untouched.

---

## Why this, and not a new project

This was a deliberate decision, recorded here so it isn't reopened (the
finishing-discipline rule from the Master Plan).

- The Master Plan's prime directive is **ship the one artifact (Sequent)**, and
  the #1 named bottleneck is **finishing, not starting**. A second half-done
  SOTA project would actively drain the only thing that matters for résumé +
  O-1A + visa clock.
- The NVIDIA itch (Cosmos / GR00T / Isaac Lab) is legitimate and plan-sanctioned
  — but as a *fold-in*, not a new front. The plan literally names
  "MuJoCo Playground (Newton) + Isaac Lab" as the headline skill and says
  *"don't start two from zero."*
- The fold-in **sharpens the moat instead of diluting it.** The README already
  says: *"capability is commoditizing (Isaac Lab Mimic, GRAIL); knowing whether
  the robot actually did the thing is not."* GR00T is the ultimate commodity
  capability — verifying it is the differentiated act.

### The grounding fact that made this work

**GR00T N1.5/N1.7 is open (Apache 2.0) and was post-trained on the Unitree G1**
— the exact robot Sequent runs. It's SOTA (98.8% on known-object placement) and
slots directly into the "skills are swappable" architecture.

- GR00T N1.5: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Isaac-GR00T (Apache 2.0, G1): https://github.com/NVIDIA/Isaac-GR00T
- GR00T-Dreams / DreamGen: https://github.com/NVIDIA/GR00T-dreams
- Cosmos synthetic data: https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/

---

## The narrative (résumé line + demo)

> "I took NVIDIA's open GR00T foundation model — the SOTA humanoid policy, on my
> exact robot — and pointed Sequent's physics verifier at it. It reports ~99% in
> the lab; my verifier caught where it silently fails."

Puts **Isaac GR00T, Cosmos, Isaac Lab** legitimately on the CV, and the demo
shows the trust layer working on a model recruiters recognize.

---

## Architecture (mostly reuse — that's the point)

```
incident (NL)
     │
     ▼
 brain/            (UNCHANGED)  SOP retrieval + LLM planner → typed plan
     │
     ▼
 executor.py       (UNCHANGED)  drives plan step-by-step
     │
     ▼
 skills_groot.py   (NEW)        GR00T-backed skill: render MuJoCo camera frame
     │                          → GR00T inference → action chunk → G1 actuators
     ▼
 verifier.py       (UNCHANGED)  pre/postconditions vs mjData → pass/retry/halt
```

The verifier and executor are **untouched on purpose** — that is the proof that
the trust layer generalizes across policies.

### Components / units

| Unit | What it does | Depends on |
|---|---|---|
| `skills_groot.py` | `GROOTSkill(obs_render, instruction) → action`, conforming to the existing skill contract (same shape as `skills_manipulation.py`) | GR00T inference, MuJoCo camera render, G1 actuator map |
| `groot_eval.py` | claimed-vs-verified harness over N GR00T rollouts (mirrors `verify_policy.py`); emits the headline number | `skills_groot`, `verifier` |
| `verifier.py`, `executor.py`, `brain/` | unchanged | — |

### Data flow

instruction + camera frame → GR00T → action chunk → MuJoCo step → `mjData` →
verifier postcondition → pass / retry / halt-with-report.

---

## Phase 0 — feasibility spike (1–2 days, HARD GATE)

The one real risk is GR00T's action/observation/embodiment format not matching
the G1 config (`g1_robotiq.xml` gripper). Front-load it.

**Spike objective:** load GR00T N1.5 weights, run inference on one rendered G1
camera frame, confirm the action output maps to G1 actuators for a single
grasp in MuJoCo.

**Gate decision:**
- Runs end-to-end on one grasp → **proceed with Flavor A (live GR00T).**
- Embodiment/action-mapping wall → **fall back to Flavor B:** verify
  GR00T-Dreams / Cosmos *synthetic* "neural trajectories" against physics
  ("are these dreamed trajectories physically valid?"). Novel, self-contained,
  no live VLA to plumb. Less visually punchy but still SOTA-flavored.

Decision made on evidence, not hope.

### Open questions for the spike to answer
- GR00T N1.5 action space / output format and how it maps to G1 actuators.
- Observation format (camera count, resolution, state vector) GR00T expects.
- Inference compute (reuse the existing Azure GPU pipeline; see `AZURE_COST_LOG.md`).
- Whether the G1 reference workflow (NVIDIA says "coming soon") is available yet.

---

## Deliverable

A **30–90s video**: *"Machine A pressure low"* → plan → GR00T runs the
grasp/press step on the G1 → verifier overlay catches a step GR00T silently
botched, halts with a structured report.
**Headline stat: GR00T claimed X%, verified Y%.**

Plus: clean public addition to the repo + the three-narration writeup
(consulting / FDE / robotics) per the Master Plan's force-multiplier rule.

---

## Sequencing (the guardrail)

- **v0 (Aug 31, 2026):** existing RL skills, untouched. Prime directive.
- **v1 (Sept–Dec 2026):** this work. Demoable by **Dec 15**, matching the plan's
  gates.
- If this work ever threatens the v0 gate, it stops. v0 ships first.

---

## Explicitly OUT (YAGNI)

- No Isaac Lab re-platform of the existing pipeline (the re-platforming trap).
- No GR00T fine-tuning.
- No Cosmos data-generation (that's Flavor B territory only).
- No whole-body GR00T — locomotion stays on AMO; GR00T does the manipulation
  step only.
- No hardware.

---

## Testing

- The **claimed-vs-verified eval (`groot_eval.py`) is the test** — same pattern
  that already caught the reward-hacked grasp checkpoint (0% verified vs claimed
  15–55%).
- Unit check: `GROOTSkill` output shape conforms to the skill contract.

---

## Next step

Feasibility spike (Phase 0). If green, the implementation plan covers
`skills_groot.py` → `groot_eval.py` → demo video.
