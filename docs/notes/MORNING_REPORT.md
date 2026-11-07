# Overnight report — 2026-06-10

## TL;DR
Five iterations run overnight (v5.2 eval → v5.3 → v5.4 → v5.5), ending in a **breakthrough: v5.5 achieves 40% grasp / 30% lift / 0% falls DETERMINISTICALLY on the full task** (20-episode eval, random spawns) — the first nonzero deterministic success of the project. It reaches and grabs in ~0.5s, the tool is held cleanly (orientation-locked, no levitation), and the best episode is rendered at `v55_best.gif` and published on the site as Step 03. What got us here: honest contact grasp + closeness-scaled income + lift-dominant reward + adaptive spawn curriculum + **fresh weights** (the resumed-weights chain was the blocker). Remaining gaps: consolidation (40→90%+) and motion polish (decisive lunge, not yet a smooth glide) — both are sample-budget items; recommendation below stands.

## What was tried and what happened (details in TRAINING_LOG.md)

| Iter | Change | Outcome |
|---|---|---|
| v5.2 eval | 20-episode deterministic eval | **0/20 grasps — diagnosed hovering**: flat +20/step "ready" bonus made parking at 0.09–0.15m optimal; grasping *cut* income |
| v5.3 | closeness-scaled ready income; calm-hold 30/step > hover max; 250-step episodes (lift runway); resumed | hover income removed, but 0 det grasps in 8 clips after 400k — gradient alone didn't cross the last 4cm |
| v5.4 | adaptive spawn curriculum (+6cm assist, anneals with success); resumed | **regression** (grasp 12% vs 24%) — closer spawn off-distribution for resumed weights; 3rd resume chasing a moving reward = baggage |
| v5.5 | fresh weights + final reward + curriculum, 400k | **40% grasp / 30% lift / 0 falls deterministic** — first ever; grasp at step 24–32; mean min_dist 0.090 |

## What is now solid (keep)
- Honest contact-gated grasp (verified: no latch across a gap; latches on touch)
- Tool held rigidly, hand-aligned once grasped (no levitation/tumble — verified programmatically)
- Calm-approach incentives; post-grasp velocity damping; lift-dominant reward
- 250-step episodes; adaptive curriculum machinery (flag-gated)
- Eval harness `_eval_policy.py` (deterministic N-episode metrics + GIF)
- Iteration discipline: TRAINING_LOG.md + named wandb runs (`v5.x-...`)

## The honest architectural conclusion
Three reward-engineering fixes failed to move the deterministic policy through the final centimeters, while stochastic exploration grasps stayed plentiful throughout (~750/run). That pattern — exploration finds the skill, the mean policy can't consolidate it — is classically solved with **more samples and bigger batches**, not cleverer rewards. We train ~190k steps/hour on sequential CPU MuJoCo; published reach-grasp results converge at 10–100M steps on massively parallel sims.

**Recommendation: move training to the Azure GPU credits now — Isaac Lab (or MuJoCo MJX) with 1–4k parallel envs ≈ 10–50M steps/day.** The env design transfers (same obs/action/reward structure). This turns tonight's plateau into a few hours of training instead of weeks of CPU iteration. Second-choice alternative if you want to stay local another week: SAC/TQC off-policy (more sample-efficient than PPO at small scale) — but it fights the symptom, not the cause.

## Website
Progress section already live locally (Step 01 vs Step 02 + ledger). Updated honesty footnote with overnight status. No new hero clip published — nothing beat the existing one *deterministically*, and per our standard we don't publish exploration luck as skill. Site remains local-only (no deploy infra yet).
