# Overnight Master Plan — 2026-06-30 → 07-01 (~10 hrs autonomous)

**Standing goal:** Language→Loco-Manip — LLM planner → verified skills executing manufacturing SOPs on the G1.
**Tonight's goal:** land the **lever** as a clean RL skill, then push toward the **end-to-end verified SOP demo** (walk → grasp → press → lever) and get it all on GitHub + the website.

## Skill status going in
- `walk_to` — controller (AMO+PID). DONE.
- **grasp/pick** — IK controller (`ik_pick.py`, DLS IK). DONE, verified held lift 11/12.
- `press_button` — RL (AMO panel rig). DONE, verified bp-v5.
- **`lever`** — RL (AMO panel rig, push-up config). **← tonight's primary: train + verify.**
- `push` — RL (fixed-base). PARKED (free-box non-prehensile was finicky; lever covers the niche).

## Cost discipline (track every wake)
- VM = F64als_v7 ≈ **$2.74/hr**. 10 hrs ≈ **$27**. Running tally started ~$15–18 (pre-overnight).
- **Deallocate the VM during any gap** (no run training). Restart only to train.
- One VM, sequential full-speed runs (multiple PPO/box thrash — proven). Spin a 2nd VM only if a clear parallel win.

## Phase plan (re-evaluate each wake; don't tunnel)
1. **Lever training (primary).** Ship AMO rig → train `lever-v0` (~1.5–2M, AMO env). Review curve + render. Success = lever driven to target angle, held, robot upright. Iterate reward/steps if needed — **max ~3 iterations**, then advance or reassess.
2. **On lever success:** render a clean rollout, commit + push, update website with the lever result.
3. **End-to-end SOP demo:** wire grasp(IK) + press_button + lever + walk_to into `sop_demo.py`; verify a full NL→SOP→plan→execute→verify run. This is the thesis deliverable.
4. **Website:** the honest pivot narrative (grasp=controller via IK pick; RL=contact-rich press+lever; the dead-end findings) + measured numbers only. Draft locally; do NOT auto-deploy outward without Jatin — leave a `vercel --prod` ready + flagged.
5. **Fallback / spare-time (if lever stalls or finishes early), in priority order:** (a) press_button polish run for the reel; (b) push retry with a *hardcoded* hand-at-box reset (the clean version); (c) tighten ik_pick edge cases; (d) repo cleanup + TRAINING_LOG updates.

## Iteration protocol (per run)
- Launch run + a background watcher that notifies on completion/crash.
- On wake: read result → render/frame-check (verify the SEMANTIC, e.g. lever held at target, not just a metric — the v12d lesson) → decide iterate vs advance → launch next.
- **Never claim success off a metric alone; verify the behavior.**
- Commit at every real milestone (no Claude trailer; branch `skills/manipulation`).

## Guardrails
- Only-measured-numbers on site/docs.
- Don't deploy the website or do other outward-facing actions without Jatin — prepare + flag.
- If something is genuinely ambiguous/visual and blocks progress, leave the best-judgment call + a clear note for Jatin rather than stalling.
- Keep this file + a running log of what happened each iteration (append to `OVERNIGHT_LOG.md`).

## E2E Demo Gate Plan (2026-07-02, Jatin-approved process: he verifies each gate video before the next)
- G1 PRESS solo: v9 (envelope 0.8) eval + hold + MEASURED center-strike -> video. [pending v9]
- G2 LEVER solo: lever5 breaker pull-down (rest 1.05 -> 0.15, latch) -> video; bar = real arm motion, ~50deg throw. [training]
- G3 WALK->PRESS continuous (no-teleport handoff). [after G1]
- G4 WALK->PICK continuous (approved pick, walked into). [tomorrow]
- G5 WALK->LEVER continuous. [after G2]
- G6 ONE-TAKE: walk->table pick->carry->panel press->lever pull, following cam. [after all]
- G7 SOP-driven wrapper (NL->SOP->plan->verified exec) + site/GitHub final video.
- RULES: root-cause before retrain; visual judgments go to Jatin with the specific question.
- OPEN (Jatin): carry the part to the panel vs place back? (rec: carry)
