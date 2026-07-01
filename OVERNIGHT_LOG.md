# Overnight Run Log — 2026-06-30 (Jatin asleep ~10h)

Running cost tally start: ~$15–18 (pre-overnight VM time). VM restarted for overnight ~now.

## Timeline
- **[start]** Master plan written (MASTER_PLAN.md). Lever set to push-up config (only reachable/drivable direction — tested down/throw, both out of reach). Gauges fixed. VM booting.
- **[~1h] lever-v0 throughput fix.** First launch (32 envs, no thread cap) ran at **18 steps/s** — thread oversubscription (32 AMO workers × 64 torch threads thrashing the 64 cores). 65k steps in 60 min = unusable, ~1h VM wasted (~$3). **Fix: `OMP_NUM_THREADS=1` per worker → ~2600 steps/s** (2M in ~13 min). Relaunched lever-v0. LESSON: the AMO env MUST cap threads per worker for parallel envs.
- **[~1.5h] lever-v0 DONE (works).** 2M steps, turn rate 0% — BUT behavior eval (rendered + measured) shows the robot reliably drives the lever to **0.44 rad (25°), upright, every episode** (beat the 0.34 open-loop probe). The 0% was only because target=0.6 (>reachable 0.44). **The lever is a working RL skill** (3rd, with press_button). Reachable throw ceiling ≈ 0.44 rad (arm reach/torque limit at this lever).
- **[~1.5h] lever-v1 launched** at target=0.4 (reachable, within the 0.1 tol of the 0.44 ceiling) → should give a clean turn-rate→success curve for the showcase. 1.5M steps, OMP=1, ~17 min.
- **[~2h] lever-v1 DONE + VERIFIED.** Target 0.4: drives the lever to ~0.35–0.4 rad (~20°), upright, 5/5 consistent, success registers (v0's 0% was only the too-high 0.6 target). **LEVER = verified RL skill** (3rd, with press_button). Reachable throw ceiling ~0.44 rad. Rollout: `_lever_v0_eval.mp4` (v1 policy).
- **[~2h] VM DEALLOCATED.** No high-value training left (lever done; push was finicky/parked; press_button done). Shifting to non-training progress: website draft (pivot story + skills) + demo artifacts. Will restart VM only if a clear training win appears.
- **[~2h] Website CONTENT drafted** → `docs/MANIPULATION_STORY.md` (honest pivot + 3 skills + measured numbers, de-fluffed tone). Left as content for Jatin to integrate into `../sequent-site/` with his taste (site copy/design = his call; not editing his HTML autonomously).

---
## HANDOFF (for when Jatin wakes)
**Done tonight:** lever = verified 3rd RL skill (operates panel lever, upright, 5/5). Throughput bug fixed (OMP=1 for AMO envs). Committed + pushed to `skills/manipulation`. Website content drafted. VM deallocated.
**Cost:** ~$22 total (incl. ~$3 wasted on the thread-thrash run before the OMP fix).
**Needs your direction (didn't do autonomously — too risky/taste/outward):**
1. **End-to-end SOP demo** — cross-rig: grasp uses the fixed-base table robot; press/lever use the AMO humanoid at the panel; walk uses AMO locomotion. Stitching them into one run needs an architectural call (one robot/scene, or stitched clips). Your call.
2. **Website deploy** — content ready in `docs/MANIPULATION_STORY.md`; styling + `vercel --prod` is yours.
3. **More training?** — lever/press/grasp/walk done; push was finicky (parked); no obvious high-value run left. Say if you want a specific one (bigger lever throw, push retry, new skill).
**Why I'm not burning more tokens autonomously:** primary goal hit; remaining work needs your input. Doing light hourly check-ins instead of speculative big tasks (per your token-efficiency rule).
- **[~3h] Check-in:** nothing new (no message, no running/failed tasks, VM deallocated). Primary goal done; remaining items await Jatin. Re-sleeping. Cost ~$22.
- **[~4h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin. Re-sleeping. Cost ~$22.
- **[~5h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin. Re-sleeping. Cost ~$22.
- **[~6h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin. Re-sleeping. Cost ~$22.
- **[~7h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin. Re-sleeping. Cost ~$22.
- **[~8h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin. Re-sleeping. Cost ~$22.
- **[~9h] Check-in:** nothing new. Idle, VM deallocated, awaiting Jatin (near 10h mark). Re-sleeping. Cost ~$22.
- **[~10h] Overnight window COMPLETE.** Ending the autonomous loop (no more wakeups — nothing training, VM off, further work awaits Jatin's direction). Resuming on Jatin's next message.

## UNIFIED SINGLE-ROBOT BUILD (Jatin's direction, morning)
- **De-risk PASSED (verified):** walking humanoid + Robotiq gripper does a real held pick (10.7cm), AMO balances, stays upright through reach-down recoil (~4.5cm/8°). `build_unified_model.py`, `unified_env.py`, `g1_amo_gripper.mjb`.
- **Envs adapted:** ButtonPressEnv/LeverPressEnv take `unified=True` → gripper contact point, closed gripper as pusher, AMO leg-DOF reads preserved. Lever arm-contortion fixed (IK forward-pitch bias, not cross-body roll).
- **Retrain results:** unified BUTTON (blue) → 0% press; eval shows gripper reaches **4.5cm from the button but never presses** (upright). Undershoot — gripper's reach_bias/geometry ≠ rubber hand; recoil budget makes the last cm hard. Lever retraining (its reach_bias WAS fixed) = the diagnostic: if it completes → button needs same reach_bias fix; if it also undershoots → switch press/lever to IK-assisted reach + short press (like the pick), keep RL for the contact.
- Cost ~$26.

- **Unified RL VERDICT: press + lever both fail on the gripper-humanoid.** Button: gripper reaches 4.5cm, 0mm press. Lever: 0.0° turn (even with the fixed reach_bias). Contact dead-zone — reach-from-distance never contacts, contact-gated reward gives no signal, recoil budget compounds it. Robot stays upright (AMO fine). PAUSED for Jatin's call: (A) IK-assisted press/lever (like the working pick — reliable, gets one clean unified demo, but press/lever become controllers; RL results bp-v5/lever-v1 stand separately) [RECOMMENDED]; (B) reset-in-contact RL (start gripper touching the target, learn press/turn from contact — keeps RL, more tuning, uncertain). VM deallocated, ~$27.

- **Jatin's call: keep RL in the demo** (train 1-2 skills via RL; invest in reward engineering). Path = reset-in-contact RL (IK reach + RL contact).
- **ROOT-CAUSE BREAKTHROUGH:** the unified press/lever 0% was NOT (only) a reach dead-zone — it was a **collision-mask bug**: Robotiq pads on mask 1, panel buttons/lever on mask 2 → (1&2)=0 → **the gripper passed straight through the buttons with zero force.** Fixed (pads → mask 3). Plus reset-in-contact (IK seats gripper on button at reset; smoke: scripted push presses to 2.42cm >2cm threshold, holds, AMO upright) + contact-mode reward (depth+hold, no reach terms). Retraining unified BUTTON (blue) with all three now.
- Cost ~$28.

- **✅ UNIFIED RL BUTTON PRESS WORKS (verified).** Reset-in-contact + collision fix + contact-mode reward: presses blue button to **29mm (>20mm threshold), holds 18 steps, upright (pelvis 0.73), arm not crossing** — render `_unified_press.mp4`. A real RL skill on the SINGLE robot (was 0% / phasing-through an hour ago). Applying the same recipe to the lever now (agent).

## FINAL OVERNIGHT SUMMARY
**Delivered:** Lever = **verified 3rd RL skill** (operates the control-panel lever, upright, 5/5) — on the panel via the proven AMO/press_button rig, after diagnosing that the pusher arm can only drive it up (down/throw tested out of reach) and fixing the gauges. Fixed a real AMO throughput bug (OMP_NUM_THREADS=1: 18→2600 steps/s). All committed + pushed to `skills/manipulation`. Website content drafted (`docs/MANIPULATION_STORY.md`).
**Skill set now:** grasp/pick (IK controller, 11/12), press_button (RL, bp-v5), **lever (RL, new)**, walk_to (controller). Push parked (finicky).
**Cost:** ~$22 (incl. ~$3 on the pre-OMP-fix slow run). VM deallocated.
**Awaits Jatin:** (1) end-to-end SOP demo — cross-rig architectural call; (2) website styling + deploy; (3) any specific further training.


- **UNIFIED DEMO stitched** (`_unified_demo.mp4`): one robot doing pick(IK)+press(RL)+lever(RL). Both RL skills verified. Concatenated clips (same robot) — NOT yet one continuous run; walk not yet included. Continuous integration (walk->pick->press->lever, one scene) = next polish. Cost ~$30, VM deallocated.
