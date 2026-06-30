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
- _(further entries appended if anything changes)_
