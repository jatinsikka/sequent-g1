# Sequent — press_button training archive

Run-by-run log of teaching the G1 humanoid to press a control-panel button (button_red).
Each run: what changed · why · result (deterministic, physics-verified) · artifacts.
This is the source-of-truth for the website training-archive page. Newest at bottom.

Verifier bar: button displacement ≥ **2cm**, **held ≥25 consecutive control steps**, base upright (no fall). Judged on the DETERMINISTIC policy + measured physics, never the policy's own claim.

| Run | Change | Why | Deterministic result | Artifacts |
|-----|--------|-----|----------------------|-----------|
| bp-v0 | PPO, depth reward | first real run after fixing reachability (button was 0.48m out of reach) | **HOVER** — max 0.04cm, 0 press. Parks farming proximity reward. | — |
| bp-v1 | contact-gated proximity (kill hover income) | stop paying for hovering far back | **0.96cm** — mean commits partway, still short of 2cm. Best of the no-BC runs. | `bp-v0_v2_noBC/bpv1_nobc_policy.gif` |
| bp-v2 | + steep hold-bonus cliff at 2cm | pull the mean across the threshold | **REGRESSED** — 0.04cm hover, stochastic 8→5/10. Reward cliff destabilized value learning. | — |
| bp-v3 | **BC warm-start** (clone the scripted press) + PPO + smooth sustained-hold reward | bp-v0..v2 cold-started into hovering; clone a pressing policy so the mean STARTS pressing | **3.56cm deep, stochastic 10/10** — BC killed the hover. BUT held 13/25 → still a deep JAB, not a hold. VERIFIER FAIL. | `bp-v3_BC/{bcv3_policy.gif, bcv3_trace.png, bcv3_training.png}` |
| MODEL FIX | found the button was modeled wrong: whole body slid (housing+cap on one joint) AND the slide axis was inverted (pressed OUTWARD) | the "press" was the hand dragging a loose cylinder out, not pressing in | rebuilt: fixed housing + cap compressing INTO the machine (−Y). Verified: housing static, cap slides −Y. | — |
| DIAGNOSIS | decomposed the dynamic reach gap (arm tracking vs base recoil vs sag) | hand kept landing ~12-15cm off the cap despite the arm tracking the command perfectly | **root cause = base recoil**: pelvis recoils 8.9cm back + 4.8cm down when reaching (AMO balancing the CoM shift) — that carries the hand off the cap. Not arm weakness (5× stiffness changed nothing). | — |
| **BREAKTHROUGH** | (1) realistic **protruding mushroom e-stop** button (brings the contact surface to the robot) + (2) **closed-loop Jacobian-IK** press controller | a straight-on press to a flat panel 31cm away is dynamically infeasible (recoil eats the reach); a protruding button + IK servo that corrects for recoil each step solves it | **IK press: 2.83cm @ 8cm protrusion (held 87), 4.2cm @ 11cm (held 128) — base upright.** The press is SOLVED, and the IK controller is a real holding-press demo. | (committed to model + env) |
| bp-v4 | **BC from the IK holding-press demo** + PPO (hold reward), on the corrected+mushroom env (spawn −1.62, action_scale 2.5) | bp-v3's demo was a jab → it learned to jab; clone a HOLDING demo so it learns to hold | **5.3cm deep, held 22/25, 10/10 stochastic, upright — NEAR-PASS.** frac_held still climbing (0.22). | `bcv4_policy.gif, bcv4_trace.png, bcv4_training.png` |
| **bp-v5** | **resume bp-v4 + 4M more steps** (same env/reward) | bp-v4 held 22/25 with frac_held still rising → just train more | **✅ VERIFIER PASS — 5.27cm deep, held 79 steps (need 25), 10/10 stochastic, base upright.** frac_held climbed 0.22→0.31→0.42. press_button SOLVED + verified. | `bcv5_policy.gif, bcv5_trace.png, bcv5_training.png` |

## ✅ FULL THESIS DEMO WORKING (2026-06-20)
`sop_demo.py`: NL incident → SOP retrieval → plan → verified skill chain. Real run:
```
INCIDENT "the machine is overheating, shut it down now"
 → RETRIEVED SOP-002 Emergency Shutdown → plan → execute:
   1. walk_to(machine)   [VERIFIED] arrived upright
   2. press_button(red)  [VERIFIED] 5.3cm · held 79 · upright   ← the RL skill
   3. wait / 4. read_sensor / 5. notify  [done]
 → SOP COMPLETE — every physical step physics-verified ✅
```
Skills: walk_to (PID+AMO) + press_button (RL, this whole saga). Wired in `executor.py` (REAL_SKILLS) + `sop_demo.py`. Interactive story site live on Vercel. Trace saved: `sop_demo_trace.txt`.

## Headline learning so far: behavior cloning (the A/B)
- **Without BC** (bp-v1, best no-BC): deterministic policy **hovers / barely presses (0.96cm)** — the mean never experienced a press, so it parks near the button collecting shaping reward.
- **With BC** (bp-v3): deterministic policy **presses deep and reliably (3.56cm, 10/10 stochastic)** — cloning the scripted press first means the mean starts in a pressing regime, then RL refines.
- **Why BC helped:** it fixes the cold-start exploration failure. PPO from scratch on a thin-margin, near-reach-limit target finds the "park and farm shaping" local optimum; starting from a demonstration puts the policy in the basin of the real behavior.

## grasp smoothness retrain (fixing the "yank") — 2026-06-20
The grasp v5.5 works (deterministic ~33–55% verified) but YANKS the object (jerky, unnatural). Cause: the action-rate (smoothness) penalty in `env_wrapper.py` was mild (0.05). Resumed v5.5 with stronger smoothness:
| Run | smoothness coef | grasp success (12 ep det) | action-change (smoothness) | verdict |
|-----|-----|-----|-----|-----|
| v5.5 | 0.05 | 4/12 (~33%) | 0.21 (jerky/yank) | works but yanks |
| grasp-v6 | 0.15 (3×) | **0/12** | **0.075** (very smooth) | TOO SMOOTH → timid, stopped grasping (only 1.6cm lift). The code comment's warning, confirmed. |
| grasp-v7 | 0.10 (2×) | **0/14** | 0.084 (smooth) | STILL timid — even 2× kills the grasp. Trade-off is SHARP. |
| grasp-v8 | phase-gated (0.04 far / 0.30 near) | **0/16** (interrupted by VM auto-shutdown; partial) | 0.086 | also timid — phase-gating didn't rescue it either |
**CONCLUSION: the yank is COUPLED to v5.5's grasp strategy (a fast slap-grab).** Every smoothness increase (uniform 0.10/0.15, phase-gated) suppressed the slap AND the grasp together → 0 grasps, only ~1.6cm lift. You cannot smooth v5.5 by a reward tweak. A smooth grasp must be **re-learned from scratch/curriculum** with smoothness shaped in from the start — a multi-iteration effort (like the original v1→v5.6). **v5.5 stands as the grasp skill of record** (functional ~33–55%, yanks). `env_wrapper.py` reverted to 0.05. Smooth-grasp redesign = future work. Artifacts: `grasp_v55.gif` (yanks but grasps), `grasp_v6/v7/v8.gif` (smooth but timid).

## Open issues found via the verifier + visual review (the value of judging on physics, not claims)
1. **Jab, not hold** (bp-v3): presses then releases; held 13/25. Next: clone a *holding* demo + reward the hold-streak.
2. **Button model bug (FIXED 2026-06-19):** the whole button body slid (housing + cap on one joint) and the slide axis was inverted (+Y, outward toward robot) — so the "press" was the hand dragging a loose cylinder *outward*. Rebuilt: fixed housing + cap that compresses INTO the machine (−Y). Invalidates v0–v3 (wrong direction); retrain on corrected model.
3. **Unnatural / cramped stance:** no motion reference — the policy optimizes only press+balance, so the pose is whatever RL finds. Corrected button direction should force a natural forward press; a motion reference is the lever if we want true realism.
