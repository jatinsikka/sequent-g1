# 90-Second Demo Video — Script & Shot List

Rule from the project brief: **nothing gets built unless it appears here or in README_PROJECT1.md.**
Narration is plain English, business framing. Target length 90 s (±5). Record narration last, over final footage.

| Time | Shot | Narration |
|---|---|---|
| 0:00–0:08 | Black screen → terminal. A human types: `pick up the screwdriver from the table and place it on the shelf`. Cut to MuJoCo: G1 standing in the factory scene. | "You shouldn't have to program a robot. You should be able to tell it what to do." |
| 0:08–0:20 | Overlay panel: the LLM planner's output appears as a checklist — `walk_to(table) → grasp(screwdriver) → carry_to(shelf) → place(screwdriver)` — each with a small "pre / post" badge. | "A language model breaks the instruction into steps. But here's what makes this different: every step comes with conditions the robot must *prove* — before it starts, and after it claims it's done." |
| 0:20–0:38 | Execution begins. `walk_to`: robot walks; its precondition badge flips green on screen. `grasp`: reach-envelope check flips green, robot grasps, postcondition "object lifted ✓" flips green. Picture-in-picture: live condition readouts (distance, contact, height). | "Watch the checks. The robot won't reach for an object it can't reach. And it doesn't *assume* the grasp worked — it measures it: contact made, object lifted." |
| 0:38–0:56 | **The money shot — failure caught.** A run where the grasp slips (or object is moved out of reach mid-run). The postcondition flips **red**, the robot stops, on-screen report: `FAILED grasp: no contact latch after 2 attempts → halting, operator notified`. | "Today's state-of-the-art policies fail one time in five — silently. Ours fails too. The difference is: it *knows*. It stops, says exactly what went wrong, and never reports a job it didn't finish." |
| 0:56–1:14 | Clean full run, slightly sped up: walk → grasp → carry → place on shelf. Final postcondition flips green; terminal prints `TASK COMPLETE — all 4 postconditions verified`. | "When it does succeed, you don't have to take its word for it. Every step is verified against the physics — an audit trail from instruction to completion." |
| 1:14–1:30 | Architecture diagram (from README) for 3 s → GitHub repo page → end card: project name, name, URL. | "Instruction in, verified work out. That's what it takes to put a humanoid on a factory floor. Code, training logs, and every failure we hit along the way — all public. Link below." |

## Production checklist

- [ ] Factory-ish scene dressing (table, shelf, screwdriver) — visual only, no new physics
- [ ] On-screen condition badges (render overlay from verifier state — this is a *feature the video requires*, so it gets built)
- [ ] One genuine caught-failure rollout saved (not staged if possible; staged-but-honest fallback: move object mid-run)
- [ ] One clean rollout, 1× and 2× speed exports
- [ ] Screen-record terminal + planner JSON
- [ ] Narration: record after picture lock; 140 words ≈ 85 s at calm pace
- [ ] Export: 1080p MP4 + GIF (first 8 s) for README top

## What this script commits us to build (and nothing more)

1. Planner producing that 4-skill JSON. 2. Verifier with live-readable condition state. 3. The 4 skills at "completes sometimes, verifiably" quality. 4. Overlay rendering of check state. 5. A failure-report path that halts cleanly.
