# Manipulation — the honest arc (website content draft, for Jatin to integrate)

_Tone: de-fluffed/research, measured numbers only. Drop into the site where it fits your taste._

## The pivot: we learned what RL is *for*

We spent real effort trying to learn a **grasp-and-lift with RL** — and it was a dead end. Across ~10 configurations (v8b→v13) the deterministic success rate was **0**. The "lifts" the metric reported were the box **popping out of the grip and going ballistic** — at peak "lift" the gripper wasn't even holding it (box 7–9cm away). Jatin caught it by watching; the metric had fooled us.

The lesson is the same one top labs already encode: **a parallel-jaw grasp is not an RL problem.** Production manipulation decomposes it — perception predicts *where* to grip, a motion planner gets the gripper there, and the grip itself is a **force-controlled close**, not a learned policy. RL is reserved for the genuinely contact-rich, dynamics-heavy skills.

So we put each tool where it belongs.

## Skills (what actually works, measured)

**Grasp / pick — a controller, not RL.** `ik_pick.py`: reach → settle → clean close → lift, using damped-least-squares IK. **11/12 held lifts** across random seeds; grips centered to **1.35cm**; the box stays held (gripping=true at peak, box <1.75cm from the gripper) and rides the hand up ~13cm. Reliable, no training lottery.

**Press button — RL (contact-rich).** Full-humanoid rig: AMO drives the legs for balance, RL drives the arm to press a panel button. **bp-v5: 10/10 deterministic, button held 79 steps, base upright.** The hard part was base recoil — solved.

**Lever — RL (contact-rich).** The press-button's twin on the same control panel: the robot reaches and **throws a hinged lever to a target angle**, upright, AMO balancing the legs. Drives the lever to **~0.4 rad consistently (5/5 episodes)**. This is where RL earns its place — maintaining contact while turning a hinge is awkward to hand-write, easy to reward.

**Walk-to — controller** (AMO + PID). Locomotion to the workstation.

## The honest negatives (we publish these too)

- **RL grasp+lift: 0% success**, every config. Parked — it's a controller's job.
- **Non-prehensile push** (slide a free box with no grip): finicky in this setup — the side-pushing arm reaches the box but a free box against table friction is a weak, easily-camped reward signal. Parked; the lever covers the contact-rich niche better.

## The shape of the result

Three working skills — **a controller for the grasp, RL for the two contact-rich panel skills (press, lever)** — plus walk-to. That's not "we RL'd everything"; it's "we know what each tool is for," which is the stronger research story.

_Measured-numbers policy: every number here is one we measured ourselves. Nothing aspirational._
