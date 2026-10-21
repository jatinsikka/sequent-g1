# Sequent

A simulated Unitree G1 humanoid that reads a plant-floor incident ticket, finds the matching Standard Operating Procedure (out of 1,012), turns it into a plan, and carries it out in one continuous MuJoCo simulation. Each step is checked against the physics state, and failed steps are reported as failed.

<p align="center">
  <img src="docs/media/ticket_demo.gif" alt="Ticket to robot: incident card, SOP retrieval, walk, RL button press, verified verdict" width="640">
</p>

<p align="center">
  <a href="https://sequentrobotics.com">Project site</a> ·
  <a href="docs/PAPER.md">Paper</a> ·
  <a href="TRAINING_LOG.md">Training log</a>
</p>

## How it works

ticket → SOP retrieval → plan (one skill per SOP step) → execution on the G1 → per-step verification → verdict

<p align="center">
  <img src="docs/figures/fig_pipeline.png" alt="Sequent pipeline: ticket, bi-encoder retrieval, faithful planner, verifying executor, body skills" width="720">
</p>

In the final demo, the retriever ranks the correct SOP first (score 0.897) and the robot runs the plan in a single simulation:

| Step | Skill | Measured outcome | Verdict |
|---|---|---|---|
| 1 | pick | box gripped, lifted, held | PASS |
| 2 | carry / place | carried to the panel, placed upright | PASS |
| 3 | press_button | 33.8 mm press, held | PASS |
| 4 | lever | reached 0.443 rad (needs under 0.25 rad) | **FAIL** |
| 5 | notify | operator sent the verdict card | PASS |

Final verdict: "PARTIALLY RESOLVED".

## Skills

**Walking** uses the AMO whole-body controller (Ze et al., RSS 2025), frozen, with a yaw correction added because AMO can't turn in place and drifts about 18 degrees while walking.

**Pick** is a controller, not RL: IK moves the gripper to the box, and the grip only latches when both finger pads measure contact force. 11 of 12 lifts held in validation. RL scored 0% across about ten reward setups.

<p align="center">
  <img src="docs/media/pick.gif" alt="IK pick with force-gated latch (1.5x speed)" width="480">
</p>

**Button press** is RL (PPO) with a curriculum. It reaches about 19 cm from the rest pose and succeeds 8/8, pressing 29 to 40 mm.

<p align="center">
  <img src="docs/media/press.gif" alt="Curriculum RL policy reaches from rest and presses the button" width="480">
</p>

**Lever** is RL and works about half the time. The cause is a per-episode reach offset the policy can't observe; the fix is designed but not implemented.

**Full chain:** walk to the table, pick the box, carry it 2.8 m with a 180 degree turn, place it, press the button, return to rest, all in one take.

<p align="center">
  <img src="docs/media/onetake.gif" alt="One-take chain: walk, pick, carry, place, press (3x speed)" width="640">
</p>

## SOP retrieval

A fine-tuned MiniLM bi-encoder (22.7M parameters) trained on 1,859 incident descriptions. Results on 269 held-out incidents against all 1,012 SOPs:

| Method | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF baseline | 0.301 | 0.535 | 0.408 |
| **Fine-tuned bi-encoder** | **0.584** | **0.851** | **0.697** |
| + cross-encoder reranker | 0.442 | 0.833 | 0.604 |

The reranker made results worse, so it isn't used. The planner is rule-based (one skill per SOP step) because a fine-tuned Flan-T5 planner produced 0 valid plans out of 20.

## Lessons

- Every reward hack (parking near the button, pumping presses, a box that flew out of the gripper and counted as a lift) was found by watching rollouts, not from metrics. Details are in the [paper](docs/PAPER.md), section 5.8.
- Check the action space before tuning rewards. The arm was limited to 0.5 rad per joint when the tasks needed up to 1.1 rad, so far-start attempts could never succeed.

## Repository layout

| Path | What |
|---|---|
| `final_ticket_demo.py` | Final demo: ticket to five-skill run |
| `g6_onetake.py` | Walk, pick, carry, place, press chain |
| `run_task.py` | Run any incident text end to end |
| `brain/` | SOP retrieval and planning |
| `verifier.py`, `executor.py` | Per-step checks and the executor |
| `ik_pick.py`, `pick_table.py` | Pick controller |
| `train_button.py`, `train_lever.py`, `train_lever_bc.py` | RL training |
| `unified_env.py`, `build_unified_model.py` | G1 with the Robotiq gripper |
| `docs/PAPER.md` | Full writeup |

## Quickstart

```bash
pip install -r requirements.txt

python run_task.py --command "Machine A pressure is low"
python g6_onetake.py          # renders _g6_onetake.mp4
python final_ticket_demo.py   # renders _final_ticket.mp4
```

Python 3.10, MuJoCo 3.2.3, Stable-Baselines3 2.1.0. For parallel training, set `OMP_NUM_THREADS=1` per worker or throughput drops from about 2,600 to 18 steps/s.

## Credits

Built by [Jatin Sikka](https://sequentrobotics.com). The AMO controller is by Ze et al. (RSS 2025). Apache 2.0.
