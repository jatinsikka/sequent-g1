# Sequent

A simulated Unitree G1 humanoid that takes a plant-floor incident ticket written in plain language, retrieves the matching Standard Operating Procedure out of 1,012, turns its steps into robot skills, and executes them in one continuous MuJoCo simulation. Every step is checked against the physics state, and steps that fail are reported as failed.

<p align="center">
  <img src="docs/media/ticket_demo.gif" alt="Ticket demo: incident card, SOP retrieval, walk, RL button press, verified verdict" width="640">
  <br><sub>Ticket demo (<code>ticket_demo.py</code>): ticket → retrieved SOP → walk → RL button press → verified verdict.</sub>
</p>

<p align="center">
  <a href="https://jatinsikka.github.io/sequent-g1/">Project page</a> ·
  <a href="https://sequentrobotics.com">Website</a> ·
  <a href="docs/PAPER.md">Paper</a> ·
  <a href="brain/benchmark/REPRO_REPORT.md">Retrieval reproduction report</a> ·
  <a href="TRAINING_LOG.md">Training log</a>
</p>

## How it works

ticket → **retriever** (fine-tuned transformer, learned) → SOP → **planner** (rules: one skill per SOP step) → execution on the G1 → **per-step verification** from simulator state → verdict

<p align="center">
  <img src="docs/figures/fig_pipeline.png" alt="Sequent pipeline: ticket, bi-encoder retrieval, planner, verifying executor, body skills" width="720">
</p>

## SOP retrieval (the learned part of the brain)

A `all-MiniLM-L6-v2` bi-encoder (22.7M parameters) fine-tuned with a contrastive loss (MultipleNegativesRankingLoss, 3 epochs, batch 32, about 2 minutes on a laptop) on 1,859 pairs of (operator incident → correct SOP). The benchmark is synthetic: 1,012 SOPs across 46 equipment families (22 look-alike procedures per machine), with incidents written in vague, symptom-first operator language.

**Held-out test: 269 incidents, ranked against all 1,012 SOPs.** Fine-tuned rows are mean ± std over 5 seeds.

| Method | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF | 0.301 | 0.535 | 0.408 |
| BM25 | 0.357 | 0.569 | 0.461 |
| MiniLM, untuned | 0.409 | 0.677 | 0.529 |
| bge-small-en-v1.5, untuned | 0.465 | 0.717 | 0.578 |
| **MiniLM, fine-tuned (ours)** | **0.577 ± 0.006** | **0.856 ± 0.006** | **0.695 ± 0.005** |
| ours + off-the-shelf cross-encoder rerank | 0.452 ± 0.006 | 0.843 ± 0.003 | 0.611 ± 0.004 |

**Generalization.** Fine-tuning helps by a similar margin on procedures and machines never seen in training. Each split has its own test set, so compare rows within a column, not across columns.

| R@1 | SOPs held out (459 incidents) | 5 machine families held out (257 incidents) |
|---|---|---|
| BM25 | 0.447 | 0.311 |
| MiniLM, untuned | 0.468 | 0.420 |
| **MiniLM, fine-tuned** (3 seeds) | **0.656 ± 0.000** | **0.607 ± 0.004** |

**How clearly the operator writes matters most.** Same index, three kinds of ticket:

| R@1 | Vague (test set) | Same tickets, machine named | Clear, informed operator (92 tickets) |
|---|---|---|---|
| BM25 | 0.357 | 0.398 | 0.837 |
| **MiniLM, fine-tuned** (3 seeds) | **0.577 ± 0.008** | **0.632 ± 0.007** | **0.920 ± 0.006** |

Clear tickets are easy for every method; the fine-tuned model's advantage is on vague language. The 92 informed tickets were hand-written by the author from each SOP's title and condition, so they share some of its vocabulary.

Notes:
- The reranker lowers R@1, so it is not used.
- About 70% of the model's misses pick an SOP for a different machine. With the machine named, most remaining misses are same-machine look-alikes.
- The model always returns its closest SOP; it has no "none of these" option yet (see Next steps).
- Everything above is reproducible from `brain/benchmark/` (corpus, training, evaluation, CSVs).

## Planner

The SOPs are written at robot-skill granularity (`press_button estop_button`), so planning is a deterministic mapping: one skill per written step, in order, with `walk_to` inserted where the location changes. Arguments are mapped onto the simulator's objects (four colored buttons, a fixed tool set). An earlier learned planner (Flan-T5 + LoRA, JSON output) was dropped; its setup could not work (T5's tokenizer has no `{` or `}`), so it is not reported as a result.

## Skills

**Walking** uses the AMO whole-body controller (Ze et al., RSS 2025), frozen, with a yaw correction because AMO can't turn in place and drifts about 18 degrees while walking.

**Pick** is a controller, not RL: IK moves the gripper to the box, and the grip latches only when both finger pads measure contact force. 11 of 12 lifts held in validation on the fixed-base rig. RL grasping did work on an earlier fixed-base rig (v5.5: 55% verified lift-and-hold, see `TRAINING_LOG.md`), but on the gripper-equipped humanoid used for the demos the RL reward variants we tried scored 0% verified; their reported "lifts" were the box popping out of a crushing pinch (`docs/MANIPULATION_STORY.md`).

<p align="center">
  <img src="docs/media/pick.gif" alt="IK pick with force-gated latch (1.5x speed)" width="480">
</p>

**Button press** is RL (PPO) with a curriculum. It reaches about 19 cm from the rest pose and presses 29 to 40 mm, deterministic across 8 of 8 starts.

<p align="center">
  <img src="docs/media/press.gif" alt="Curriculum RL policy reaches from rest and presses the button" width="480">
</p>

**Lever** is RL and partial: its best policy completed the throw in 1 of 2 deterministic evaluation episodes, with jerky motion. The cause is a per-episode reach offset the policy can't observe; the fix is designed but not trained.

**Full chain:** walk to the table, pick the box, carry it to the control panel with a 180 degree turn, place it, press the button, return to rest, all in one take.

<p align="center">
  <img src="docs/media/onetake.gif" alt="One-take chain: walk, pick, carry, place, press (3x speed)" width="640">
</p>

## Final demo

One ticket, one continuous simulation. SOP-1013 was written for this demo and added to the corpus; the retriever ranks it first (score 0.897). Because the ticket and SOP were written together, this demonstrates the pipeline end to end, not retrieval accuracy (TF-IDF also ranks it first). Best of three takes (the lever row is the best of 4 stance variants):

| Step | Skill | Measured outcome | Verdict |
|---|---|---|---|
| 1 | pick | box gripped, lifted, held | PASS |
| 2 | carry / place | carried to the panel, placed upright | PASS |
| 3 | press_button | 33.8 mm press, held | PASS |
| 4 | lever | reached 0.443 rad (needs under 0.25 rad) | **FAIL** |
| 5 | notify | operator sent the verdict card | PASS |

Final verdict: "PARTIALLY RESOLVED". In the weakest take, place, press and lever all failed and the verifier reported each one.

## What is released

| Released | Not released |
|---|---|
| Retrieval benchmark, training and evaluation code, results (`brain/benchmark/`) | RL checkpoints for the press and lever policies |
| G1 + gripper model sources, environments, reward functions, training scripts | Compiled scenes (`*.mjb`; rebuild with `build_unified_model.py`, needs `mujoco_menagerie`) |
| Frozen AMO walking policy (`amo_jit.pt`, from Ze et al.) and the grasp v5.5 checkpoint | Training run logs |

The demo scripts check for the unreleased files and exit with a message saying which ones are missing.

## Quickstart

```bash
pip install -r requirements.txt   # Python 3.10

# Brain: reproduce the retrieval results (CPU or Apple MPS, ~2 min per training run)
cd brain/benchmark
python -m src.retrieval.train_eval_sentence_encoder          # train + evaluate; saves artifacts/retriever_st
python repro/repro_retrieval.py --split official --baselines --seeds 0 1 2 3 4
python brain_query.py "Hydraulic press ram is moving slower than usual" 5
cd ../..

# Ticket -> SOP -> plan through the bridge (uses the fine-tuned model once trained)
python -c "import brain_bridge as b; p,s,sc=b.incident_to_plan('Machine A pressure is low'); print(s['title'], sc); print(p)"

# Robot demos (need the unreleased RL checkpoints)
python g6_onetake.py
python final_ticket_demo.py
```

For parallel RL training, set `OMP_NUM_THREADS=1` per worker or throughput drops from about 2,600 to 18 steps/s.

## Repository layout

| Path | What |
|---|---|
| `brain/benchmark/` | 1,012-SOP corpus, retriever training/eval, reproduction scripts and result CSVs |
| `brain_bridge.py` | Ticket → SOP → typed plan |
| `final_ticket_demo.py` | Final demo: ticket to five-skill run |
| `g6_onetake.py` | Walk, pick, carry, place, press chain |
| `verifier.py`, `executor.py` | Per-step checks and the executor |
| `ik_pick.py`, `pick_table.py` | Pick controller |
| `train_button.py`, `train_lever.py`, `train_lever_bc.py` | RL training |
| `unified_env.py`, `build_unified_model.py` | G1 with the Robotiq gripper |
| `docs/PAPER.md` | Full write-up |
| `docs/notes/` | Working notes from development |

## Next steps

- Ask when unsure: refuse or ask a clarifying question ("which machine?") when the top match is weak or two candidates are nearly tied, instead of always acting on the closest SOP.
- Use the machine ID that a real ticket carries to filter candidates.
- Hard-negative training against same-machine look-alike SOPs.
- An LLM baseline, and a learned planner for procedures written in ordinary prose.
- Move the body to Isaac Lab; sim-to-real.

## Credits

Built by [Jatin Sikka](https://jatinsikka.me). The AMO controller is by Ze et al. (RSS 2025). Apache 2.0.
