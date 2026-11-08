# Running the demos

Tested on an Apple Silicon Mac, no GPU.

## Setup

```bash
git clone https://github.com/jatinsikka/sequent-g1.git && cd sequent-g1
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt

# train the retriever once (~2 min); also downloads all-MiniLM-L6-v2
cd brain/benchmark && ../../.venv/bin/python -m src.retrieval.train_eval_sentence_encoder && cd ../..
```

Run each demo once before presenting so the models are cached.

## What runs

| Demo | Runs? | Notes |
|---|---|---|
| `demo/brain_walkthrough.py` | yes | 1,012 SOPs, fine-tuned retriever; `--library course` for the 100-SOP set |
| `demo/brain_eval.py` | yes | the paper's retrieval table, live, ~30 s; `--informed` for the 92 title-derived tickets |
| `demo/brain_eval.py --hard` | yes | 10 hard tickets, 100-SOP library |
| `demo/body_walk.py`, `demo/drive_g1.py` | needs MuJoCo | AMO walking; last run August 2026 |
| `pick_table.py` | needs MuJoCo + `mujoco_menagerie` | IK pick |
| `g6_onetake.py`, `final_ticket_demo.py`, `ticket_demo.py` | no | need the unreleased RL checkpoints |

Robot videos: the [project page](https://jatinsikka.github.io/sequent-g1/), files in `docs/static/videos/`.

## Presentation (~12 min)

### 1. Walkthrough (6 min)

```bash
.venv/bin/python demo/brain_walkthrough.py          # Enter advances each stage
```

| Stage | Shows |
|---|---|
| 1 Ticket | the operator's text |
| 2 SOP library | 1,012 SOPs, 46 machines |
| 3 TF-IDF retrieval | top 5, matched words |
| 4 MiniLM retrieval | top 5 from the fine-tuned model |
| 5 Candidates | both rankers interleaved |
| 6 Result | rank of the correct SOP |
| 7 Selected SOP | its steps |
| 8 Plan | one skill per step |
| 9 Inserted navigation | `walk_to` added where the location changes |
| 10 Verifier checks | postcondition per skill |
| 11 Output | the plan JSON |

Default ticket: INC-00056 (correct at rank 1). Other options:

```bash
demo/brain_walkthrough.py --incident INC-00026   # correct SOP at rank 3; top pick is another machine's SOP
demo/brain_walkthrough.py --ticket "..."         # a ticket from the audience
demo/brain_walkthrough.py --list                 # all 269 test tickets
demo/brain_walkthrough.py --auto --pace 1.0      # hands-free
```

Tickets already run on the 1,012-SOP library:

| `--ticket "..."` | TF-IDF picks | fine-tuned picks | shows |
|---|---|---|---|
| `The hydraulic press is making a grinding noise and the ram is moving slower than usual.` | Ram Descent Velocity Below Specified Rate | same | both correct |
| `The machine is overheating.` | Machine Lockout-Tagout (CNC mill) | Fan Motor Overtemperature Alarm (fume extractor); top-2 gap 0.006 | no machine named |
| `Screwdriving robot torque went over the max limit during fastening, might be cross-threaded.` | Torque Reading Above Maximum Ceiling (screwdriving robot), correct | Torque Exceeds Maximum Ceiling (torque station) | wrong machine; why TF-IDF stays in the candidates |
| `Smoke coming out of the electrical cabinet near line 3, smells like burning plastic.` | Electrical Cabinet Cooling Filter | Emergency Stop for Fire or Smoke (air compressor) | right emergency, wrong machine |
| `My cat knocked over a plant.` | an e-stop SOP | Sudden Catastrophic Pressure Collapse | out-of-scope ticket still returns an SOP |

### 2. Numbers (3 min)

```bash
.venv/bin/python demo/brain_eval.py
.venv/bin/python demo/brain_eval.py --informed
```

| ranker | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF | 0.301 | 0.535 | 0.408 |
| BM25 | 0.357 | 0.569 | 0.461 |
| MiniLM, untuned | 0.409 | 0.677 | 0.529 |
| MiniLM, fine-tuned (one model) | 0.569 | 0.855 | 0.689 |

Paper: mean of 5 seeds, 0.577 ± 0.006. On the 92 informed tickets the fine-tuned model scores 0.90 with the correct SOP in the top 5 every time.

`--library course` shows the 100-SOP set at ~1.0 for every ranker (saturated).

### 3. Robot (3 min)

Play the videos: the one-take chain, then the finale and its "partially resolved" verdict.

With MuJoCo installed (`pip install mujoco==3.2.3 glfw mujoco-python-viewer imageio imageio-ffmpeg`):

```bash
.venv/bin/python demo/drive_g1.py                  # viewer; W walk, A/D turn, ESC quit
.venv/bin/python demo/body_walk.py --seconds 6     # headless; distance, height, heading error (~19°)
```

IK pick (no RL checkpoint needed):

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # 2.2 GB
.venv/bin/python build_unified_model.py --pick
.venv/bin/python demo/run_on_cpu.py pick_table.py                            # ~4 min
```

The August run printed `box lift: 4.6 cm ... BOX HELD: False`: the latch engaged but the held-lift check failed.

## Running without an NVIDIA GPU

`amo_jit.pt` was traced on CUDA and contains a literal `device=cuda:0`, so on a Mac `play_amo.py` fails with `Could not run 'aten::empty_strided' ... 'CUDA' backend`. `patch_jit_to_cpu()` in `demo/body_walk.py` rewrites those constants; `demo/run_on_cpu.py` applies it and runs any script unchanged:

```bash
.venv/bin/python demo/run_on_cpu.py pick_table.py
```

It does nothing on a CUDA machine.
