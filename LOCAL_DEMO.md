# Running Sequent locally, and showing it to people

How to get the brain running on a laptop, what runs and what does not, and a script for walking someone through it. Tested on an Apple Silicon Mac with no GPU.

---

## 1. Setup, once

```bash
git clone https://github.com/jatinsikka/sequent-g1.git
cd sequent-g1
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Train the retriever once (about 2 minutes on a laptop). This also downloads all-MiniLM-L6-v2.
cd brain/benchmark && ../../.venv/bin/python -m src.retrieval.train_eval_sentence_encoder && cd ../..
```

That writes `brain/benchmark/artifacts/retriever_st`. The demo scripts use it automatically. Run one demo once before presenting so the models are cached.

---

## 2. What runs

| Demo | Runs locally? | Notes |
|---|---|---|
| `demo/brain_walkthrough.py` | yes | 1,012 SOPs and the fine-tuned retriever by default; `--library course` for the old 100-SOP set |
| `demo/brain_eval.py` | yes | reproduces the paper's retrieval table live (TF-IDF, BM25, untuned, fine-tuned) |
| `demo/brain_eval.py --informed` | yes | the 92 clear tickets |
| `demo/brain_eval.py --hard` | yes | ten plant-floor complaints against the 100-SOP course library |
| `demo/body_walk.py`, `demo/drive_g1.py` | needs MuJoCo installed | AMO walking; tested August 2026 on this Mac |
| `pick_table.py` | needs MuJoCo + `mujoco_menagerie` (2.2 GB) | see §3, Act 3 |
| `g6_onetake.py`, `final_ticket_demo.py`, `ticket_demo.py` | no | need the press/lever RL checkpoints, which are not released; they exit with a message saying so |

For the robot itself, show the videos: the [project page](https://jatinsikka.github.io/sequent-g1/) has the one-take chain, pick, press, lever, the ticket-to-robot run, and the full finale. The files are in `docs/static/videos/`.

---

## 3. The presentation: about 12 minutes, three acts

### Act 1: the brain, step by step (6 min)

```bash
.venv/bin/python demo/brain_walkthrough.py
```

Eleven stages, each waiting for Enter, so you can talk over it. Everything on screen is computed live.

| Stage | On screen | What to say |
|---|---|---|
| 1 | The operator's ticket | "No SOP number, no equipment code. That's the problem." |
| 2 | One SOP and the library | "1,012 procedures across 46 machines, about 22 look-alikes each." |
| 3 | TF-IDF top 5 and the words it matched | "Word overlap. Works until the operator uses different words." |
| 4 | The 384-d vector, then the fine-tuned top 5 | "This model was trained on how operators describe problems." |
| 5 | The hybrid, labelled by source | "Semantic first; keyword matches stay in the candidate set." |
| 6 | Gold label revealed | If it missed, it says so. |
| 7 | The chosen SOP | "The robot can only run a procedure from this approved library." |
| 8 | Each step → one skill | "One skill per written step. Nothing dropped, nothing invented." |
| 9 | Inferred `walk_to` steps | The planner adds navigation the SOP left out, and tags it `inferred`. |
| 10 | What the verifier checks per skill | "A step isn't done because the policy says so." |
| 11 | The JSON handed to the robot | "That's the whole brain/body contract." |

The default ticket (INC-00056, an AGV dock voltage collapse) is a correct rank-1 retrieval. Other options:

```bash
demo/brain_walkthrough.py --list                    # the 269 held-out tickets
demo/brain_walkthrough.py --incident INC-00026      # a miss: right problem, wrong machine
demo/brain_walkthrough.py --ticket "..."            # take one from the audience
demo/brain_walkthrough.py --auto --pace 1.0         # hands-free, for screen recording
```

**Take a ticket from the room.** It is the most convincing thing you can do, and stage 6 reports a miss as a miss. These were run beforehand on the 1,012-SOP library so you know what will appear:

| `--ticket "..."` | TF-IDF picks | fine-tuned picks | good for |
|---|---|---|---|
| `The hydraulic press is making a grinding noise and the ram is moving slower than usual.` | Ram Descent Velocity Below Specified Rate | same | both right; a clear ticket is easy |
| `The machine is overheating.` | Machine Lockout-Tagout (CNC mill) | Fan Motor Overtemperature Alarm (fume extractor), gap to #2 only 0.006 | it can't know which machine; the case for asking "which machine?" |
| `Screwdriving robot torque went over the max limit during fastening, might be cross-threaded.` | **Torque Reading Above Maximum Ceiling (screwdriving robot)** | Torque Exceeds Maximum Ceiling (torque station) | keyword wins, the model picks the wrong machine; why the hybrid keeps keyword candidates |
| `Smoke coming out of the electrical cabinet near line 3, smells like burning plastic.` | Electrical Cabinet Cooling Filter | Emergency Stop for Fire or Smoke (air compressor) | right kind of emergency, wrong machine |
| `My cat knocked over a plant.` | an e-stop SOP | Sudden Catastrophic Pressure Collapse | it always answers; the case for refusing below a confidence cutoff |

The last four show the limits. Showing them makes the next-step argument for you: ask when unsure, and filter by the machine ID a real ticket carries.

### Act 2: the numbers, measured live (3 min)

```bash
.venv/bin/python demo/brain_eval.py
```

Runs the 269 held-out tickets against all 1,012 SOPs and prints the paper's table live, in about 30 seconds:

| ranker | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF | 0.301 | 0.535 | 0.408 |
| BM25 | 0.357 | 0.569 | 0.461 |
| MiniLM, untuned | 0.409 | 0.677 | 0.529 |
| MiniLM, fine-tuned | 0.569 | 0.855 | 0.689 |

The fine-tuned row is one trained model; the paper reports the mean of 5 seeds (0.577 ± 0.006), and the script says so. Then:

```bash
.venv/bin/python demo/brain_eval.py --informed      # 92 clear tickets: fine-tuned 0.90, top 5 every time
```

"When the operator writes clearly, every method does well. Training matters on the vague tickets."

Optional: `demo/brain_eval.py --library course` shows the old 100-SOP set scoring about 1.0 for every ranker, with a warning that it is saturated. It explains why the 1,012-SOP benchmark exists.

### Act 3: the body (3 min)

Play the videos from the project page: the one-take chain, then the finale with its "partially resolved" verdict card. Say what the verifier caught.

If MuJoCo is installed (`pip install mujoco==3.2.3 glfw mujoco-python-viewer imageio imageio-ffmpeg`), you can also run it live:

```bash
.venv/bin/python demo/drive_g1.py          # the G1 in the viewer, AMO running; W walks, A/D turn, ESC quits
.venv/bin/python demo/body_walk.py --seconds 6   # headless; prints distance, height, heading error
```

Hold W and watch it drift sideways: about 19 degrees off a straight line, which is why the project wraps AMO in a yaw servo. `body_walk.py` measures that number from `mjData`.

**Optional: the IK pick.** It needs no RL checkpoint, only the Robotiq gripper meshes from Google DeepMind's model collection:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # 2.2 GB
.venv/bin/python build_unified_model.py --pick
.venv/bin/python demo/run_on_cpu.py pick_table.py     # renders _unified_pick_table.mp4, about 4 minutes
```

On the August run it printed `box lift: 4.6 cm ... BOX HELD: False`: the latch engaged but the held-lift check failed. That is the verifier doing its job on an untuned run. Do a few takes beforehand if you want a clean one on screen.

---

## 4. Running the robot code without an NVIDIA GPU

`amo_jit.pt` was traced on a CUDA machine, and the trace contains a literal `device=cuda:0`. `map_location` moves parameters but not constants, so on a Mac `python play_amo.py` fails with:

```
NotImplementedError: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend
```

`demo/body_walk.py` has `patch_jit_to_cpu()`, which rewrites those constants after loading. `demo/drive_g1.py` uses it. For any other script, `demo/run_on_cpu.py` patches `torch.jit.load` and runs the script unchanged:

```bash
.venv/bin/python demo/run_on_cpu.py pick_table.py
.venv/bin/python demo/run_on_cpu.py play_amo.py
```

It does nothing on a CUDA machine, so it is safe everywhere.

---

## 5. Files

```
demo/brain_walkthrough.py    the 11-stage walkthrough (--library plant | course)
demo/brain_eval.py           live retrieval table; --informed, --hard, --misses, --disagree
demo/body_walk.py            headless AMO walk with measurements, optional mp4
demo/drive_g1.py             the live viewer, CPU-patched
demo/run_on_cpu.py           run any repo script without an NVIDIA GPU
LOCAL_DEMO.md                this file
```
