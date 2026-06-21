# The Sequent Pipeline — language → SOP → verified action

The thesis in one line: **an operator describes an incident in plain English; the robot
retrieves the right Standard Operating Procedure, turns it into a chain of robot skills, and
executes it step-by-step — verifying each physical step against measured physics, and
honestly reporting any step it cannot do.**

```
  natural-language incident
        │   (1) RETRIEVAL   — semantic match
        ▼
  Standard Operating Procedure (1 of 100)
        │   (2) PLANNER     — faithful 1:1 mapping
        ▼
  executable skill chain  (walk_to, press_button, read_sensor, …)
        │   (3) EXECUTOR    — run + physics-verify each step
        ▼
  honest report:  ✅ verified  ·  ⚑ not verified (halt)
```

---

## 1 · Natural language → SOP retrieval  (`brain/src/retrieval/`)

The corpus is **100 real manufacturing SOPs** (`brain/src/data/sop_examples.json`), each with
a `title`, a `condition` (when it applies), and ordered `steps`.

- **Encoder:** a semantic sentence encoder — **sentence-transformers `all-MiniLM-L6-v2`**
  (384-dim). Each SOP is embedded as `title. condition. Steps: …`; the incident query is
  embedded the same way. This matches by *meaning*, so "smoke coming from the unit" finds the
  **Smoke Detection Response** SOP and "replace a blown fuse" finds **Fuse Replacement** —
  even with no shared keywords. (Offline fallback: TF-IDF, if transformers is unavailable.)
- **Search:** cosine similarity over the normalized embeddings (FAISS if present, else numpy).
- **Re-rank:** a small keyword *nudge* breaks ties without overriding meaning (a stronger
  lexical boost is used only in the TF-IDF fallback).
- **API:** `retrieve_topk(incident_text, k) -> [{sop_id, score, text}]`.
- **Quality:** 20/20 self-retrieval (condition→SOP); paraphrased incidents resolve to the
  correct or a closely-related SOP. Where several SOPs are equally relevant (e.g. multiple
  vibration procedures) it returns a plausible one — there is no single ground truth.

## 2 · SOP → skill chain — the chain of command  (`brain/src/planner/infer_plan.py`)

`plan_from_sop(sop_steps)` is the **faithful planner**: it maps **each SOP step to exactly one
robot skill, in order, with repeats preserved.** The plan therefore *is* the SOP expressed in
robot skills — the executor walks the SOP line-by-line.

- `_classify_step(step)` scores each skill by **word-boundary keyword evidence** (so "Read
  **press**ure_sensor" maps to `read_sensor`, not `press_button`) and picks the strongest;
  a step with no action evidence falls back to `notify`.
- `_extract_args_for_skill` pulls the concrete arguments (which button colour, which sensor,
  which object, how many seconds).
- Every plan step keeps its **source `sop_step`** → the execution is fully traceable/auditable.

> This replaced a legacy planner that *deduplicated* skills and dropped repeated steps (a SOP's
> second sensor read or second button press vanished). `plan_from_sop` maps 1:1 on all 100 SOPs.

Example — *"the machine is overheating, shut it down"* → **SOP-002 Emergency Shutdown**:

| # | SOP step | skill | args |
|---|----------|-------|------|
| 1 | Walk to machine | `walk_to` | target=machine |
| 2 | Press red_button immediately | `press_button` | button=red_button |
| 3 | Wait 5 seconds for shutdown | `wait` | seconds=5 |
| 4 | Read temperature_sensor | `read_sensor` | sensor=temperature_sensor |
| 5 | Notify technician of shutdown event | `notify` | level=tech |

## 3 · The skills — the registry  (`brain/src/data/skill_registry.py`)

`SKILL_REGISTRY` is the **single source of truth**: every skill's description, args,
pre/postconditions, runner, and **honest verification status**. `schemas.SKILLS` and the
executor whitelist derive from it.

| skill | kind | status | what it does |
|-------|------|--------|--------------|
| `walk_to` | locomotion | **verified** | PID + AMO navigation to a location; verified arrived-upright |
| `press_button` | manipulation | reach-limited | reach + hold a button; AMO recoil blocks a flush reach — verifier flags it |
| `pick` | manipulation | reach-limited | grasp + lift an object; same reach limit — verifier flags miss/knock-off |
| `place` | manipulation | operational | place a held object (depends on `pick`) |
| `read_sensor` | operational | operational | read pressure / temperature / vibration / light |
| `wait` | operational | operational | pause a fixed duration |
| `notify` | operational | operational | escalate to a technician / manager |

Statuses are honest (validated 2026-06-21): **locomotion is solid; tabletop manipulation is
reach-limited by the frozen AMO base controller** (it recoils ~18 cm when the arm extends).
The verifier reports those as *not verified* rather than faking success — which is the point.

## 4 · Verifying execution  (`executor.py`)

`VerifyingExecutor.run(plan)` runs each step **in order** and halts on the first step that is
neither verified nor an operational stub:

- `walk_to` → `skills_locomotion.run_walk_to`, **verified** iff arrived within threshold AND upright.
- `pick`/`press_button` → real RL policies, **verified** iff the physics postcondition holds
  (object lifted + held / button held past threshold, upright) — measured from `mjData`,
  never the policy's own claim. Retries, then reports the failing postcondition.
- operational skills (`read_sensor`, `wait`, `notify`, `place`) → logged, marked `stubbed`.

A step counts only if the simulator agrees it happened. No silent success.

---

## Files

| stage | file |
|-------|------|
| SOP corpus | `brain/src/data/sop_examples.json` (100 SOPs) |
| schemas | `brain/src/data/schemas.py` (SOPEntry, Plan, …) |
| skill registry | `brain/src/data/skill_registry.py` (single source of truth) |
| retrieval | `brain/src/retrieval/{infer_retrieve,index_utils}.py` |
| planner | `brain/src/planner/infer_plan.py` (`plan_from_sop`) |
| executor | `executor.py` (`VerifyingExecutor`) |
| end-to-end demo | `sop_demo.py` |

Run the demo: `PYTHONPATH=brain python sop_demo.py "the machine is overheating, shut it down"`
