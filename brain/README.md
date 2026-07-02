# SOP-Guided Robotic Task Planning

An end-to-end system that retrieves Standard Operating Procedures (SOPs) from natural language incident descriptions and generates executable robot plans. Built as the "brain" of a larger autonomous robotics project — this repo handles perception-to-plan, while the physical execution runs on a [Unitree G1 humanoid via MuJoCo](https://github.com/jatinsikka/mujoco_G1_AMO).

> **Status:** WIRED INTO THE RUNTIME. The live path (used by `sop_demo.py` / `run_task.py` via
> `../brain_bridge.py`) is: **hybrid retrieval** (semantic `all-MiniLM-L6-v2` interleaved with
> lexical TF-IDF over the 100-SOP library) → an LLM picks the SOP from the merged top-5 →
> **faithful planner** (`plan_from_sop`: one skill per written SOP step, in order, repeats
> preserved, `walk_to` navigation inferred). The BERT dual-encoder / Flan-T5-LoRA stack below is
> the original course-project design — kept for reference, not what runs.

## How It Works (current runtime path)

```
Incident ──► Hybrid Retriever ─────► Top-5 SOPs ──► LLM planner ─────► JSON Plan ──► Verified Execution
             (MiniLM ⊕ TF-IDF)                      (plan_from_sop      (7 skills)    (../executor.py)
                                                     as fallback)
```

## Results (historical — course project, NOT reproduced in this repo)

> ⚠️ The numbers below are from the Fall-2025 course project report. The trained checkpoints and
> eval harness behind them are **not in this repo**, and we have not reproduced them here — they
> are kept for provenance only and are not claims of this system. (Policy: we only publish
> numbers we measured ourselves.)

| Model | Recall@1 | Recall@5 | MRR | Latency |
|-------|----------|----------|-----|---------|
| Trained Dual-Encoder BERT | 0.96 | 0.99 | 0.98 | 58ms |
| TF-IDF (baseline) | 0.96 | 0.99 | 0.98 | 1.6ms |
| Pretrained BERT (zero-shot) | 0.22 | 0.50 | 0.31 | 60ms |
| Ollama RAG (zero-shot) | 0.95 | 1.00 | 0.97 | 2297ms |

| Planner Metric | Heuristic Baseline | Flan-T5 + LoRA |
|----------------|-------------------|----------------|
| Plan F1 | 0.45 | 0.78 |
| Execution Success | 0.60 | 0.95 |
| Valid JSON Rate | 0.80 | 0.98 |

## Dataset

100 synthetic manufacturing SOPs across three categories:
- **Machine Control** (SOP-001 to SOP-020): Pressure warnings, temperature alerts, startup/shutdown
- **Table Manipulation** (SOP-021 to SOP-060): Object handling, tool management, part organization
- **Complex Workflows** (SOP-061 to SOP-100): Multi-step emergency procedures, conditional logic

Plus 100 incident examples with ground-truth SOP labels for evaluation.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train retriever (dual-encoder BERT) — ~2-5 min
python src/retrieval/build_dual_encoder.py

# Train planner (Flan-T5 + LoRA) — ~1-2 min
python src/planner/train_planner_lora.py

# Build FAISS index
python src/cli/demo.py build-index

# Run full pipeline
python src/cli/demo.py plan --q "Machine A pressure is low"

# Run with execution
python src/cli/demo.py exec --q "Machine A pressure is low"

# Evaluate on all 100 incidents
python src/eval/evaluate_all.py
```

## Project Structure

```
src/
├── retrieval/          # Dual-encoder BERT, FAISS indexing, Ollama RAG baseline
├── planner/            # Flan-T5 + LoRA training and inference
├── pipeline/           # End-to-end plan pipeline, skill definitions
├── eval/               # Evaluation scripts, metrics, baseline comparisons
├── data/               # 100 SOPs + 100 incidents (JSON)
├── cli/                # Typer CLI (build-index, retrieve, plan, exec)
├── env/                # MuJoCo skill API (stub)
└── ner/                # Token labeling utilities
config/                 # Training hyperparameters (YAML)
artifacts/              # Trained model checkpoints and FAISS index (gitignored)
tests/                  # Unit tests
```

## Architecture Details

| Component | Model | Details |
|-----------|-------|---------|
| Retriever | BERT-base-uncased (110M params) | Dual-encoder, InfoNCE loss, 50 epochs, batch size 8 |
| Planner | Flan-T5-base (250M params) | LoRA r=32, alpha=64, 10 epochs, batch size 4 |
| Reranker | DeBERTa-v3-base | Cross-encoder for precision reranking of top-K |
| Index | FAISS | Flat index over 768-dim SOP embeddings |

Configuration is driven by `config/retriever_config.yaml` and `config/planner_config.yaml`. CLI arguments override config values.

## Related Repo

This project is the planning/DL side of a larger autonomous robotics system:

- **This repo (`sop_planner`)** — SOP retrieval + plan generation (the "brain")
- **[`mujoco_G1_AMO`](https://github.com/jatinsikka/mujoco_G1_AMO)** — Unitree G1 locomotion + manipulation in MuJoCo (the "body")

The planner outputs structured JSON plans that map to robot primitives executable by the MuJoCo environment.

## Paper

See [`DL_Project_report.pdf`](DL_Project_report.pdf) for the full writeup, including ablation studies and error analysis.
