# Brain reproduction report

Reproduced November 2026 on an Apple M1 Pro (CPU/MPS, Python 3.10, sentence-transformers 3.0.1, transformers 4.44.2, torch 2.3.1). Every number below comes from a CSV in `repro/results/`, and each script that produced it is listed. The RL skill numbers (press, pick, lever) were not re-run: their checkpoints are not released, and they are documented by the recorded videos.

## Claimed vs reproduced

| Claim (earlier write-up) | Reproduced | Verdict | Source |
|---|---|---|---|
| Corpus: 1,012 SOPs, 46 families, 2,338 incidents, 1,859/210/269 split | Exact (46 × 22 SOPs) | reproduced | `audit_data.py`, `audit_family.py` → `data_audit.csv` |
| TF-IDF R@1/R@5/MRR 0.301/0.535/0.408 | 0.301 / 0.535 / 0.408 | exact | `repro_retrieval.py --baselines` |
| Fine-tuned MiniLM 0.584/0.851/0.697 | 0.577 ± 0.006 / 0.856 ± 0.006 / 0.695 ± 0.005 (5 seeds) | within seed noise (the original script has no seed; one seed gives 0.584) | `retrieval_runs.csv` |
| Reranker lowers R@1 to 0.442 | 0.452 ± 0.006 | direction reproduced; old row was never re-measured on its paired base | `retrieval_runs.csv` |
| Finale: SOP-1013 rank 1, score 0.897 | rank 1, 0.898 | reproduced, but TF-IDF also ranks it first → pipeline demo, not retrieval evidence | `brain_query.py` |
| "Fine-tuned model roughly doubles keyword search" | true vs TF-IDF only; BM25 0.357, untuned MiniLM 0.409, bge-small 0.465 | reframed: +0.17 over the same model untuned | `retrieval_runs.csv` |
| From-scratch bert-base lost to TF-IDF (0.312 vs 0.396) | measured on a different 418-SOP corpus | removed (not comparable to the 1,012-SOP table) | `../../docs/PAPER.md` history |
| Flan-T5 LoRA planner: 0/20 valid plans | 0/20 JSON-parseable, reproduced | removed as a finding: setup bug (T5's tokenizer has no `{` `}` `<`; trained on the 100-SOP file with keyword-template targets) | `planner_t5_eval.py` → `planner_t5_config_as_shipped.csv` |
| Runtime "interleaves the trained semantic ranking with TF-IDF" | runtime used an untuned MiniLM over the old 100-SOP file | was untrue; `brain_bridge.py` now uses the fine-tuned model over 1,012 SOPs when trained | `brain_bridge.py` |

## New results

**Generalization** (each split has its own test set; compare within a column only)

| R@1 | Held-out SOPs (459) | Held-out families (257) |
|---|---|---|
| TF-IDF | 0.392 | 0.296 |
| BM25 | 0.447 | 0.311 |
| MiniLM untuned | 0.468 | 0.420 |
| bge-small untuned | 0.523 | 0.490 |
| MiniLM fine-tuned (3 seeds) | 0.656 ± 0.000 | 0.607 ± 0.004 |

**Ticket clarity** (`tiered_eval.py`, fine-tuned = 3 seeds)

| R@1 | Vague (269) | Machine named (269) | Informed operator (92) |
|---|---|---|---|
| TF-IDF | 0.301 | 0.312 | 0.750 |
| BM25 | 0.357 | 0.398 | 0.837 |
| MiniLM untuned | 0.409 | 0.416 | 0.837 |
| bge-small untuned | 0.465 | 0.439 | 0.891 |
| MiniLM fine-tuned | 0.577 ± 0.008 | 0.632 ± 0.007 | 0.920 ± 0.006 |

The 92 informed tickets (`repro/data/informed_tickets.json`) were written by the author, 2 per family, from SOPs sampled with a fixed seed, using only each SOP's title and condition and without looking at model outputs.

**Errors.** About 70% of errors on the test tickets select an SOP for a different machine. Naming the machine raises correct-machine top-1 from 0.71 to 0.85, so the remaining misses are same-machine look-alikes. Of the misses, only 1–3% happen to have the same skill sequence as the gold SOP, so a miss means executing a different procedure (`planner_end_to_end.csv`).

**Selective retrieval (not part of the system).** Cutoffs are chosen on val and reported on test, over 3 seeds (`abstain_eval.py` → `abstain_operating.csv`). Acting only when the top-1 vs top-2 margin clears the cutoff handles 41% ± 3% of vague tickets at 86% ± 3% precision and escalates the rest; acting always gives 58%.

**Epochs.** 1 epoch: R@1 0.539 ± 0.013 (3 seeds); 3 epochs (default): 0.577. The 5-epoch run was not completed.

**Benchmark planner bug, fixed.** `src/planner/infer_plan.py` (`deterministic_plan`) silently dropped 411 of 6,853 steps (106 of them stop, e-stop or isolation actions) from 5 families written in prose. Fixed with `_normalize_step` plus regression tests (`tests/test_step_normalizer.py`); coverage is now 6,851 of 6,853 (the remaining 2 are conditionals). The robot's own planner (`brain/src/planner/infer_plan.py`, `plan_from_sop`) never drops steps. It misread every snake_case step (`press_button X` → `notify`) because of a regex word-boundary bug, and that is fixed too (6,442/6,442).

## Data audit notes
- Every test SOP also has at least one training incident (by construction in `data_gen/merge_corpus.py`), so the official test is closed-set; the held-out splits above address this.
- No near-duplicate train/test incidents (highest test-to-train TF-IDF cosine 0.31; mean of each test incident's nearest 0.13).
- Test incidents share on average 32% of their gold SOP's title words; 17.5% share none. "Avoids title words" is partly true.
- All data comes from one generator; there is no real-plant evaluation.

## Reproduce

```bash
cd brain/benchmark
python repro/audit_data.py
python repro/repro_retrieval.py --split official --baselines --seeds 0 1 2 3 4 --rerank
python repro/repro_retrieval.py --split unseen_sop --baselines --seeds 0 1 2
python repro/repro_retrieval.py --split family_out --baselines --seeds 0 1 2
python repro/abstain_eval.py        # trains 3 seeds, saves artifacts/retriever_ft_s*
python repro/tiered_eval.py         # uses those models
python repro/planner_eval.py
python -m pytest -q tests
```
