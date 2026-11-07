# Brain

The language side of Sequent: ticket → SOP → plan.

| Path | What |
|---|---|
| `benchmark/` | The 1,012-SOP retrieval benchmark: corpus, retriever training and evaluation, reproduction scripts, results. Start with `benchmark/REPRO_REPORT.md`. |
| `src/planner/infer_plan.py` | The planner the robot uses (`plan_from_sop`): one skill per SOP step, `walk_to` inserted when the location changes, arguments mapped to the simulator's objects. |
| `src/data/` | Skill schema (`schemas.py`) and the original 100-SOP course corpus (`sop_examples.json`), used only as a fallback. |
| `src/retrieval/` | Retrieval code from the original course project (100-SOP corpus). |
| `tests/` | Planner and schema tests: `python -m pytest -q tests` from this folder. |

At runtime, `../brain_bridge.py` ranks the 1,012 SOPs with the fine-tuned retriever if `benchmark/artifacts/retriever_st` exists, and otherwise falls back to the 100-SOP corpus with an untuned model. To train the retriever:

```bash
cd benchmark
python -m src.retrieval.train_eval_sentence_encoder
```

This folder started as a course project (100 SOPs, a BERT dual-encoder and a Flan-T5 planner). That benchmark turned out to be too easy (TF-IDF scored about 0.96 R@1), which is why `benchmark/` exists. The Flan-T5 planner never produced valid output because T5's tokenizer cannot write `{` or `}`; see `benchmark/REPRO_REPORT.md`.
