"""Experiment 7: selective retrieval. Act only when confident, otherwise escalate to a human.

For each seed: train the fine-tuned MiniLM (same recipe as repro_retrieval.py), score the val and
test incidents against the 1,012-SOP index, and save the score matrices (results/scores_s<seed>.npz)
so later steps (LLM baseline, figures) reuse them without retraining.

Confidence signals: top-1 cosine ("top1") and top-1 minus top-2 cosine ("margin").
The threshold is chosen on VAL to reach a target precision, then reported on TEST.

Outputs
  results/risk_coverage.csv     coverage/precision curve on test for each seed and signal
  results/abstain_operating.csv val-chosen threshold -> test coverage and precision
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from common import RESULTS, ROOT, load, synth

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--targets", type=float, nargs="+", default=[0.80, 0.90, 0.95])
args = ap.parse_args()

sops, split = load(exclude_finale=True)
ids = np.array([s.sop_id for s in sops])
texts = [synth(s) for s in sops]
id2text = dict(zip(ids, texts))


def scores_for(seed):
    f = RESULTS / f"scores_s{seed}.npz"
    if f.exists():
        d = np.load(f)
        return d["val"], d["test"]
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ex = [InputExample(texts=[i.text, id2text[i.labels["sop_id"]]]) for i in split["train"]]
    dl = DataLoader(ex, shuffle=True, batch_size=32)
    m.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(m))], epochs=3,
          warmup_steps=int(len(dl) * 3 * 0.1), show_progress_bar=False)
    m.save(str(ROOT / "artifacts" / f"retriever_ft_s{seed}"))
    enc = lambda t: m.encode(t, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    S = enc(texts)
    val, test = enc([i.text for i in split["val"]]) @ S.T, enc([i.text for i in split["test"]]) @ S.T
    np.savez(f, val=val, test=test, sop_ids=ids)
    return val, test


def conf_and_correct(sc, inc):
    o = np.argsort(-sc, axis=1)
    top1 = sc[np.arange(len(sc)), o[:, 0]]
    margin = top1 - sc[np.arange(len(sc)), o[:, 1]]
    correct = ids[o[:, 0]] == np.array([i.labels["sop_id"] for i in inc])
    return {"top1": top1, "margin": margin}, correct


def curve(conf, correct):
    o = np.argsort(-conf)
    c = correct[o].astype(float)
    n = np.arange(1, len(c) + 1)
    return n / len(c), np.cumsum(c) / n, conf[o]  # coverage, precision, threshold


rc, op = [], []
for seed in args.seeds:
    val, test = scores_for(seed)
    cv, okv = conf_and_correct(val, split["val"])
    ct, okt = conf_and_correct(test, split["test"])
    for sig in ("top1", "margin"):
        cov, prec, thr = curve(ct[sig], okt)
        rc += [dict(seed=seed, signal=sig, coverage=a, precision=b, threshold=t) for a, b, t in zip(cov, prec, thr)]
        vcov, vprec, vthr = curve(cv[sig], okv)
        for tgt in args.targets:
            ok = np.where(vprec >= tgt)[0]
            t = vthr[ok.max()] if len(ok) else np.inf  # lowest threshold that still meets the target on val
            act = ct[sig] >= t
            op.append(dict(seed=seed, signal=sig, target_precision=tgt, threshold=float(t),
                           test_coverage=float(act.mean()),
                           test_precision=float(okt[act].mean()) if act.any() else np.nan,
                           test_wrong_actions=int((act & ~okt).sum()), test_escalated=int((~act).sum()),
                           test_n=len(okt)))
    print(f"seed {seed}: R@1 val {okv.mean():.3f} test {okt.mean():.3f}")

pd.DataFrame(rc).to_csv(RESULTS / "risk_coverage.csv", index=False)
o = pd.DataFrame(op)
o.to_csv(RESULTS / "abstain_operating.csv", index=False)
print(o.groupby(["signal", "target_precision"])[["test_coverage", "test_precision", "test_wrong_actions"]]
      .agg(["mean", "std"]).round(3).to_string())
