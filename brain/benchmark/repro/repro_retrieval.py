"""Experiments 2-4: seeded retrieval reproduction, extra baselines, generalization splits.

Training recipe is identical to src/retrieval/train_eval_sentence_encoder.py
(all-MiniLM-L6-v2, MultipleNegativesRankingLoss, batch 32, 10% warmup, SOP text = synth()),
with an explicit seed added.

Splits
  official    : the shipped train/test files (every test gold SOP also has a train incident)
  unseen_sop  : 20% of SOPs held out entirely; test = all their incidents (golds never trained on)
  family_out  : 5 equipment families held out entirely (22 SOPs each)
Evaluation always ranks against the full 1,012-SOP index (or 1,013 with --with_finale).

Writes results/retrieval_runs.csv (one row per method/run) and
results/ranks_<split>_<method>_s<seed>.csv (gold rank per query).
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd
from common import RESULTS, ROOT, family_map, load, metrics, synth

ZS_MODELS = {
    "zs_minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "zs_bge_small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
}


def make_split(name, sops, split, seed_split=0):
    fam = family_map()
    if name == "official":
        return split["train"], split["test"]
    allinc = split["train"] + split["val"] + split["test"]
    rng = random.Random(seed_split)
    ids = sorted(s.sop_id for s in sops)
    if name == "unseen_sop":
        held = set(rng.sample(ids, k=round(0.2 * len(ids))))
    elif name == "family_out":
        fams = sorted({fam[i] for i in ids})
        hf = set(rng.sample(fams, k=5))
        held = {i for i in ids if fam[i] in hf}
        print("held-out families:", sorted(hf))
    else:
        raise ValueError(name)
    train = [i for i in split["train"] if i.labels["sop_id"] not in held]
    test = [i for i in allinc if i.labels["sop_id"] in held]
    return train, test


def ranks_from_scores(scores, sop_ids, golds):
    order = np.argsort(-scores, axis=1)
    ranked = [[sop_ids[j] for j in row] for row in order]
    gold_rank = [r.index(g) + 1 for r, g in zip(ranked, golds)]
    return ranked, gold_rank


def encode(model, texts, prefix=""):
    return model.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="official", choices=["official", "unseen_sop", "family_out"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--with_finale", action="store_true", help="include SOP-1013 in the index")
    ap.add_argument("--baselines", action="store_true", help="also run TF-IDF/BM25/zero-shot (deterministic)")
    ap.add_argument("--rerank", action="store_true", help="MS-MARCO cross-encoder top-10 over each fine-tuned run")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    sops, split = load(exclude_finale=not args.with_finale)
    train_inc, test_inc = make_split(args.split, sops, split)
    sop_ids = [s.sop_id for s in sops]
    sop_texts = [synth(s) for s in sops]
    id2text = dict(zip(sop_ids, sop_texts))
    q = [i.text for i in test_inc]
    golds = [i.labels["sop_id"] for i in test_inc]
    seen = {i.labels["sop_id"] for i in train_inc}
    print(f"split={args.split} index={len(sops)} train={len(train_inc)} test={len(test_inc)} "
          f"test-golds-seen-in-train={np.mean([g in seen for g in golds]):.2f}")

    rows = []

    def record(method, scores_or_ranked, seed=-1, extra=None, secs=None):
        if isinstance(scores_or_ranked, np.ndarray):
            ranked, gr = ranks_from_scores(scores_or_ranked, sop_ids, golds)
        else:
            ranked = scores_or_ranked
            gr = [r.index(g) + 1 for r, g in zip(ranked, golds)]
        m = metrics(ranked, golds)
        row = dict(split=args.split, index_size=len(sops), n_test=len(golds), method=method, seed=seed,
                   epochs=args.epochs if method.startswith("ft") else None, tag=args.tag, secs=secs, **m)
        rows.append(row)
        pd.DataFrame(dict(incident_id=[i.incident_id for i in test_inc], gold=golds,
                          top1=[r[0] for r in ranked], gold_rank=gr)).to_csv(
            RESULTS / f"ranks_{args.split}{'_fin' if args.with_finale else ''}_{method}_s{seed}{args.tag}.csv",
            index=False)
        print(f"  {method:22s} seed={seed:>2}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}")

    if args.baselines:
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2))  # identical to the original script
        S = vec.fit_transform(sop_texts)
        record("tfidf", (vec.transform(q) @ S.T).toarray())
        import re
        tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        bm = BM25Okapi([tok(t) for t in sop_texts])
        record("bm25", np.stack([bm.get_scores(tok(t)) for t in q]))
        for name, (hf, prefix) in ZS_MODELS.items():
            m = SentenceTransformer(hf)
            record(name, encode(m, q, prefix) @ encode(m, sop_texts).T)

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if args.rerank else None
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        t0 = time.time()
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ex = [InputExample(texts=[i.text, id2text[i.labels["sop_id"]]]) for i in train_inc]
        loader = DataLoader(ex, shuffle=True, batch_size=args.batch)
        loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
                  warmup_steps=int(len(loader) * args.epochs * 0.1), show_progress_bar=False)
        secs = round(time.time() - t0, 1)
        sc = encode(model, q) @ encode(model, sop_texts).T
        record("ft_minilm", sc, seed, secs=secs)
        if ce is not None:
            order = np.argsort(-sc, axis=1)
            rr = []
            for i in range(len(q)):
                cand = order[i][:10]
                s = ce.predict([(q[i], sop_texts[j]) for j in cand], show_progress_bar=False)
                rr.append([sop_ids[cand[j]] for j in np.argsort(-s)] + [sop_ids[j] for j in order[i][10:]])
            record("ft_minilm+rerank", rr, seed)

    out = RESULTS / "retrieval_runs.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
