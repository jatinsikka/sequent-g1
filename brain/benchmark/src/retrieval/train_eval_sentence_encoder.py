"""Train a proper sentence-embedding bi-encoder and evaluate on the held-out test.

Replaces the weak recipe (bert-base CLS token, tiny batch) with the standard one:
  - a real sentence-encoder base (all-MiniLM-L6-v2),
  - mean pooling,
  - MultipleNegativesRankingLoss (every other SOP in the batch is a negative),
trained on the (incident -> gold SOP) pairs from the TRAIN split only.

Then it evaluates, on the held-out TEST split over the full SOP index:
  TF-IDF (lexical)  vs  this bi-encoder  vs  bi-encoder + cross-encoder rerank.

Usage:
  python -m src.retrieval.train_eval_sentence_encoder \
      --sops src/data/plant_sops.json \
      --train src/data/plant_incidents_train.json \
      --test src/data/plant_incidents_test.json \
      --out artifacts/retriever_st --epochs 3 --batch 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.schemas import SOPEntry, IncidentEntry, load_json  # noqa: E402


def synth(s: SOPEntry) -> str:
    return f"{s.title}. {s.condition}. Steps: " + " ; ".join(s.steps)


def metrics(ranked: List[List[str]], golds: List[str]) -> dict:
    r1 = np.mean([g == r[0] for r, g in zip(ranked, golds)])
    r5 = np.mean([g in r[:5] for r, g in zip(ranked, golds)])
    rr = [1.0 / (r.index(g) + 1) if g in r else 0.0 for r, g in zip(ranked, golds)]
    return {"R@1": float(r1), "R@5": float(r5), "MRR": float(np.mean(rr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sops", default="src/data/plant_sops.json")
    ap.add_argument("--train", default="src/data/plant_incidents_train.json")
    ap.add_argument("--test", default="src/data/plant_incidents_test.json")
    ap.add_argument("--out", default="artifacts/retriever_st")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--topk_rerank", type=int, default=10)
    ap.add_argument("--no_train", action="store_true", help="skip training; load existing model from --out and eval only")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    sops = load_json(str(root / args.sops), SOPEntry, key="sop_examples")
    train_inc = load_json(str(root / args.train), IncidentEntry, key="incident_examples")
    test_inc = load_json(str(root / args.test), IncidentEntry, key="incident_examples")

    sop_ids = [s.sop_id for s in sops]
    sop_texts = [synth(s) for s in sops]
    id_to_text = {s.sop_id: t for s, t in zip(sops, sop_texts)}
    print(f"SOPs: {len(sops)} | train: {len(train_inc)} | test: {len(test_inc)}")

    from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder
    from torch.utils.data import DataLoader

    # ---- train (or load existing for eval-only) ----
    out = root / args.out
    if args.no_train:
        print(f"loading existing sentence-encoder <- {out} (eval only, no training)")
        model = SentenceTransformer(str(out))
    else:
        model = SentenceTransformer(args.base)
        examples = [
            InputExample(texts=[inc.text, id_to_text[inc.labels["sop_id"]]])
            for inc in train_inc
            if inc.labels and inc.labels.get("sop_id") in id_to_text
        ]
        print(f"training pairs: {len(examples)}")
        loader = DataLoader(examples, shuffle=True, batch_size=args.batch)
        loss = losses.MultipleNegativesRankingLoss(model)
        warmup = int(len(loader) * args.epochs * 0.1)
        model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=warmup, show_progress_bar=False)
        model.save(str(out))
        print(f"saved sentence-encoder -> {out}")

    # ---- eval ----
    q_texts = [i.text for i in test_inc]
    golds = [i.labels.get("sop_id", "") for i in test_inc]
    results = {}

    # TF-IDF baseline
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2))
    S = vec.fit_transform(sop_texts)
    sims = (vec.transform(q_texts) @ S.T).toarray()
    tf_order = np.argsort(-sims, axis=1)
    results["TF-IDF (lexical)"] = metrics([[sop_ids[j] for j in tf_order[i]] for i in range(len(test_inc))], golds)

    # Sentence bi-encoder
    S_emb = model.encode(sop_texts, normalize_embeddings=True, show_progress_bar=False)
    Q_emb = model.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
    bi = Q_emb @ S_emb.T
    bi_order = np.argsort(-bi, axis=1)
    results["Bi-encoder (all-MiniLM + MNR)"] = metrics(
        [[sop_ids[j] for j in bi_order[i]] for i in range(len(test_inc))], golds
    )

    # Bi-encoder + cross-encoder rerank
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    k = args.topk_rerank
    re_ranked = []
    for i in range(len(test_inc)):
        cand = bi_order[i][:k]
        scores = ce.predict([(q_texts[i], sop_texts[j]) for j in cand])
        head = [sop_ids[cand[j]] for j in np.argsort(-scores)]
        tail = [sop_ids[j] for j in bi_order[i][k:]]
        re_ranked.append(head + tail)
    results[f"Bi + cross-encoder rerank (top{k})"] = metrics(re_ranked, golds)

    print("\n=== Held-out retrieval results ===")
    print(f"{'method':36} {'R@1':>7} {'R@5':>7} {'MRR':>7}")
    for name, m in results.items():
        print(f"{name:36} {m['R@1']:7.3f} {m['R@5']:7.3f} {m['MRR']:7.3f}")


if __name__ == "__main__":
    main()
