"""Honest retrieval evaluation on the held-out test split.

Compares three retrievers over the SAME SOP index, on the SAME unseen test incidents:
  1. TF-IDF        - lexical baseline (no learning)
  2. Bi-encoder    - the trained dual-encoder (this run's artifact)
  3. Bi + reranker - bi-encoder top-k re-scored by a real cross-encoder

The point: on a corpus that is finally hard (within-family confusable SOPs,
operator-register queries with no title overlap), does the neural encoder beat
keywords, and does reranking add anything? Encoders are loaded ONCE.

Usage:
  python -m src.eval.eval_retrieval_plant \
      --encoder_dir artifacts/retriever_plant \
      --sops src/data/plant_sops.json \
      --test src/data/plant_incidents_test.json
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


def metrics(ranked_ids: List[List[str]], golds: List[str]) -> dict:
    r1 = np.mean([g == r[0] for r, g in zip(ranked_ids, golds)])
    r5 = np.mean([g in r[:5] for r, g in zip(ranked_ids, golds)])
    rr = []
    for r, g in zip(ranked_ids, golds):
        rr.append(1.0 / (r.index(g) + 1) if g in r else 0.0)
    return {"R@1": float(r1), "R@5": float(r5), "MRR": float(np.mean(rr))}


def embed(texts: List[str], model_dir: str, tok_dir: str, batch: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
            cls = model(**enc).last_hidden_state[:, 0, :]
            cls = cls / (cls.norm(dim=1, keepdim=True) + 1e-9)
            out.append(cls.cpu().numpy())
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_dir", default="artifacts/retriever_plant")
    ap.add_argument("--sops", default="src/data/plant_sops.json")
    ap.add_argument("--test", default="src/data/plant_incidents_test.json")
    ap.add_argument("--topk_rerank", type=int, default=10)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    enc = root / args.encoder_dir
    sops = load_json(str(root / args.sops), SOPEntry, key="sop_examples")
    incidents = load_json(str(root / args.test), IncidentEntry, key="incident_examples")

    sop_ids = [s.sop_id for s in sops]
    sop_texts = [synth(s) for s in sops]
    q_texts = [i.text for i in incidents]
    golds = [i.labels.get("sop_id", "") for i in incidents]
    print(f"SOPs: {len(sops)} | test incidents: {len(incidents)}")

    results = {}

    # 1. TF-IDF lexical baseline
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2))
    S = vec.fit_transform(sop_texts)
    Q = vec.transform(q_texts)
    sims = (Q @ S.T).toarray()
    order = np.argsort(-sims, axis=1)
    ranked = [[sop_ids[j] for j in order[i]] for i in range(len(incidents))]
    results["TF-IDF (lexical)"] = metrics(ranked, golds)

    # 2. Trained bi-encoder
    S_emb = embed(sop_texts, str(enc / "sop_encoder"), str(enc / "tokenizer"))
    Q_emb = embed(q_texts, str(enc / "incident_encoder"), str(enc / "tokenizer"))
    bi_sims = Q_emb @ S_emb.T
    bi_order = np.argsort(-bi_sims, axis=1)
    bi_ranked = [[sop_ids[j] for j in bi_order[i]] for i in range(len(incidents))]
    results["Bi-encoder (trained)"] = metrics(bi_ranked, golds)

    # 3. Bi-encoder topK re-ranked by a real cross-encoder
    try:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        k = args.topk_rerank
        re_ranked = []
        for i in range(len(incidents)):
            cand_idx = bi_order[i][:k]
            pairs = [(q_texts[i], sop_texts[j]) for j in cand_idx]
            scores = ce.predict(pairs)
            reordered = [sop_ids[cand_idx[j]] for j in np.argsort(-scores)]
            # keep the bi-encoder tail after the reranked head so R@5/MRR are defined
            tail = [sop_ids[j] for j in bi_order[i][k:]]
            re_ranked.append(reordered + tail)
        results[f"Bi + cross-encoder rerank (top{k})"] = metrics(re_ranked, golds)
    except Exception as e:
        print(f"[reranker skipped: {e}]")

    print("\n=== Held-out retrieval results ===")
    print(f"{'method':36} {'R@1':>7} {'R@5':>7} {'MRR':>7}")
    for name, m in results.items():
        print(f"{name:36} {m['R@1']:7.3f} {m['R@5']:7.3f} {m['MRR']:7.3f}")


if __name__ == "__main__":
    main()
