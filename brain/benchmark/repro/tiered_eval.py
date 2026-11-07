"""Experiment 9: retrieval accuracy vs how informed the operator is.

Tiers (all ranked against the same 1,012-SOP index, no retraining):
  hard           the shipped held-out test incidents (vague, symptom-first operator language)
  machine_named  the same incidents with the equipment named up front: "<Machine>: <incident>"
                 (simulates an operator who states which machine; same gold labels)
  informed       hand-written clear tickets from an operator who knows the system
                 (data/informed_tickets.json; machine + symptom stated, SOP title not copied)

Models: TF-IDF, BM25, zero-shot MiniLM, zero-shot bge-small, and the fine-tuned MiniLM
(artifacts/retriever_ft_s<seed> if present, else artifacts/retriever_st_orig).
Writes results/tiered_eval.csv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from common import RESULTS, ROOT, family_map, load, metrics, synth

HERE = Path(__file__).resolve().parent
sops, split = load(exclude_finale=True)
ids = [s.sop_id for s in sops]
texts = [synth(s) for s in sops]
fam = family_map()
ACRONYMS = {"agv": "AGV", "cmm": "CMM", "vfd": "VFD", "hvac": "HVAC", "cnc": "CNC"}


def machine(slug: str) -> str:
    """'agv_dock' -> 'AGV dock', 'hydraulic_press' -> 'Hydraulic press'."""
    words = [ACRONYMS.get(w, w) for w in slug.split("_")]
    return " ".join([words[0] if words[0].isupper() else words[0].capitalize()] + words[1:])

tiers = {
    "hard": [(i.text, i.labels["sop_id"]) for i in split["test"]],
    "machine_named": [(f"{machine(fam[i.labels['sop_id']])}: {i.text}", i.labels["sop_id"]) for i in split["test"]],
}
inf = HERE / "data" / "informed_tickets.json"
if inf.exists():
    tiers["informed"] = [(t["text"], t["sop_id"]) for t in json.loads(inf.read_text())]


def rank(scores):
    o = np.argsort(-scores, axis=1)
    return [[ids[j] for j in row] for row in o]


from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

vec = TfidfVectorizer(ngram_range=(1, 2)); S_tf = vec.fit_transform(texts)
tok = lambda t: re.findall(r"[a-z0-9]+", t.lower()); bm = BM25Okapi([tok(t) for t in texts])
ft_dirs = sorted((ROOT / "artifacts").glob("retriever_ft_s*")) or [ROOT / "artifacts" / "retriever_st_orig"]
encoders = {"zs_minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
            "zs_bge_small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ")}
encoders.update({f"ft_minilm_{d.name.split('_')[-1]}": (str(d), "") for d in ft_dirs})
models = {k: (SentenceTransformer(p), pre) for k, (p, pre) in encoders.items()}
S_emb = {k: m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False) for k, (m, _) in models.items()}

rows = []
for tier, data in tiers.items():
    q = [d[0] for d in data]; g = [d[1] for d in data]
    res = {"tfidf": rank((vec.transform(q) @ S_tf.T).toarray()),
           "bm25": rank(np.stack([bm.get_scores(tok(t)) for t in q]))}
    for k, (m, pre) in models.items():
        res[k] = rank(m.encode([pre + t for t in q], normalize_embeddings=True, show_progress_bar=False) @ S_emb[k].T)
    for k, ranked in res.items():
        mm = metrics(ranked, g)
        same_fam = np.mean([fam[r[0]] == fam[gg] for r, gg in zip(ranked, g)])
        rows.append(dict(tier=tier, n=len(g), method=k, **mm, top1_right_machine=same_fam))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "tiered_eval.csv", index=False)
ft = df[df.method.str.startswith("ft_")]
summ = pd.concat([df[~df.method.str.startswith("ft_")],
                  ft.groupby("tier", as_index=False).agg(n=("n", "first"), **{c: (c, "mean") for c in
                     ["R@1", "R@5", "MRR", "top1_right_machine"]}).assign(method=f"ft_minilm (mean of {ft.method.nunique()})")])
print(summ.pivot_table(index="method", columns="tier", values="R@1", sort=False).round(3).to_string())
print("\nR@5:"); print(summ.pivot_table(index="method", columns="tier", values="R@5", sort=False).round(3).to_string())
print("\ntop-1 is the right machine:"); print(summ.pivot_table(index="method", columns="tier", values="top1_right_machine", sort=False).round(3).to_string())
