"""Retrieval using the trained sentence-transformer bi-encoder (all-MiniLM + MNR).

This is the retriever validated in RESULTS.md (R@1 0.572 on the held-out test,
~90% over TF-IDF). It replaces the old bert-CLS/FAISS path for the live pipeline.

SOP embeddings are computed once and cached to disk so repeated queries are fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.schemas import SOPEntry, load_json

_ROOT = Path(__file__).resolve().parent.parent.parent
_ENCODER_DIR = _ROOT / "artifacts" / "retriever_st"
_SOPS_PATH = _ROOT / "src" / "data" / "plant_sops.json"

_state: dict = {}


def _synth(s: SOPEntry) -> str:
    return f"{s.title}. {s.condition}. Steps: " + " ; ".join(s.steps)


def _load() -> dict:
    if _state:
        return _state
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(_ENCODER_DIR))
    sops = load_json(str(_SOPS_PATH), SOPEntry, key="sop_examples")
    ids = [s.sop_id for s in sops]

    # Cache is written by this module only (trusted); ids are stored as a plain
    # unicode array so no pickle is needed on load.
    cache = _ENCODER_DIR / "sop_emb_cache.npz"
    if cache.exists():
        d = np.load(cache)
        if len(d["ids"]) == len(ids):
            _state.update(model=model, emb=d["emb"], ids=[str(x) for x in d["ids"]], sops={s.sop_id: s for s in sops})
            return _state
    emb = model.encode([_synth(s) for s in sops], normalize_embeddings=True, show_progress_bar=False)
    np.savez(cache, emb=emb, ids=np.array(ids))
    _state.update(model=model, emb=np.asarray(emb), ids=ids, sops={s.sop_id: s for s in sops})
    return _state


def retrieve_topk_st(incident_text: str, k: int = 5) -> List[Dict[str, object]]:
    """Return the top-k SOPs for an incident, ranked by the trained bi-encoder."""
    st = _load()
    q = st["model"].encode([incident_text], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = st["emb"] @ q
    order = np.argsort(-scores)[:k]
    hits = []
    for i in order:
        sid = st["ids"][int(i)]
        s = st["sops"][sid]
        hits.append({"sop_id": sid, "score": float(scores[int(i)]), "title": s.title, "text": _synth(s)})
    return hits
