"""
Bridge: incident text -> retrieved SOP -> typed Plan (the no-API "B" path of run_task.py).

SOP library and retriever, in order of preference:
  1. brain/benchmark/: the 1,012-SOP corpus ranked by the fine-tuned MiniLM bi-encoder, when
     brain/benchmark/artifacts/retriever_st exists (train it with
     `cd brain/benchmark && python -m src.retrieval.train_eval_sentence_encoder`).
  2. Otherwise the original 100-SOP library (brain/src/data/sop_examples.json) ranked by
     off-the-shelf MiniLM, plus numpy TF-IDF as a no-dependency fallback.
The semantic and lexical rankings are interleaved (semantic first, so the top-1 pick is the
semantic one); the step->skill mapping is the keyword planner in brain/src/planner.

Author: Jatin Sikka
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan, PlanStep, SKILLS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, "brain", "benchmark")
_FT_DIR = os.path.join(_BENCH, "artifacts", "retriever_st")
_USE_BENCH = os.path.isdir(_FT_DIR)
_SOP_PATH = (os.path.join(_BENCH, "src", "data", "plant_sops.json") if _USE_BENCH
             else os.path.join(_HERE, "brain", "src", "data", "sop_examples.json"))
_FT_STATE: dict = {}
_TOKEN = re.compile(r"[a-z0-9_]+")


def _load_sops() -> List[dict]:
    with open(_SOP_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["sop_examples"] if isinstance(d, dict) else d


def _text(sop: dict) -> str:
    return f"{sop['title']}. {sop['condition']}. " + " ; ".join(sop.get("steps", []))


def _tok(s: str) -> List[str]:
    return _TOKEN.findall(s.lower())


# ---- retrieval: SEMANTIC (sentence-transformers, the upgraded brain) with ----
# ---- the numpy TF-IDF cosine as the no-dependency fallback -------------------

def retrieve_sop(incident: str, sops: List[dict], k: int = 3) -> List[Tuple[dict, float]]:
    """HYBRID retrieval: interleave the SEMANTIC ranking (brain/src MiniLM retriever — wins
    paraphrases: smoke -> Smoke Detection) with the lexical TF-IDF ranking (wins exact title
    matches: 'pressure is low' -> Low Pressure Warning), dedup, top-k. The LLM planner picks
    the SOP from this candidate set, so what matters is that the right SOP is IN it — each
    ranker's best candidates are guaranteed a slot. Falls back to pure lexical if the
    semantic stack is unavailable."""
    lex = _retrieve_sop_tfidf(incident, sops, k=max(k, 3))
    if _USE_BENCH:
        sem = _retrieve_sop_finetuned(incident, sops, k=max(k, 3))
        return _interleave(sem, lex, k)
    try:
        from src.retrieval.infer_retrieve import retrieve_topk
        by_id = {s["sop_id"]: s for s in sops}
        sem = [(by_id[r["sop_id"]], float(r["score"])) for r in retrieve_topk(incident, k=max(k, 3))
               if r["sop_id"] in by_id]
    except Exception:
        sem = []  # semantic stack missing/broken -> lexical only
    return _interleave(sem, lex, k)


def _interleave(sem, lex, k):
    merged, seen = [], set()
    for pair in [p for ab in zip(sem, lex) for p in ab] + sem + lex:   # interleave sem/lex, then leftovers
        if pair[0]["sop_id"] not in seen:
            seen.add(pair[0]["sop_id"]); merged.append(pair)
    return merged[:k] if merged else lex[:k]


def _retrieve_sop_finetuned(incident: str, sops: List[dict], k: int = 3) -> List[Tuple[dict, float]]:
    """Rank the 1,012-SOP corpus with the fine-tuned bi-encoder (cosine over normalized embeddings)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if not _FT_STATE:
        m = SentenceTransformer(_FT_DIR)
        # same SOP text format the retriever was trained on (train_eval_sentence_encoder.synth)
        synth = [f"{s['title']}. {s['condition']}. Steps: " + " ; ".join(s.get("steps", [])) for s in sops]
        _FT_STATE.update(model=m, emb=m.encode(synth, normalize_embeddings=True,
                                              show_progress_bar=False))
    q = _FT_STATE["model"].encode([incident], normalize_embeddings=True, show_progress_bar=False)[0]
    sc = _FT_STATE["emb"] @ q
    return [(sops[i], float(sc[i])) for i in np.argsort(-sc)[:k]]


def _retrieve_sop_tfidf(incident: str, sops: List[dict], k: int = 3) -> List[Tuple[dict, float]]:
    """Fallback: rank SOPs by TF-IDF cosine similarity to the incident text; top-k (sop, score)."""
    docs = [_tok(_text(s)) for s in sops]
    N = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}

    def vec(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        v = {t: (c / len(tokens)) * idf.get(t, math.log(1 + N) + 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    qv = vec(_tok(incident))
    scored = []
    for sop, d in zip(sops, docs):
        dv = vec(d)
        common = set(qv) & set(dv)
        scored.append((sop, sum(qv[t] * dv[t] for t in common)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


# ---- SOP steps -> typed skills ----------------------------------------------

def _find(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(0) if m else default


def step_to_skill(step: str) -> PlanStep:
    """Map one natural-language SOP step to a typed skill via keyword rules (the no-LLM fallback)."""
    s = step.lower()
    if "wait" in s:
        sec = _find(r"\d+", s, "1")
        return PlanStep(skill="wait", args={"sec": float(sec)})
    if "notify" in s or "alert" in s:
        who = "technician" if "technician" in s else ("operator" if "operator" in s else "tech")
        return PlanStep(skill="notify", args={"level": who})
    if "press" in s or "button" in s:
        return PlanStep(skill="press_button", args={"button": _find(r"\w+_button", s, "button")})
    if "read" in s or "check" in s or "sensor" in s:
        sensor = _find(r"\w+_(?:sensor|light)", s) or _find(r"\w+", s, "sensor")
        return PlanStep(skill="read_sensor", args={"sensor": sensor})
    if any(w in s for w in ("pick", "grasp", "grab", "retrieve", "lift")):
        obj = _find(r"\w+driver|tool|part|object|wrench|bolt", s, "object")
        return PlanStep(skill="pick", args={"obj": obj})
    if "place" in s or "put" in s:
        return PlanStep(skill="place", args={"target": _find(r"\w+", s, "target")})
    if "walk" in s or "go to" in s or "navigate" in s:
        tgt = _find(r"machine\w*|table|shelf|bin|station", s, "machine")
        return PlanStep(skill="walk_to", args={"target": tgt})
    # default: treat as a notify so nothing is silently dropped
    return PlanStep(skill="notify", args={"level": "tech", "note": step})


def incident_to_plan(incident: str) -> Tuple[Plan, dict, float]:
    """Retrieve the best SOP and turn its steps into a typed Plan. Returns (plan, sop, score).
    Uses the FAITHFUL planner (plan_from_sop: one skill per written SOP step, in order, with
    inferred walk_to navigation) — the keyword step_to_skill loop is the fallback."""
    sops = _load_sops()
    ranked = retrieve_sop(incident, sops)
    sop, score = ranked[0]
    try:
        from src.planner.infer_plan import plan_from_sop
        steps = [PlanStep(skill=s["skill"], args=s["args"]) for s in plan_from_sop(sop.get("steps", []))]
    except Exception:
        steps = [step_to_skill(st) for st in sop.get("steps", [])]
    plan = Plan(goal=f"{incident}  [SOP {sop['sop_id']}: {sop['title']}]", steps=steps)
    return plan, sop, score
