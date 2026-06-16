"""
Bridge: incident text -> retrieved SOP -> typed Plan, using the brain's 100-SOP
library and schema, with zero external dependencies (numpy only).

This is the no-API "B" path. The retriever (numpy TF-IDF cosine) and the
step->skill mapper (keyword rules) deliberately stand in for the brain's
sklearn/BERT retriever and Flan-T5 planner — both get replaced by a frontier
LLM in the "A" path. What's real here is the spine: a natural-language incident
selects a real SOP from the library, and that SOP's steps become an executable,
verifiable plan.

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

_SOP_PATH = os.path.join(os.path.dirname(__file__), "brain", "src", "data", "sop_examples.json")
_TOKEN = re.compile(r"[a-z0-9_]+")


def _load_sops() -> List[dict]:
    with open(_SOP_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["sop_examples"] if isinstance(d, dict) else d


def _text(sop: dict) -> str:
    return f"{sop['title']}. {sop['condition']}. " + " ; ".join(sop.get("steps", []))


def _tok(s: str) -> List[str]:
    return _TOKEN.findall(s.lower())


# ---- retrieval: numpy-free TF-IDF cosine over the SOP corpus -----------------

def retrieve_sop(incident: str, sops: List[dict], k: int = 3) -> List[Tuple[dict, float]]:
    """Rank SOPs by TF-IDF cosine similarity to the incident text; return the top-k (sop, score)."""
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
    """Retrieve the best SOP and turn its steps into a typed Plan. Returns (plan, sop, score)."""
    sops = _load_sops()
    ranked = retrieve_sop(incident, sops)
    sop, score = ranked[0]
    steps = [step_to_skill(st) for st in sop.get("steps", [])]
    plan = Plan(goal=f"{incident}  [SOP {sop['sop_id']}: {sop['title']}]", steps=steps)
    return plan, sop, score
