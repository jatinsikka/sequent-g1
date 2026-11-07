from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from src.data.schemas import Plan


def retrieval_recall_at_k(golds: Sequence[str], preds: Sequence[Sequence[str]], k: int) -> float:
    """Recall@k: fraction of queries where gold id appears in top-k predictions."""
    ok = 0
    for g, p in zip(golds, preds):
        if g in p[:k]:
            ok += 1
    return ok / max(1, len(golds))


def mrr_score(golds: Sequence[str], ranked_lists: Sequence[Sequence[str]]) -> float:
    """Mean Reciprocal Rank."""
    total = 0.0
    for g, lst in zip(golds, ranked_lists):
        rr = 0.0
        for i, x in enumerate(lst):
            if x == g:
                rr = 1.0 / (i + 1)
                break
        total += rr
    return total / max(1, len(golds))


def json_plan_f1(ref_plan: Plan, hyp_plan: Plan) -> float:
    """Compare skill sequences and args keys overlap."""
    ref_skills = [s.skill for s in ref_plan.steps]
    hyp_skills = [s.skill for s in hyp_plan.steps]
    if not ref_skills and not hyp_skills:
        return 1.0
    inter = len([s for s in hyp_skills if s in ref_skills])
    prec = inter / max(1, len(hyp_skills))
    rec = inter / max(1, len(ref_skills))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def execution_success_rate(summaries: Sequence[Dict[str, object]]) -> float:
    ok = sum(1 for s in summaries if bool(s.get("success")))
    return ok / max(1, len(summaries))


