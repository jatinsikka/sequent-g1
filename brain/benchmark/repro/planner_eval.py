"""Experiment 5: the deterministic ("faithful") planner, measured over the whole corpus and end to end.

1. Corpus parse coverage: for every SOP, what fraction of its steps become a contract-valid PlanStep
   via src.planner.infer_plan.deterministic_plan, and how many plans are fully valid.
2. End-to-end plan accuracy: using the gold rank files written by repro_retrieval.py, is the plan from the
   top-1 retrieved SOP identical to the gold SOP's plan? Distinguishes "wrong SOP but right actions"
   (benign miss) from "wrong actions" (harmful miss).
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd
from common import RESULTS, family_map, load

from src.planner.infer_plan import _normalize_step, _parse_step, deterministic_plan  # noqa: E402

sops, _ = load(exclude_finale=True)
by_id = {s.sop_id: s for s in sops}
fam = family_map()

rows, skill_counts = [], Counter()
plans = {}
for s in sops:
    parsed_raw = [_parse_step(st) for st in s.steps]  # behaviour before the _normalize_step fix
    parsed = [all(_parse_step(x) for x in _normalize_step(st)) or None for st in s.steps]
    p = deterministic_plan(s.model_dump())
    plans[s.sop_id] = p
    for st in p.steps:
        skill_counts[st.skill] += 1
    rows.append(dict(sop_id=s.sop_id, family=fam[s.sop_id], n_steps=len(s.steps),
                     n_parsed_before_fix=sum(x is not None for x in parsed_raw), n_parsed=sum(x is not None for x in parsed), n_plan_steps=len(p.steps),
                     fallback_walk=(not any(parsed)), executable_on_g1=sum(
                         st.skill in {"walk_to", "pick", "place", "press_button"} for st in p.steps)))
cov = pd.DataFrame(rows)
cov.to_csv(RESULTS / "planner_coverage.csv", index=False)
tot = cov.n_steps.sum()
print(f"SOPs: {len(cov)}  steps: {tot}  parsed before fix: {cov.n_parsed_before_fix.sum()} ({cov.n_parsed_before_fix.sum()/tot:.1%})"
      f"  after fix: {cov.n_parsed.sum()} ({cov.n_parsed.sum()/tot:.1%})")
print(f"SOPs fully parsed before fix: {(cov.n_parsed_before_fix == cov.n_steps).mean():.1%}")
print(f"SOPs with every step parsed: {(cov.n_parsed == cov.n_steps).mean():.1%}  | empty->fallback: {cov.fallback_walk.sum()}")
print("skill mix in plans:", dict(skill_counts.most_common()))
print(f"plan steps physically executed by the G1 demo skills (walk/pick/place/press): "
      f"{cov.executable_on_g1.sum()/cov.n_plan_steps.sum():.1%}")

def sig(p, with_args=True):
    return tuple((st.skill, tuple(sorted(st.args.items())) if with_args else ()) for st in p.steps)

out = []
for f in sorted(RESULTS.glob("ranks_official_*_s*.csv")):
    method = f.stem.split("_", 2)[2]
    r = pd.read_csv(f)
    ok1 = r.top1 == r.gold
    same_args = [sig(plans[t]) == sig(plans[g]) for t, g in zip(r.top1, r.gold)]
    same_skills = [sig(plans[t], False) == sig(plans[g], False) for t, g in zip(r.top1, r.gold)]
    same_fam = [fam[t] == fam[g] for t, g in zip(r.top1, r.gold)]
    miss = ~ok1
    out.append(dict(method=method, R1=ok1.mean(), plan_exact=pd.Series(same_args).mean(),
                    skill_seq_match=pd.Series(same_skills).mean(),
                    misses=int(miss.sum()),
                    misses_same_family=float(pd.Series(same_fam)[miss].mean()) if miss.any() else None,
                    misses_same_skill_seq=float(pd.Series(same_skills)[miss].mean()) if miss.any() else None))
e2e = pd.DataFrame(out).round(3)
e2e.to_csv(RESULTS / "planner_end_to_end.csv", index=False)
print(e2e.to_string(index=False))
