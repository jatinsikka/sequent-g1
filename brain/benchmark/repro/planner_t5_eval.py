"""Experiment 5b: the Flan-T5 LoRA planner's "0/20 valid plans" claim.

Feeds the first N held-out test incidents together with their GOLD SOP (so retrieval is not
being tested) through the repo's own prompt builder and generator, then scores each output:
  json_ok      raw output parses with json.loads (the original criterion)
  plan_ok      the repo's lenient parser recovers >=1 contract-valid step
  faithful     the recovered skill sequence equals deterministic_plan(gold SOP)'s skill sequence
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from common import RESULTS, load

from src.data.schemas import SKILLS, validate_skill_args  # noqa: E402
from src.planner import infer_plan as ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--tag", default="config_as_shipped")
args = ap.parse_args()

sops, split = load(exclude_finale=True)
by_id = {s.sop_id: s for s in sops}
model, tok = ip._load_lora_model()
assert model is not None, "LoRA adapter failed to load"

rows = []
for inc in split["test"][: args.n]:
    sop = by_id[inc.labels["sop_id"]].model_dump()
    prompt = ip._build_prompt(incident=inc.text, sop_title=sop["title"], sop_cond=sop["condition"],
                              sop_steps=sop["steps"], skills=SKILLS)
    out = ip._generate_with_lora(prompt, model, tok)
    try:
        json.loads(out); json_ok = True
    except Exception:
        json_ok = False
    try:
        p = ip.parse_plan_from_model_output(out)
        steps = [s for s in p.steps if validate_skill_args(s.skill, s.args) is None]
    except Exception:
        steps = []
    gold = [s.skill for s in ip.deterministic_plan(sop).steps]
    rows.append(dict(incident_id=inc.incident_id, sop_id=sop["sop_id"], json_ok=json_ok, plan_ok=bool(steps),
                     faithful=[s.skill for s in steps] == gold, n_steps=len(steps), n_gold=len(gold),
                     output=out[:400]))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"planner_t5_{args.tag}.csv", index=False)
print(f"[{args.tag}] json_ok {df.json_ok.sum()}/{len(df)}  plan_ok {df.plan_ok.sum()}/{len(df)}  "
      f"faithful {df.faithful.sum()}/{len(df)}  mean steps {df.n_steps.mean():.1f} vs gold {df.n_gold.mean():.1f}")
print("example output:", df.output.iloc[0][:300])
