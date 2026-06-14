"""
Front door: natural-language incident -> retrieved SOP -> plan -> verified execution.

    python run_task.py --command "Machine A pressure is low"
    python run_task.py --plan myplan.json

Planning today: the brain bridge (numpy TF-IDF retrieval over the 100-SOP library
+ deterministic step->skill mapping). Swapping in a frontier LLM for retrieval +
planning is the "A" path; the executor below is already the real thing.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan  # noqa: E402

from brain_bridge import incident_to_plan
from executor import VerifyingExecutor


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--command", help="natural-language incident")
    g.add_argument("--plan", help="path to a JSON plan")
    p.add_argument("--grasp_model", default="checkpoints/v55_final.zip")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.plan:
        with open(args.plan) as f:
            plan = Plan.model_validate(json.load(f))
    else:
        plan, sop, score = incident_to_plan(args.command)
        print(f"\nINCIDENT — {args.command}")
        print(f"RETRIEVED — SOP {sop['sop_id']}: {sop['title']} (score {score:.3f})")
        print(f"  condition: {sop['condition']}")

    print(f"\nPLAN — {plan.goal}")
    for i, s in enumerate(plan.steps):
        print(f"  [{i}] {s.skill}({', '.join(f'{k}={v}' for k, v in s.args.items())})")
    print()

    ex = VerifyingExecutor(grasp_model_path=args.grasp_model, device=args.device)
    report = ex.run(plan)
    print(report.render())


if __name__ == "__main__":
    main()
