"""
Front door: natural-language command -> plan -> verified execution.

    python run_task.py --command "pick up the screwdriver"
    python run_task.py --plan myplan.json

Planning today: a thin keyword stub that emits a Plan for the demo command.
Wiring the real brain (SOP retrieval + frontier-LLM planner) into this seam is
the next step; the executor below is already the real thing.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan, PlanStep  # noqa: E402

from executor import VerifyingExecutor


def stub_plan(command: str) -> Plan:
    """Placeholder planner. Replaced by brain retrieval + LLM in the next step."""
    c = command.lower()
    if any(w in c for w in ("pick", "grab", "grasp", "screwdriver", "tool")):
        return Plan(goal=command, steps=[PlanStep(skill="pick", args={"obj": "screwdriver"})])
    raise SystemExit(f"stub planner has no plan for: {command!r} (wire the brain to handle this)")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--command", help="natural-language instruction")
    g.add_argument("--plan", help="path to a JSON plan")
    p.add_argument("--grasp_model", default="checkpoints/v55_final.zip")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.plan:
        with open(args.plan) as f:
            plan = Plan.model_validate(json.load(f))
    else:
        plan = stub_plan(args.command)

    print(f"\nPLAN — {plan.goal}")
    for i, s in enumerate(plan.steps):
        print(f"  [{i}] {s.skill}({', '.join(f'{k}={v}' for k, v in s.args.items())})")
    print()

    ex = VerifyingExecutor(grasp_model_path=args.grasp_model, device=args.device)
    report = ex.run(plan)
    print(report.render())


if __name__ == "__main__":
    main()
