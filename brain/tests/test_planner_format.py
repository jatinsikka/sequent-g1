from __future__ import annotations

from src.planner.infer_plan import parse_plan_from_model_output
from src.data.schemas import Plan


def test_planner_parsing_and_repair():
    bad_text = "{'goal':'Reset','steps':[{'skill':'press_button','args':{'button':'green_reset'}},],}"
    plan = parse_plan_from_model_output(bad_text)
    assert isinstance(plan, Plan)
    assert plan.steps and isinstance(plan.steps[0].args, dict)


