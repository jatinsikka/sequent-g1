from __future__ import annotations

from src.data.schemas import Plan
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan


def _plan_with_all_skills() -> Plan:
    """A plan exercising every skill with the arg keys the planner actually emits."""
    return Plan.model_validate(
        {
            "goal": "Exercise full skill contract",
            "steps": [
                {"skill": "walk_to", "args": {"target": "machine"}},
                {"skill": "read_sensor", "args": {"sensor": "pressure_sensor"}},
                {"skill": "press_button", "args": {"button": "red_button"}},
                {"skill": "pick", "args": {"object": "wrench"}},
                {"skill": "place", "args": {"object": "wrench", "location": "table"}},
                {"skill": "wait", "args": {"seconds": 3}},
                {"skill": "notify", "args": {"level": "tech"}},
            ],
            "fallback": [],
        }
    )


def test_execute_plan_runs_all_skills_without_crashing():
    """Every skill the planner can emit must dispatch cleanly through the adapter."""
    plan = _plan_with_all_skills()
    summary = execute_plan(plan, DummyMuJoCoAdapter())
    assert summary["success"] is True, summary
    assert summary["failed_step"] is None


def test_execute_plan_reports_bad_args_instead_of_crashing():
    """A step with an unexpected arg must be reported as a failed step, not raised."""
    bad = Plan.model_validate(
        {
            "goal": "bad",
            "steps": [{"skill": "wait", "args": {"nonexistent": 5}}],
            "fallback": [],
        }
    )
    summary = execute_plan(bad, DummyMuJoCoAdapter())  # must not raise
    assert summary["success"] is False
    assert summary["failed_step"] == 0
