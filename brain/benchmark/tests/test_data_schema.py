from __future__ import annotations

from src.data.schemas import IncidentEntry, Plan, PlanStep, SOPEntry, load_json, SKILLS


def test_load_schemas():
    sops = load_json("src/data/sop_examples.json", SOPEntry, key="sop_examples")
    incs = load_json("src/data/incident_examples.json", IncidentEntry, key="incident_examples")
    assert len(sops) == 100  # Updated for 100 SOP examples
    assert len(incs) == 100  # Updated for 100 incident examples
    assert isinstance(sops[0].steps, list)
    assert isinstance(incs[0].text, str)


def test_plan_roundtrip():
    p = Plan(goal="Test", steps=[PlanStep(skill="notify", args={"level": "tech"})], fallback=[])
    s = p.model_dump_json()
    p2 = Plan.model_validate_json(s)
    assert p2.goal == "Test"
    assert p2.steps[0].skill in SKILLS


