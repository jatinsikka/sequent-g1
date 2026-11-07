"""The robot planner must read steps written in skill syntax (the 1,012-SOP corpus).

Before the fix, r"\\bpress\\b" never matched "press_button" ("_" is a word character),
so 994 press steps and every walk_to/read_sensor step fell back to `notify`.
"""
from src.planner.infer_plan import _classify_step, plan_from_sop


def test_skill_syntax_is_read_directly():
    assert _classify_step("press_button estop_button")[0] == "press_button"
    assert _classify_step("walk_to control_panel")[0] == "walk_to"
    assert _classify_step("read_sensor pump_pressure_sensor")[0] == "read_sensor"
    assert _classify_step("wait 300 seconds") == ("wait", {"seconds": 300})


def test_prose_steps_still_classified_by_keywords():
    assert _classify_step("Press red_button immediately")[0] == "press_button"
    assert _classify_step("Check the pressure gauge")[0] == "read_sensor"


def test_finale_sop_plan_has_no_spurious_notifies():
    steps = ["walk_to assembly_table", "pick jammed_part", "walk_to control_panel",
             "place jammed_part staging_area", "press_button yellow_reset_button",
             "notify maintenance_supervisor"]
    skills = [p["skill"] for p in plan_from_sop(steps) if p["source"] == "sop"]
    assert skills == ["walk_to", "pick", "walk_to", "place", "press_button", "notify"]
