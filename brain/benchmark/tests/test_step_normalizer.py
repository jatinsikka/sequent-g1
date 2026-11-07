"""Regression tests for the deterministic planner's prose/compound step normalizer.

Before the fix, 411 corpus steps (106 of them stop/e-stop/isolation actions) in five
equipment families were silently dropped because they were written as prose.
"""
from __future__ import annotations

from src.planner.infer_plan import _normalize_step, deterministic_plan


def test_prose_estop_is_kept():
    assert _normalize_step("Press estop_button immediately to halt all machine motion.") == [
        "press_button estop_button"]


def test_compound_steps_are_split():
    assert _normalize_step("Walk to HMI_panel and read_sensor spindle_rpm_display to confirm.") == [
        "walk_to HMI_panel", "read_sensor spindle_rpm_display"]
    assert _normalize_step("Pick lockout_tag and place at coolant_pump_panel before maintenance.") == [
        "pick lockout_tag", "place lockout_tag at coolant_pump_panel"]
    assert _normalize_step("press_button mold_close_button and notify process_engineer that it is done") == [
        "press_button mold_close_button", "notify process_engineer"]


def test_canonical_single_steps_unchanged():
    for s in ["walk_to spill_station", "place soaked_absorbent_pad into hazmat_waste_bin", "wait 300 seconds"]:
        assert _normalize_step(s) == [s]


def test_conditionals_left_unparsed():
    s = "If RPM remains below setpoint, press spindle_stop_button and notify maintenance_technician."
    assert _normalize_step(s) == [s]


def test_plan_keeps_estop():
    plan = deterministic_plan({"title": "Overspeed", "steps": [
        "Press spindle_stop_button to halt spindle rotation immediately.",
        "Walk to HMI_panel and read_sensor spindle_rpm_display to confirm the overspeed reading."]})
    assert [s.skill for s in plan.steps] == ["press_button", "walk_to", "read_sensor", "notify"]
    assert plan.steps[0].args == {"button": "spindle_stop_button"}
