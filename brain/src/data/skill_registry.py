"""
SKILL REGISTRY — the single source of truth for the robot's skill library.

The whole pipeline speaks these skills: the retriever maps an incident to an SOP, the
planner maps each SOP step to one of these skills (see `planner.infer_plan.plan_from_sop`),
and the verifying executor runs each skill and checks its postcondition against measured
physics (see `executor.VerifyingExecutor`).

Each entry documents: what the skill does, its args, which subsystem runs it, its
pre/postconditions, and its HONEST verification status (validated 2026-06-21):
  - "verified"      : runs in sim and is checked against measured physics; reliable.
  - "reach-limited" : physically attempted, but the frozen AMO base controller recoils ~18cm
                      when the arm reaches, so it cannot reliably complete; the verifier
                      reports it as NOT verified rather than faking success.
  - "operational"   : a non-physical step (sensor read, wait, notify, logical place) executed
                      as a logged operation; not a physics-checked skill.

Author: Jatin Sikka
"""
from __future__ import annotations
from typing import Dict, List

SKILL_REGISTRY: Dict[str, dict] = {
    "walk_to": {
        "kind": "locomotion",
        "description": "Navigate the robot to a named location via PID position control on top of the AMO whole-body locomotion policy.",
        "args": {"target": "machine | table | shelf | <location>"},
        "precondition": "robot is upright and standing",
        "postcondition": "robot is within the arrival threshold of the target AND did not fall (measured from mjData)",
        "runner": "skills_locomotion.run_walk_to",
        "status": "verified",
        "note": "Solid and repeatable: arrives upright across targets, never falls.",
    },
    "press_button": {
        "kind": "manipulation",
        "description": "Reach out and press a control-panel button, holding it depressed past the trip threshold.",
        "args": {"button": "red_button | green_button | blue_button | yellow_button"},
        "precondition": "robot is standing in front of the panel",
        "postcondition": "button displacement held >= threshold for >=25 steps with the base upright",
        "runner": "skills_manipulation.run_press_button",
        "status": "reach-limited",
        "note": "The arm cannot reach a flush wall button standing (AMO recoils the pelvis ~18cm). The verifier flags it NOT verified; needs a base-controller rework.",
    },
    "pick": {
        "kind": "manipulation",
        "description": "Reach to an object on a surface and grasp it (magnetic latch on slow contact), then lift and hold.",
        "args": {"object": "screwdriver | wrench | intake_cover | cleaning_cloth | <object>"},
        "precondition": "object present on the work surface within nominal reach",
        "postcondition": "object grasped AND lifted >= min_lift sustained for >= sustain_steps",
        "runner": "skills_manipulation.run_grasp",
        "status": "reach-limited",
        "note": "Same AMO recoil limit: the hand reaches only ~to the body, so tabletop grasps mostly miss/knock-off. Verifier reports honestly.",
    },
    "place": {
        "kind": "manipulation",
        "description": "Place a held object at a target location.",
        "args": {"object": "<object>", "location": "machine | shelf | table | container_bin"},
        "precondition": "object currently held",
        "postcondition": "object released at the target location (depends on a working pick)",
        "runner": None,
        "status": "operational",
        "note": "Logical/operational until pick is verified; not independently physics-checked yet.",
    },
    "read_sensor": {
        "kind": "operational",
        "description": "Read a machine sensor value (pressure / temperature / vibration / light).",
        "args": {"sensor": "pressure_sensor | temperature_sensor | vibration_sensor | light_sensor"},
        "precondition": "robot at the machine",
        "postcondition": "sensor value recorded",
        "runner": None,
        "status": "operational",
        "note": "Non-physical step; executed as a logged read.",
    },
    "wait": {
        "kind": "operational",
        "description": "Pause for a fixed duration (e.g., let a shutdown settle).",
        "args": {"seconds": "<int>"},
        "precondition": None,
        "postcondition": "duration elapsed",
        "runner": None,
        "status": "operational",
        "note": "Non-physical timing step.",
    },
    "notify": {
        "kind": "operational",
        "description": "Notify a human (technician / manager) — the standard end-of-SOP escalation.",
        "args": {"level": "tech | manager"},
        "precondition": None,
        "postcondition": "notification logged",
        "runner": None,
        "status": "operational",
        "note": "Non-physical communication step.",
    },
}

# The whitelist the planner/executor enforce (derived from the registry — single source of truth).
SKILLS: List[str] = list(SKILL_REGISTRY.keys())

# Convenience views.
PHYSICAL_SKILLS = [s for s, m in SKILL_REGISTRY.items() if m["kind"] in ("locomotion", "manipulation")]
VERIFIED_SKILLS = [s for s, m in SKILL_REGISTRY.items() if m["status"] == "verified"]


def skill_card(name: str) -> str:
    """One-line human-readable summary of a skill (for docs/UI)."""
    m = SKILL_REGISTRY[name]
    return f"{name} [{m['kind']}/{m['status']}] — {m['description']}"
