from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from rich.console import Console

from src.data.schemas import Plan, SKILLS, validate_skill_args

console = Console()


class SkillAPI(ABC):
    @abstractmethod
    def walk_to(self, target: str) -> bool: ...

    @abstractmethod
    def press_button(self, button: str) -> bool: ...

    @abstractmethod
    def wait(self, seconds: float) -> bool: ...

    @abstractmethod
    def read_sensor(self, sensor: str, expect: str | None = None) -> dict: ...

    @abstractmethod
    def pick(self, object: str) -> bool: ...

    @abstractmethod
    def place(self, object: str, location: str) -> bool: ...

    @abstractmethod
    def notify(self, level: str = "tech") -> bool: ...


class DummyMuJoCoAdapter(SkillAPI):
    def walk_to(self, target: str) -> bool:
        console.log(f"walk_to: {target}")
        return True

    def press_button(self, button: str) -> bool:
        console.log(f"press_button: {button}")
        return True

    def wait(self, seconds: float) -> bool:
        console.log(f"wait: {seconds}s")
        return True

    def read_sensor(self, sensor: str, expect: str | None = None) -> dict:
        console.log(f"read_sensor: {sensor}, expect={expect}")
        return {"sensor": sensor, "value": "ok"}

    def pick(self, object: str) -> bool:
        console.log(f"pick: {object}")
        return True

    def place(self, object: str, location: str) -> bool:
        console.log(f"place: {object} -> {location}")
        return True

    def notify(self, level: str = "tech") -> bool:
        console.log(f"notify: {level}")
        return True


def execute_plan(plan: Plan, api: SkillAPI) -> Dict[str, object]:
    """Sequentially execute plan steps against the given SkillAPI.

    Each step is validated against the SKILL_ARGS contract *before* dispatch, and
    the dispatch itself is wrapped so that a misbehaving skill is recorded as a
    failed step rather than raising. This guarantees execute_plan always returns a
    structured summary — callers (the demo CLI and evaluate_all) get an honest
    success/failure signal instead of an uncaught exception that aborts the run.

    Returns a dict with:
      success:      True iff every step ran and returned a truthy result
      failed_step:  index of the first failing step, or None on full success
      results:      per-step record [{"step", "skill", "ok", "error"?}, ...]
    """
    results: list[Dict[str, object]] = []
    for i, step in enumerate(plan.steps):
        skill = step.skill
        args = step.args

        def _fail(reason: str) -> Dict[str, object]:
            console.log(f"[red]Step {i} ({skill}) failed: {reason}[/red]")
            results.append({"step": i, "skill": skill, "ok": False, "error": reason})
            return {"success": False, "failed_step": i, "results": results}

        # 1. Skill must be known to both the contract and the adapter.
        if skill not in SKILLS or not hasattr(api, skill):
            return _fail(f"unknown skill: {skill!r}")

        # 2. Args must satisfy the skill's contract before we dispatch.
        arg_error = validate_skill_args(skill, args)
        if arg_error:
            return _fail(arg_error)

        # 3. Dispatch, capturing any runtime error instead of crashing.
        method = getattr(api, skill)
        try:
            ret = method(**args)
        except Exception as e:  # noqa: BLE001 - any skill failure is a failed step
            return _fail(f"{type(e).__name__}: {e}")

        # read_sensor returns a dict (always truthy on success); actions return bool.
        ok = True if isinstance(ret, dict) else bool(ret)
        results.append({"step": i, "skill": skill, "ok": ok})
        if not ok:
            console.log(f"[red]Step {i} ({skill}) returned failure[/red]")
            return {"success": False, "failed_step": i, "results": results}

    return {"success": True, "failed_step": None, "results": results}


