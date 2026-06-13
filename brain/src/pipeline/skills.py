from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from rich.console import Console

from src.data.schemas import Plan

console = Console()


class SkillAPI(ABC):
    @abstractmethod
    def walk_to(self, target: str) -> bool: ...

    @abstractmethod
    def press_button(self, button: str) -> bool: ...

    @abstractmethod
    def wait(self, sec: float) -> bool: ...

    @abstractmethod
    def read_sensor(self, sensor: str, expect: str | None = None) -> dict: ...

    @abstractmethod
    def pick(self, obj: str) -> bool: ...

    @abstractmethod
    def place(self, target: str) -> bool: ...

    @abstractmethod
    def notify(self, level: str = "tech") -> bool: ...


class DummyMuJoCoAdapter(SkillAPI):
    def walk_to(self, target: str) -> bool:
        console.log(f"walk_to: {target}")
        return True

    def press_button(self, button: str) -> bool:
        console.log(f"press_button: {button}")
        return True

    def wait(self, sec: float) -> bool:
        console.log(f"wait: {sec}s")
        return True

    def read_sensor(self, sensor: str, expect: str | None = None) -> dict:
        console.log(f"read_sensor: {sensor}, expect={expect}")
        return {"sensor": sensor, "value": "ok"}

    def pick(self, obj: str) -> bool:
        console.log(f"pick: {obj}")
        return True

    def place(self, target: str) -> bool:
        console.log(f"place: {target}")
        return True

    def notify(self, level: str = "tech") -> bool:
        console.log(f"notify: {level}")
        return True


def execute_plan(plan: Plan, api: SkillAPI) -> Dict[str, object]:
    """Sequentially executes plan steps using provided SkillAPI."""
    for i, step in enumerate(plan.steps):
        skill = step.skill
        args = step.args
        try:
            method = getattr(api, skill)
        except AttributeError:
            console.log(f"[red]Unknown skill: {skill}[/red]")
            return {"success": False, "failed_step": i}
        ok = method(**args)
        if not ok:
            return {"success": False, "failed_step": i}
    return {"success": True, "failed_step": None}


