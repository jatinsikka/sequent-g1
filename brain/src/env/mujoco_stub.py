from __future__ import annotations

from rich.console import Console
from src.pipeline.skills import SkillAPI

console = Console()


class MuJoCoAdapter(SkillAPI):
    """Placeholder mapping from SOP names to MJCF actuators/sensors."""

    def walk_to(self, target: str) -> bool:
        console.log(f"[MJCF] walk_to: {target} (mock)")
        return True

    def press_button(self, button: str) -> bool:
        act = f"act_{button}"
        console.log(f"[MJCF] press button {act} (mock)")
        return True

    def wait(self, sec: float) -> bool:
        console.log(f"[MJCF] wait {sec}s (mock)")
        return True

    def read_sensor(self, sensor: str, expect: str | None = None) -> dict:
        name = f"sen_{sensor}"
        console.log(f"[MJCF] read sensor {name}, expect={expect} (mock)")
        return {"sensor": sensor, "value": "ok"}

    def pick(self, obj: str) -> bool:
        console.log(f"[MJCF] pick {obj} (mock)")
        return True

    def place(self, target: str) -> bool:
        console.log(f"[MJCF] place to {target} (mock)")
        return True

    def notify(self, level: str = "tech") -> bool:
        console.log(f"[MJCF] notify level={level} (mock)")
        return True


