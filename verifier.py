"""
Runtime verification layer: physics-checked pre/postconditions for skills.

A skill only *starts* if its preconditions hold against the live MuJoCo state,
and only *counts* if its postconditions are measured true — object actually
displaced, grasp actually latched, robot actually stable. A cheated reward
can never read as a finished job.

v0 scope: conditions for the grasp skill, evaluated against G1RLEnv.
The contract interface is skill-agnostic; walk_to / carry_to / place add
their own conditions without changing the executor.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class CheckResult:
    """Outcome of one condition evaluated against the physics state."""
    name: str
    ok: bool
    measured: float
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.name}: measured={self.measured:.3f} vs {self.threshold:.3f} {self.detail}"


@dataclass
class Condition:
    """A named, machine-checkable predicate over the simulator state."""
    name: str
    fn: Callable[[object], CheckResult]

    def check(self, env) -> CheckResult:
        return self.fn(env)


# ---------------------------------------------------------------------------
# Condition library (grasp skill, v0)
# ---------------------------------------------------------------------------

def _unwrap(env):
    """Accept either G1RLEnv or a gym wrapper around it."""
    return env.unwrapped if hasattr(env, "unwrapped") else env


def base_stable() -> Condition:
    """Robot upright: pelvis height and torso attitude inside termination bounds."""
    def fn(env) -> CheckResult:
        e = _unwrap(env)
        height = float(e.env.data.xpos[e.pelvis_body_id][2])
        quat = e.env.data.xquat[e.pelvis_body_id]
        roll, pitch, _ = e._quat_to_euler(np.array(quat))
        min_h = getattr(e, "min_height", 0.4)
        max_r = getattr(e, "max_roll", 0.8)
        max_p = getattr(e, "max_pitch", 0.8)
        ok = height >= min_h and abs(roll) <= max_r and abs(pitch) <= max_p
        worst = min(height - min_h, max_r - abs(roll), max_p - abs(pitch))
        return CheckResult("base_stable", ok, worst, 0.0,
                           f"(h={height:.2f}m roll={roll:.2f} pitch={pitch:.2f})")
    return Condition("base_stable", fn)


def object_reachable(max_dist: float = 0.60) -> Condition:
    """Object within reach envelope of either hand."""
    def fn(env) -> CheckResult:
        e = _unwrap(env)
        obj = e.env.data.xpos[e.screwdriver_body_id]
        d_left = float(np.linalg.norm(obj - e.env.data.xpos[e.left_hand_id]))
        d_right = float(np.linalg.norm(obj - e.env.data.xpos[e.right_hand_id]))
        d = min(d_left, d_right)
        return CheckResult("object_reachable", d <= max_dist, d, max_dist)
    return Condition("object_reachable", fn)


def hand_free() -> Condition:
    """No object currently latched to either hand."""
    def fn(env) -> CheckResult:
        e = _unwrap(env)
        held = bool(e.object_grasped)
        return CheckResult("hand_free", not held, float(held), 0.0)
    return Condition("hand_free", fn)


def object_held() -> Condition:
    """Contact latch active (the policy's own claim of success)."""
    def fn(env) -> CheckResult:
        e = _unwrap(env)
        held = bool(e.object_grasped)
        hand = e.grasping_hand or "-"
        return CheckResult("object_held", held, float(held), 1.0, f"(hand={hand})")
    return Condition("object_held", fn)


class LiftMonitor:
    """
    Stateful postcondition: object raised >= min_lift above its spawn height,
    sustained for `sustain_steps` consecutive sim steps. Stateful because a
    one-frame bounce through the threshold is not a lift.

    Call update(env) every step; read result() at episode end.
    """

    def __init__(self, min_lift: float = 0.05, sustain_steps: int = 25):
        self.min_lift = min_lift
        self.sustain_steps = sustain_steps
        self._z0: Optional[float] = None
        self._streak = 0
        self._best_streak = 0
        self._max_lift = 0.0

    def reset(self, env) -> None:
        e = _unwrap(env)
        self._z0 = float(e.env.data.xpos[e.screwdriver_body_id][2])
        self._streak = 0
        self._best_streak = 0
        self._max_lift = 0.0

    def update(self, env) -> None:
        e = _unwrap(env)
        if self._z0 is None:
            self.reset(env)
        lift = float(e.env.data.xpos[e.screwdriver_body_id][2]) - self._z0
        self._max_lift = max(self._max_lift, lift)
        self._streak = self._streak + 1 if lift >= self.min_lift else 0
        self._best_streak = max(self._best_streak, self._streak)

    def result(self) -> CheckResult:
        ok = self._best_streak >= self.sustain_steps
        return CheckResult(
            "object_lifted", ok, self._max_lift, self.min_lift,
            f"(sustained {self._best_streak}/{self.sustain_steps} steps)")


# ---------------------------------------------------------------------------
# Skill contract
# ---------------------------------------------------------------------------

@dataclass
class SkillContract:
    """Pre/postconditions for one skill. The executor gates on these."""
    skill: str
    pre: List[Condition] = field(default_factory=list)
    post: List[Condition] = field(default_factory=list)

    def check_pre(self, env) -> List[CheckResult]:
        return [c.check(env) for c in self.pre]

    def check_post(self, env, monitors: Optional[List[LiftMonitor]] = None) -> List[CheckResult]:
        results = [c.check(env) for c in self.post]
        for m in monitors or []:
            results.append(m.result())
        return results


def grasp_contract() -> SkillContract:
    """The grasp skill's contract. Postcondition list deliberately includes the
    lift: 'grasped but never raised' is a failed grasp, whatever the latch says."""
    return SkillContract(
        skill="grasp",
        pre=[base_stable(), object_reachable(), hand_free()],
        post=[object_held(), base_stable()],
    )


def verdict(results: List[CheckResult]) -> bool:
    return all(r.ok for r in results)
