"""
Verifying executor: runs a plan step-by-step, gating each step on physics-checked
pre/postconditions. The opposite of brain/src/pipeline/skills.py's stub, which
logs and returns True (silent success). Here a step only counts if the simulator
agrees it happened; otherwise the executor retries, then halts with a report.

v0 scope:
  - `pick`/`grasp` run the real RL grasp policy (v5.5) and verify against mjData.
  - other skills (walk_to, press_button, ...) are HONESTLY stubbed: marked
    `stubbed` in the report, never reported as `verified`. They get real runners
    as the skill library grows.

Each real skill runs as its own reset->rollout->verify episode for now; chaining
skills in one continuous sim is a v1 concern (noted, not hidden).

Author: Jatin Sikka
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from stable_baselines3 import PPO

from config import get_training_config
from verifier import grasp_contract, LiftMonitor, verdict, CheckResult
from verify_policy import make_env

# brain plan schema
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan, PlanStep  # noqa: E402

# Which plan skills currently have a real, verified runner.
REAL_SKILLS = {"pick", "grasp"}


@dataclass
class StepReport:
    index: int
    skill: str
    args: Dict[str, Any]
    status: str           # verified | precondition_failed | postcondition_failed | stubbed | error
    attempts: int
    pre: List[CheckResult] = field(default_factory=list)
    post: List[CheckResult] = field(default_factory=list)
    detail: str = ""

    def line(self) -> str:
        icon = {"verified": "[OK]", "stubbed": "[--]"}.get(self.status, "[XX]")
        a = f" (x{self.attempts})" if self.attempts > 1 else ""
        return f"  {icon} [{self.index}] {self.skill}({_fmt_args(self.args)}) - {self.status}{a}{(' | ' + self.detail) if self.detail else ''}"


@dataclass
class ExecReport:
    goal: str
    steps: List[StepReport]
    completed: bool

    def render(self) -> str:
        head = f"TASK {'COMPLETE' if self.completed else 'HALTED'} — {self.goal}"
        body = "\n".join(s.line() for s in self.steps)
        n_ver = sum(s.status == "verified" for s in self.steps)
        n_stub = sum(s.status == "stubbed" for s in self.steps)
        tail = f"  {n_ver} verified, {n_stub} stubbed, {len(self.steps)} planned"
        return f"{head}\n{body}\n{tail}"


def _fmt_args(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


class VerifyingExecutor:
    def __init__(self, grasp_model_path: str, device: Optional[str] = None,
                 max_retries: int = 2, min_lift: float = 0.05, sustain_steps: int = 25):
        """Load the grasp policy and sim env once, reused across all plan steps."""
        config = get_training_config()
        if device:
            config.device = device
        self.config = config
        self.env = make_env(config)
        self.env.unwrapped.no_early_success_stop = True  # give the lift check its runway
        self.grasp_model = PPO.load(grasp_model_path, device=config.device)
        self.max_retries = max_retries
        self.min_lift = min_lift
        self.sustain_steps = sustain_steps

    def run(self, plan: Plan) -> ExecReport:
        """Run each plan step in order; halt on the first step that isn't verified or stubbed."""
        reports: List[StepReport] = []
        completed = True
        for i, step in enumerate(plan.steps):
            if step.skill in REAL_SKILLS:
                r = self._run_grasp(i, step)
            else:
                r = StepReport(i, step.skill, step.args, "stubbed", 1,
                               detail="no verified runner yet")
            reports.append(r)
            if r.status not in ("verified", "stubbed"):
                completed = False
                break  # halt: never execute a step whose predecessor isn't done
        return ExecReport(plan.goal, reports, completed)

    def _run_grasp(self, index: int, step: PlanStep) -> StepReport:
        """Run the real grasp policy: check preconditions, roll out, measure the lift.
        Retry up to max_retries; if it never holds, report the failing postcondition."""
        contract = grasp_contract()
        attempts = 0
        last_pre: List[CheckResult] = []
        last_post: List[CheckResult] = []
        while attempts < self.max_retries + 1:
            attempts += 1
            obs, _ = self.env.reset()
            last_pre = contract.check_pre(self.env)
            if not verdict(last_pre):
                failed = "; ".join(str(r) for r in last_pre if not r.ok)
                return StepReport(index, step.skill, step.args, "precondition_failed",
                                  attempts, last_pre, [], failed)
            lift = LiftMonitor(min_lift=self.min_lift, sustain_steps=self.sustain_steps)
            lift.reset(self.env)
            done = False
            while not done:
                action, _ = self.grasp_model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = self.env.step(action)
                lift.update(self.env)
                done = term or trunc
            last_post = contract.check_post(self.env, monitors=[lift])
            if verdict(last_post):
                return StepReport(index, step.skill, step.args, "verified",
                                  attempts, last_pre, last_post,
                                  "lift held + stable")
        failed = "; ".join(str(r) for r in last_post if not r.ok)
        return StepReport(index, step.skill, step.args, "postcondition_failed",
                          attempts, last_pre, last_post, failed)
