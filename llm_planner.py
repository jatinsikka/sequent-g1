"""
LLM planner (path A): given an incident, a frontier LLM selects the right SOP
from retrieved candidates and emits a typed, executable Plan.

This replaces brain_bridge's deterministic step->skill mapper with real
reasoning: the model diagnoses which SOP applies and translates its procedure
into clean, well-argumented skills. Retrieval still grounds it (TF-IDF top-k
from brain_bridge), so the LLM chooses among real candidates rather than
hallucinating procedures.

Provider: Gemini REST (free tier) via `requests` — no SDK needed. Swappable;
the only provider-specific surface is `_call_gemini`. Reads GEMINI_API_KEY from
the environment; model is GEMINI_MODEL (default below) or the --model arg.

    python llm_planner.py --list-models          # see what your key can use
    python llm_planner.py --command "..."        # dry-run: print the plan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan, SKILLS  # noqa: E402

from brain_bridge import retrieve_sop, _load_sops

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
_BASE = "https://generativelanguage.googleapis.com/v1beta"

SKILL_SPEC = """\
walk_to      {"target": str}     navigate to a machine/table/shelf/station
press_button {"button": str}     e.g. red_button, green_button, blue_button, yellow_button
wait         {"sec": number}     pause for N seconds
read_sensor  {"sensor": str}     e.g. pressure_sensor, temperature_sensor, light_sensor
pick         {"obj": str}        pick up an object (the only skill physically executed today)
place        {"target": str}     place the held object at a location
notify       {"level": str}      alert a human (tech / technician / operator)"""


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        # fallback: a gitignored key file next to this module (see .gitignore)
        kf = os.path.join(os.path.dirname(__file__), ".gemini_key")
        if os.path.exists(kf):
            with open(kf) as f:
                key = f.read().strip()
    if not key:
        raise RuntimeError("No Gemini key - get a free one at https://aistudio.google.com/apikey, "
                           "then either `setx GEMINI_API_KEY \"...\"` or paste it into a file "
                           "named .gemini_key in this folder (gitignored).")
    return key


def list_models() -> List[str]:
    r = requests.get(f"{_BASE}/models", params={"key": _api_key()}, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].replace("models/", ""))
    return out


def _call_gemini(prompt: str, model: str) -> str:
    url = f"{_BASE}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    r = requests.post(url, params={"key": _api_key()}, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {model} HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _prompt(incident: str, candidates: List[dict]) -> str:
    cand = "\n".join(
        f"- {s['sop_id']} | {s['title']} | condition: {s['condition']}\n    steps: {s['steps']}"
        for s in candidates
    )
    return f"""You are the planning module of a humanoid robot that executes factory Standard \
Operating Procedures. Given an incident report, pick the SINGLE most relevant SOP from the \
candidates and translate its procedure into an ordered plan of robot skills.

INCIDENT: {incident}

CANDIDATE SOPs (retrieved):
{cand}

ALLOWED SKILLS (use ONLY these; args must match exactly):
{SKILL_SPEC}

Return STRICT JSON, no prose, matching:
{{"sop_id": "<chosen id>", "goal": "<one line>", "steps": [{{"skill": "<one of {SKILLS}>", "args": {{...}}}}]}}

Rules: choose exactly one SOP; every step's skill must be in the allowed list; give concrete \
args (real button/sensor/object names from the SOP steps); preserve the SOP's order."""


def plan_with_llm(incident: str, model: str = DEFAULT_MODEL, k: int = 5,
                  max_retries: int = 1) -> Tuple[Plan, str]:
    sops = _load_sops()
    candidates = [s for s, _ in retrieve_sop(incident, sops, k=k)]
    prompt = _prompt(incident, candidates)
    last_err = ""
    for attempt in range(max_retries + 1):
        raw = _call_gemini(prompt if attempt == 0 else
                           prompt + f"\n\nYour previous output was invalid: {last_err}. Return valid JSON only.",
                           model)
        try:
            obj = json.loads(raw)
            bad = [s for s in obj.get("steps", []) if s.get("skill") not in SKILLS]
            if bad:
                raise ValueError(f"steps use unknown skills: {[s.get('skill') for s in bad]}")
            plan = Plan(goal=f"{obj.get('goal', incident)}  [SOP {obj.get('sop_id', '?')}, via {model}]",
                        steps=obj["steps"])
            return plan, obj.get("sop_id", "?")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_err = str(e)
    raise RuntimeError(f"LLM planner failed to produce a valid plan after retries: {last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()
    if args.list_models:
        for m in list_models():
            print(m)
        return
    if not args.command:
        raise SystemExit("provide --command or --list-models")
    plan, sop_id = plan_with_llm(args.command, model=args.model)
    print(f"chosen SOP: {sop_id}")
    print(json.dumps(plan.model_dump(), indent=2))


if __name__ == "__main__":
    main()
