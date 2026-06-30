"""
LLM planner — language incident -> retrieved SOPs -> a frontier LLM picks the right SOP and
translates it into a typed, executable robot-skill plan (with the navigation human SOPs leave implicit).

WHY THIS EXISTS (the honest story, see docs/PLANNER_EVOLUTION.md):
  The original planner was built from scratch — a keyword classifier + a fine-tuned Flan-T5 (LoRA).
  It was brittle (exact-keyword matching didn't generalize to novel phrasings) and the small model
  couldn't reliably emit valid structured plans (the codebase is full of JSON-repair/salvage hacks).
  A frontier LLM does language->structured-plan robustly out of the box. So we stop reinventing NLP
  here and concentrate originality where the moat actually is: PHYSICS-VERIFIED skill execution.
  The LLM proposes; the verifier (executor.py) disposes — every physical step is checked against
  measured MuJoCo physics, not the planner's say-so.

PROVIDER-ABSTRACTED: Claude (preferred) when ANTHROPIC_API_KEY is set, else Gemini (GEMINI_API_KEY
or .gemini_key), else a deterministic keyword fallback (plan_from_sop) so the demo never hard-fails.

    python llm_planner.py --command "the machine is overheating, shut it down"   # dry-run: print the plan
    python llm_planner.py --command "..." --provider gemini                      # force a provider
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.data.schemas import Plan, SKILLS  # noqa: E402
from brain_bridge import retrieve_sop, _load_sops  # noqa: E402

DEFAULT_CLAUDE = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_GEMINI = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
HERE = os.path.dirname(os.path.abspath(__file__))

# Where the robot must physically BE to perform a skill — so we can prompt the LLM to insert the
# navigation that human SOPs omit ("walk to the machine" is obvious to a person, not to a robot).
SKILL_SPEC = """\
walk_to      {"target": str}     navigate to a location: machine | table | shelf | bin
press_button {"button": str}     press a button AT THE MACHINE: red_button|green_button|blue_button|yellow_button
read_sensor  {"sensor": str}     read a gauge AT THE MACHINE: pressure_sensor|temperature_sensor|vibration_sensor|light_sensor
pick         {"object": str}     pick an object up (from the table/shelf) — must be AT that location first
place        {"object": str, "location": str}   place a held object (requires a prior pick)
wait         {"seconds": number} pause for N seconds
notify       {"level": str}      escalate to a human: tech | manager"""


# ---------------------------------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------------------------------
def _anthropic_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY")


def _gemini_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        kf = os.path.join(HERE, ".gemini_key")
        if os.path.exists(kf):
            key = open(kf).read().strip()
    return key or None


def pick_provider(force: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return (provider, model). Claude preferred; Gemini if it's the only key; None -> keyword fallback."""
    if force == "claude":
        return ("claude", DEFAULT_CLAUDE) if _anthropic_key() else (None, None)
    if force == "gemini":
        return ("gemini", DEFAULT_GEMINI) if _gemini_key() else (None, None)
    if _anthropic_key():
        return ("claude", DEFAULT_CLAUDE)
    if _gemini_key():
        return ("gemini", DEFAULT_GEMINI)
    return (None, None)


def _call_claude(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_anthropic_key())
    msg = client.messages.create(
        model=model, max_tokens=1024, temperature=0.2,
        system="You are the planning module of a humanoid robot. Output ONLY a single JSON object, no prose.",
        messages=[{"role": "user", "content": prompt},
                  {"role": "assistant", "content": "{"}],   # prefill -> forces a JSON object
    )
    return "{" + msg.content[0].text


def _call_gemini(prompt: str, model: str) -> str:
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    r = requests.post(url, params={"key": _gemini_key()}, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {model} HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_llm(prompt: str, provider: str, model: str) -> str:
    return _call_claude(prompt, model) if provider == "claude" else _call_gemini(prompt, model)


# ---------------------------------------------------------------------------------------------------
# Prompt + planning
# ---------------------------------------------------------------------------------------------------
def _prompt(incident: str, candidates: List[dict]) -> str:
    cand = "\n".join(
        f"- {s['sop_id']} | {s['title']} | condition: {s['condition']}\n    steps: {s['steps']}"
        for s in candidates)
    return f"""Given a factory incident, pick the SINGLE most relevant SOP from the candidates and \
translate its procedure into an ordered plan of robot skills.

INCIDENT: {incident}

CANDIDATE SOPs (retrieved by semantic search):
{cand}

ALLOWED SKILLS (use ONLY these; args must match exactly):
{SKILL_SPEC}

RULES:
- Choose exactly one SOP and use its sop_id.
- Translate its steps IN ORDER; every step's skill must be in the allowed list with concrete args
  (real button/sensor/object names from the SOP text).
- INFER IMPLICIT NAVIGATION: human SOPs omit obvious moves. Before any step performed at a location
  (press_button/read_sensor are AT the machine; pick is AT the table/shelf; place is AT its location),
  insert a walk_to(target) FIRST if the robot is not already there. Do not repeat walk_to if already there.
- `place` requires the object to have been `pick`ed earlier; keep that dependency.
- End with a notify step.

Return STRICT JSON only:
{{"sop_id": "<id>", "goal": "<one line>", "steps": [{{"skill": "<allowed>", "args": {{...}}}}]}}"""


def _keyword_fallback(incident: str) -> Tuple[Plan, str]:
    """No LLM available: retrieve top-1 + the deterministic keyword planner (location-aware)."""
    from src.planner.infer_plan import plan_from_sop
    sops = _load_sops()
    sop, _ = retrieve_sop(incident, sops, k=1)[0]
    steps = [{"skill": s["skill"], "args": s["args"]} for s in plan_from_sop(sop["steps"])]
    return Plan(goal=f"{sop['title']}  [keyword fallback]", steps=steps), sop["sop_id"]


def plan_with_llm(incident: str, provider: Optional[str] = None, k: int = 5,
                  max_retries: int = 1) -> Tuple[Plan, str, str]:
    """Retrieve top-k SOPs, let the LLM pick one + emit a validated Plan. Falls back to the keyword
    planner if no key is configured or the LLM can't produce valid JSON. Returns (plan, sop_id, how)."""
    prov, model = pick_provider(provider)
    if prov is None:
        plan, sid = _keyword_fallback(incident)
        return plan, sid, "keyword-fallback"

    sops = _load_sops()
    candidates = [s for s, _ in retrieve_sop(incident, sops, k=k)]
    prompt = _prompt(incident, candidates)
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            raw = _call_llm(prompt if attempt == 0 else
                            prompt + f"\n\nPrevious output was invalid ({last_err}). Return valid JSON only.",
                            prov, model)
            obj = json.loads(raw if raw.lstrip().startswith("{") else re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            bad = [s.get("skill") for s in obj.get("steps", []) if s.get("skill") not in SKILLS]
            if bad:
                raise ValueError(f"unknown skills: {bad}")
            plan = Plan(goal=f"{obj.get('goal', incident)}  [SOP {obj.get('sop_id','?')}, via {prov}:{model}]",
                        steps=obj["steps"])
            return plan, obj.get("sop_id", "?"), f"{prov}:{model}"
        except Exception as e:
            last_err = str(e)[:160]
    # LLM failed after retries -> never hard-fail the demo
    plan, sid = _keyword_fallback(incident)
    return plan, sid, f"keyword-fallback (after {prov} error: {last_err})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command", required=True)
    p.add_argument("--provider", choices=["claude", "gemini"], default=None)
    a = p.parse_args()
    plan, sop_id, how = plan_with_llm(a.command, provider=a.provider)
    print(f"provider: {how}")
    print(f"chosen SOP: {sop_id}")
    print(json.dumps(plan.model_dump(), indent=2))


if __name__ == "__main__":
    main()
