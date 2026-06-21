from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from rich.console import Console

from src.data.schemas import Plan, SKILLS

console = Console()

# Global cache for the model and tokenizer
_model = None
_tokenizer = None


def _repair_json(text: str) -> Optional[dict]:
    """Attempt minimal brace/quote repairs to recover a JSON object."""
    # Extract first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    cand = m.group(0)
    # Replace single quotes with double quotes conservatively
    cand = re.sub(r"'", '"', cand)
    # Remove trailing commas
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None


def parse_plan_from_model_output(text: str) -> Plan:
    """Parse model text into Plan. Repair once if invalid."""
    try:
        obj = json.loads(text)
        console.log(f"[green]✓ Direct JSON parse succeeded[/green]")
    except Exception as e1:
        console.log(f"[yellow]Direct parse failed: {e1}[/yellow]")
        console.log(f"[dim]Raw text: {text[:200]}...[/dim]")
        obj = _repair_json(text)
        if obj:
            console.log(f"[green]✓ Repair succeeded[/green]")
        else:
            console.log(f"[yellow]Repair failed, trying to salvage skills from text[/yellow]")
            salvaged_steps = _extract_skills_from_text(text)
            if not salvaged_steps:
                console.log(f"[yellow]Salvage empty, wrapping into notify-only plan[/yellow]")
                salvaged_steps = [{"skill": "notify", "args": {"level": "tech"}}]
            obj = {
                "goal": "Resolve incident",
                "steps": salvaged_steps,
                "fallback": [],
            }
    
    if not isinstance(obj, dict):
        console.log(f"[red]✗ obj is not dict: {type(obj)} = {obj}[/red]")
        # Fallback heuristic minimal plan
        obj = {"goal": "Resolve incident", "steps": [{"skill": "notify", "args": {"level": "tech"}}], "fallback": []}
    
    try:
        plan = Plan.model_validate(obj)
    except Exception as e2:
        console.log(f"[red]✗ Plan.model_validate failed: {e2}[/red]")
        console.log(f"[dim]obj was: {obj}[/dim]")
        raise
    
    # Enforce skill whitelist
    for step in plan.steps:
        if step.skill not in SKILLS:
            step.skill = "notify"
            step.args = {"level": "tech"}
    return plan


def _extract_skills_from_text(text: str) -> List[dict]:
    """Heuristic salvage: extract skills from free-form model text."""
    lower = text.lower()
    steps: List[dict] = []
    if any(k in lower for k in ["walk", "go", "move", "proceed", "navigate"]):
        steps.append({"skill": "walk_to", "args": {"target": "machine"}})
    if any(k in lower for k in ["sensor", "read", "check", "monitor", "gauge", "pressure", "temperature", "vibration", "light"]):
        sensor = "pressure_sensor"
        if "temperature" in lower:
            sensor = "temperature_sensor"
        elif "vibration" in lower:
            sensor = "vibration_sensor"
        elif "light" in lower:
            sensor = "light_sensor"
        steps.append({"skill": "read_sensor", "args": {"sensor": sensor}})
    if any(k in lower for k in ["press", "button", "switch", "push"]):
        button = "green_button"
        if "red" in lower:
            button = "red_button"
        elif "blue" in lower:
            button = "blue_button"
        elif "yellow" in lower:
            button = "yellow_button"
        steps.append({"skill": "press_button", "args": {"button": button}})
    if any(k in lower for k in ["pick", "grab", "take", "collect"]):
        steps.append({"skill": "pick", "args": {"object": "tool"}})
    if any(k in lower for k in ["place", "put", "set", "return"]):
        steps.append({"skill": "place", "args": {"object": "tool", "location": "table"}})
    if "wait" in lower:
        steps.append({"skill": "wait", "args": {"seconds": 3}})
    if not steps:
        steps.append({"skill": "notify", "args": {"level": "tech"}})
    elif steps[-1].get("skill") != "notify":
        steps.append({"skill": "notify", "args": {"level": "tech"}})
    return steps


def _build_prompt(incident: str, sop_title: str, sop_cond: str, sop_steps: List[str], skills: List[str]) -> str:
    """Structured instruction prompt to force valid JSON output."""
    steps = " ; ".join(sop_steps)
    skills_csv = ",".join(skills)
    return (
        "You are a planner that outputs ONLY valid JSON. Do not add explanations.\n"
        "Use exactly these keys: goal (string), steps (list), fallback (list).\n"
        f"Each step must have: skill (one of [{skills_csv}]) and args (object).\n"
        "Respond with a single JSON object.\n\n"
        "[INCIDENT]\n"
        f"{incident}\n"
        "[SOP]\n"
        f"{sop_title}\n{sop_cond}\n{steps}\n"
        "[ALLOWED_SKILLS]\n"
        f"{skills_csv}\n\n"
        "Return JSON exactly in this format (no extra text):\n"
        "{\n"
        "  \"goal\": \"<sop title>\",\n"
        "  \"steps\": [\n"
        "    {\"skill\": \"walk_to\", \"args\": {\"target\": \"machine\"}},\n"
        "    {\"skill\": \"notify\", \"args\": {\"level\": \"tech\"}}\n"
        "  ],\n"
        "  \"fallback\": []\n"
        "}\n"
        "Now fill with the best plan."
    )


def _extract_skills_from_sop(sop_steps: List[str], skills: List[str]) -> List[dict]:
    """Extract and map SOP steps to robot skills using semantic keywords."""
    extracted = []
    step_text = " ".join(sop_steps).lower()
    
    # Keywords for each skill (semantic matching) - Updated for manufacturing environment
    skill_keywords = {
        "walk_to": ["walk", "go", "move", "proceed", "head to", "navigate", "travel"],
        "read_sensor": ["read", "check", "monitor", "observe", "measure", "indicator", "sensor", "display", "gauge", "verify", "confirm"],
        "press_button": ["press", "push", "click", "activate", "button", "switch", "trigger"],
        "pick": ["pick", "grab", "take", "collect", "retrieve", "hold"],
        "place": ["place", "put", "set", "deposit", "position", "return", "replace"],
        "wait": ["wait", "pause", "delay", "seconds"],
        "notify": ["notify", "report", "alert", "inform", "communicate", "log", "technician"],
    }
    
    # Track which skills we've already added (avoid duplicates)
    added_skills = set()
    
    # Scan through steps and match keywords
    for step in sop_steps:
        step_lower = step.lower()
        
        for skill, keywords in skill_keywords.items():
            # Only add if skill exists, not already added, and keyword matches
            if skill in skills and skill not in added_skills:
                if any(kw in step_lower for kw in keywords):
                    # Extract contextual args
                    args = _extract_args_for_skill(skill, step)
                    extracted.append({"skill": skill, "args": args})
                    added_skills.add(skill)
                    break  # Move to next step after matching one skill
    
    return extracted


def _extract_args_for_skill(skill: str, step_text: str) -> dict:
    """Extract arguments/parameters for a skill from the step description."""
    step_lower = step_text.lower()
    
    if skill == "walk_to":
        # Try to extract target location
        if "table" in step_lower:
            return {"target": "table"}
        elif "machine" in step_lower:
            return {"target": "machine"}
        elif "shelf" in step_lower:
            return {"target": "shelf"}
        else:
            return {"target": "machine"}  # Default
    
    elif skill == "read_sensor":
        # Try to extract sensor type - Updated for environment sensors
        if "pressure" in step_lower:
            return {"sensor": "pressure_sensor"}
        elif "temperature" in step_lower:
            return {"sensor": "temperature_sensor"}
        elif "vibration" in step_lower:
            return {"sensor": "vibration_sensor"}
        elif "light" in step_lower and "sensor" in step_lower:
            return {"sensor": "light_sensor"}
        else:
            return {"sensor": "pressure_sensor"}  # Default
    
    elif skill == "press_button":
        # Try to extract button type - Updated for environment buttons
        if "blue" in step_lower:
            return {"button": "blue_button"}
        elif "green" in step_lower:
            return {"button": "green_button"}
        elif "red" in step_lower:
            return {"button": "red_button"}
        elif "yellow" in step_lower:
            return {"button": "yellow_button"}
        else:
            return {"button": "green_button"}  # Default
    
    elif skill == "pick":
        # Try to extract object - Updated for table objects
        if "wrench" in step_lower:
            return {"object": "wrench"}
        elif "screwdriver" in step_lower:
            return {"object": "screwdriver"}
        elif "battery" in step_lower:
            return {"object": "battery"}
        elif "lubricant" in step_lower:
            return {"object": "lubricant_bottle"}
        elif "intake" in step_lower or "cover" in step_lower:
            return {"object": "intake_cover"}
        elif "cloth" in step_lower or "cleaning_cloth" in step_lower:
            return {"object": "cleaning_cloth"}
        elif "fuse" in step_lower:
            return {"object": "spare_fuses"}
        elif "brush" in step_lower:
            return {"object": "cleaning_brush"}
        elif "goggles" in step_lower or "safety" in step_lower:
            return {"object": "safety_goggles"}
        elif "cable" in step_lower or "ties" in step_lower:
            return {"object": "cable_ties"}
        elif "container" in step_lower or "bin" in step_lower:
            return {"object": "container_bin"}
        else:
            return {"object": "tool"}  # Default
    
    elif skill == "place":
        # Keep the REAL object being placed (reuse pick's object extraction) + the location.
        obj = _extract_args_for_skill("pick", step_text).get("object", "tool")
        if "machine" in step_lower:   location = "machine"
        elif "shelf" in step_lower:   location = "machine_shelf"
        elif "table" in step_lower:   location = "table"
        elif "bin" in step_lower or "container" in step_lower: location = "container_bin"
        else:                          location = "table"
        return {"object": obj, "location": location}
    
    elif skill == "wait":
        # Parse the actual duration from the step ("Wait 5 seconds" -> 5; minutes -> *60).
        m = re.search(r"(\d+)\s*(second|sec|minute|min)", step_lower)
        if m:
            n = int(m.group(1))
            return {"seconds": n * 60 if m.group(2).startswith("min") else n}
        return {"seconds": 3}
    
    elif skill == "notify":
        # Extract notification level
        if "manager" in step_lower or "supervisor" in step_lower:
            return {"level": "manager"}
        elif "tech" in step_lower or "technician" in step_lower:
            return {"level": "tech"}
        else:
            return {"level": "tech"}  # Default
    
    return {}


# ======================================================================================
# FAITHFUL planner  (canonical: the plan IS the SOP, mapped 1:1 to robot skills)
# ======================================================================================
# Keyword evidence per skill. Multi-word/longer phrases are more specific, so we score by
# the matched phrase length -- "pick up" beats a stray "press", "re-read" maps to read, etc.
SKILL_KEYWORDS = {
    "walk_to":      ["walk to", "walk", "go to", "move to", "proceed to", "head to", "navigate", "travel", "approach"],
    "pick":         ["pick up", "pick", "grab", "take", "collect", "retrieve", "remove", "lift up"],
    "place":        ["place", "put down", "put", "set down", "deposit", "position", "return", "replace", "install", "mount", "insert"],
    "press_button": ["press", "push", "click", "activate", "trigger", "hold down"],
    "read_sensor":  ["re-read", "read", "check", "monitor", "observe", "measure", "inspect", "verify", "confirm", "gauge", "indicator", "display", "look at"],
    "wait":         ["wait", "pause", "delay", "hold for", "allow", "let it"],
    "notify":       ["notify", "report", "alert", "inform", "log", "record", "escalate", "call", "document"],
}


def _classify_step(step: str) -> tuple:
    """Map ONE SOP step (a natural-language imperative) to (skill, args), independently.
    Scores each skill by matched-keyword specificity (phrase length) and picks the strongest;
    a step with no action evidence falls back to `notify` (a safe operational step). Running
    this PER STEP is what makes the plan a faithful 1:1 translation of the SOP."""
    sl = step.lower()
    # WORD-BOUNDARY match so "press" doesn't fire on "pressure", "read" not on "ready", etc.
    scores = {sk: max([len(kw) for kw in kws if re.search(r"\b" + re.escape(kw) + r"\b", sl)] or [0])
              for sk, kws in SKILL_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "notify"
    return best, _extract_args_for_skill(best, step)


def plan_from_sop(sop_steps: List[str]) -> List[dict]:
    """FAITHFUL planner — the canonical SOP→plan mapping.

    Translate a retrieved SOP into an executable skill chain where EACH SOP step becomes
    exactly one plan step, IN ORDER, with repeats preserved. The plan therefore *is* the SOP
    expressed in robot skills, so the verifying executor walks the SOP step-by-step. Every
    plan step carries its source `sop_step` so execution is fully traceable/auditable.

    (Contrast with the legacy `_extract_skills_from_sop`, which deduped skills and dropped
    repeated steps — it did not faithfully follow the SOP.)
    """
    plan: List[dict] = []
    for i, step in enumerate(sop_steps):
        skill, args = _classify_step(step)
        plan.append({"skill": skill, "args": args, "sop_step": step.strip(), "index": i})
    return plan


def _load_lora_model():
    """Load the LoRA-adapted Flan-T5 model (cached globally)."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        adapter_path = project_root / "artifacts" / "planner_lora" / "adapter"
        tokenizer_path = project_root / "artifacts" / "planner_lora" / "tokenizer"
        
        if not adapter_path.exists():
            console.log(f"[yellow]LoRA adapter not found at {adapter_path}, falling back to heuristic[/yellow]")
            return None, None
        
        console.log(f"[cyan]Loading LoRA model from {adapter_path}...[/cyan]")
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        # Load LoRA adapter
        _model = PeftModel.from_pretrained(base_model, str(adapter_path))
        _model.eval()
        
        console.log(f"[green]✓ LoRA model loaded successfully[/green]")
        return _model, _tokenizer
        
    except Exception as e:
        console.log(f"[yellow]Failed to load LoRA model: {e}[/yellow]")
        console.log(f"[yellow]Falling back to heuristic planner[/yellow]")
        return None, None


def _generate_with_lora(prompt: str, model, tokenizer) -> str:
    """Generate plan JSON using the LoRA-adapted model."""
    try:
        import torch
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.05,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        console.log(f"[red]Generation failed: {e}[/red]")
        raise


def plan(incident: str, sop: dict, skills: List[str] = SKILLS) -> Plan:
    """Generate a plan from (incident, sop) using LoRA-adapted Flan-T5."""
    console.log(f"[dim]Planning for: {sop['title']}[/dim]")
    
    # Try to use LoRA model first
    model, tokenizer = _load_lora_model()
    
    if model is not None and tokenizer is not None:
        # Use LoRA model for generation
        prompt = _build_prompt(
            incident=incident,
            sop_title=sop["title"],
            sop_cond=sop.get("condition", ""),
            sop_steps=sop.get("steps", []),
            skills=skills
        )
        
        console.log(f"[cyan]Using LoRA model for generation...[/cyan]")
        generated_text = _generate_with_lora(prompt, model, tokenizer)
        console.log(f"[dim]Generated: {generated_text[:200]}...[/dim]")
        
        # Parse the generated JSON
        plan_obj = parse_plan_from_model_output(generated_text)

        # If the generated plan is too short, augment with heuristic skills from SOP
        if len(plan_obj.steps) < 3:
            heuristic_steps = _extract_skills_from_sop(sop.get("steps", []), skills)
            merged = []
            seen = set()
            # Keep generated steps first
            for st in plan_obj.steps:
                if st.skill not in seen:
                    merged.append(st)
                    seen.add(st.skill)
            # Add missing heuristic steps
            for hs in heuristic_steps:
                if hs["skill"] not in seen:
                    merged.append(Plan.model_validate({"goal": plan_obj.goal, "steps": [hs], "fallback": []}).steps[0])
                    seen.add(hs["skill"])
            # Ensure notify at end
            merged = [s for s in merged if s.skill != "notify"] + [Plan.model_validate({"goal": plan_obj.goal, "steps": [{"skill": "notify", "args": {"level": "tech"}}], "fallback": []}).steps[0]]
            plan_obj = Plan(goal=plan_obj.goal, steps=merged, fallback=[])
        
    else:
        # Fallback to heuristic planner
        console.log(f"[yellow]Using heuristic fallback planner[/yellow]")
        goal = sop["title"]
        steps = _extract_skills_from_sop(sop.get("steps", []), skills)
        
        # Always end with notify if not already present
        if not steps or steps[-1]["skill"] != "notify":
            steps.append({"skill": "notify", "args": {"level": "tech"}})
        
        # Ensure at least one action before notify
        if len(steps) == 1:
            steps.insert(0, {"skill": "walk_to", "args": {"target": "machine"}})
        
        plan_obj = Plan(goal=goal, steps=steps, fallback=[])
    
    console.log(f"[green]✓ Generated plan: {len(plan_obj.steps)} steps[/green]")
    for i, step in enumerate(plan_obj.steps, 1):
        console.log(f"[dim]  {i}. {step.skill} → {step.args}[/dim]")
    
    return plan_obj


