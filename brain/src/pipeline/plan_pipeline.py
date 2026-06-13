from __future__ import annotations

from typing import List

from rich.console import Console

from src.data.schemas import Plan, SKILLS, SOPEntry, load_json
from src.retrieval.infer_retrieve import retrieve_topk
from src.planner.infer_plan import plan as planner_plan

console = Console()


def _get_sop_by_id(sops: List[SOPEntry], sop_id: str) -> dict:
    """Find an SOP by its ID and convert to a plain dictionary."""
    for s in sops:
        if s.sop_id == sop_id:
            return {
                "sop_id": s.sop_id,
                "title": s.title,
                "condition": s.condition,
                "steps": s.steps,
                "equipment": s.equipment,
            }
    raise KeyError(f"SOP not found: {sop_id}")


def _repair_text_to_json(text: str) -> str:
    """
    Quick fix for malformed JSON from model output.
    
    Sometimes the LLM cuts off mid-generation, leaving unbalanced braces.
    This adds missing closing braces to make it parseable.
    
    Note: This is a simple heuristic - for production, consider more robust parsing.
    """
    opens = text.count("{")
    closes = text.count("}")
    if opens > closes:
        text += "}" * (opens - closes)
    return text


def run_pipeline(incident_text: str, k: int = 5) -> Plan:
    """
    Run the full end-to-end pipeline: retrieve SOP → generate plan.
    
    This is the main user-facing function that:
    1. Takes a natural language incident description
    2. Retrieves the most relevant SOP using semantic search
    3. Generates a structured robot plan from that SOP
    
    Args:
        incident_text: Natural language description of the incident
        k: Number of top SOPs to retrieve (default: 5, we use the top one)
        
    Returns:
        A structured Plan object with robot skills and arguments
    """
    console.log("[bold]Running pipeline: retrieve → plan[/bold]")
    sops = load_json("src/data/sop_examples.json", SOPEntry, key="sop_examples")
    
    # Step 1: Semantic retrieval - find the most relevant SOP
    hits = retrieve_topk(incident_text, k=k)
    console.log(f"[dim]Retrieved top-{len(hits)} SOPs by dual-encoder[/dim]")
    for i, h in enumerate(hits[:3]):
        console.log(f"  {i+1}. {h['sop_id']}: {h['text'][:60]}... (score: {h['score']:.4f})")
    
    # Step 2: Use the top-ranked SOP (highest similarity score)
    best_id = hits[0]["sop_id"]
    sop = _get_sop_by_id(sops, str(best_id))
    console.log(f"[bold green]Selected SOP: {best_id} - {sop['title']}[/bold green]")
    console.log(f"[dim]Condition: {sop['condition']}[/dim]")
    
    # Step 3: Generate structured robot plan from the SOP
    plan = planner_plan(incident_text, sop, skills=SKILLS)
    return plan


