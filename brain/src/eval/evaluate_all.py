from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console

from src.data.schemas import IncidentEntry, SOPEntry, load_json, Plan, PlanStep
from src.eval.metrics import execution_success_rate, json_plan_f1, mrr_score, retrieval_recall_at_k
from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan

console = Console()


def main() -> None:
    """
    Comprehensive evaluation of the entire system.
    
    Evaluates both components:
    1. Retrieval performance: How well does the retriever find the correct SOP?
    2. Plan generation quality: How well does the planner generate executable plans?
    """
    sops = load_json("src/data/sop_examples.json", SOPEntry, key="sop_examples")
    incidents = load_json("src/data/incident_examples.json", IncidentEntry, key="incident_examples")

    # Evaluate retrieval component
    # For each incident, check if the correct SOP appears in top-k results
    golds = [i.labels.get("sop_id", "") if i.labels else "" for i in incidents]
    ranked_ids: List[List[str]] = []
    from src.retrieval.infer_retrieve import retrieve_topk

    console.log("[bold]Evaluating retrieval...[/bold]")
    for inc in incidents:
        hits = retrieve_topk(inc.text, k=5)
        ranked_ids.append([str(h["sop_id"]) for h in hits])

    r1 = retrieval_recall_at_k(golds, ranked_ids, k=1)
    r5 = retrieval_recall_at_k(golds, ranked_ids, k=5)
    mrr = mrr_score(golds, ranked_ids)
    console.log(f"Retrieval: Recall@1={r1:.2f}, Recall@5={r5:.2f}, MRR={mrr:.2f}")

    # Evaluate plan generation component
    # Compare generated plans against reference plans to compute F1 score
    console.log("[bold]Evaluating plan generation...[/bold]")
    refs = {
        "INC-001": Plan(
            goal="Low Pressure Warning",
            steps=[
                PlanStep(skill="walk_to", args={"target": "machine"}),
                PlanStep(skill="read_sensor", args={"sensor": "pressure_sensor"}),
                PlanStep(skill="press_button", args={"button": "blue_button"}),
                PlanStep(skill="notify", args={"level": "tech"})
            ],
            fallback=[]
        ),
        "INC-002": Plan(
            goal="Emergency Shutdown Procedure",
            steps=[
                PlanStep(skill="walk_to", args={"target": "machine"}),
                PlanStep(skill="press_button", args={"button": "red_button"}),
                PlanStep(skill="read_sensor", args={"sensor": "temperature_sensor"}),
                PlanStep(skill="notify", args={"level": "tech"})
            ],
            fallback=[]
        ),
        "INC-003": Plan(
            goal="Machine Startup Sequence",
            steps=[
                PlanStep(skill="walk_to", args={"target": "machine"}),
                PlanStep(skill="press_button", args={"button": "green_button"}),
                PlanStep(skill="read_sensor", args={"sensor": "pressure_sensor"}),
                PlanStep(skill="notify", args={"level": "tech"})
            ],
            fallback=[]
        ),
    }
    # Evaluate plan quality by comparing against reference plans
    # F1 score measures how well generated plans match ground truth
    f1s: List[float] = []
    exec_summaries = []
    api = DummyMuJoCoAdapter()
    
    for inc in incidents:
        # Generate plan for this incident
        p = run_pipeline(inc.text, k=5)
        
        # Compare with reference plan if available
        if inc.incident_id in refs:
            f1s.append(json_plan_f1(refs[inc.incident_id], p))
        
        # Test if the plan can be executed (safety check)
        exec_summaries.append(execute_plan(p, api))
    
    # Report results
    avg_f1 = sum(f1s) / max(1, len(f1s))
    console.log(f"Plan JSON F1 (on {len(refs)} reference plans): {avg_f1:.2f}")
    console.log(f"Execution success rate: {execution_success_rate(exec_summaries):.2f}")


if __name__ == "__main__":
    main()


