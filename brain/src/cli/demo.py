from __future__ import annotations

import json
import sys
from pathlib import Path
import typer
from rich.console import Console

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.plan_pipeline import run_pipeline
from src.pipeline.skills import DummyMuJoCoAdapter, execute_plan
from src.retrieval.infer_retrieve import retrieve_topk
from src.data.schemas import SOPEntry, load_json
from src.retrieval.index_utils import build_and_save_index

app = typer.Typer(help="SOP → Planner → MuJoCo Executor demo CLI")
console = Console()


@app.command()
def build_index():
    """
    Build the search index for fast SOP retrieval.
    
    This embeds all SOPs using the trained BERT encoder and builds a FAISS index
    for fast similarity search. Run this after training the retriever.
    """
    project_root = Path(__file__).parent.parent.parent
    sops = load_json(str(project_root / "src" / "data" / "sop_examples.json"), SOPEntry, key="sop_examples")
    texts = [f"{s.title}. {s.condition}. Steps: " + " ; ".join(s.steps) for s in sops]
    ids = [s.sop_id for s in sops]
    build_and_save_index(texts, ids, str(project_root / "artifacts" / "retriever_bert" / "index"))
    console.log(f"[green]✓ Index built with {len(sops)} SOPs[/green]")


@app.command()
def retrieve(q: str = typer.Option(..., "--q", help="Incident text"), k: int = 5):
    """
    Retrieve the top-k most relevant SOPs for an incident.
    
    Uses semantic similarity search to find SOPs that match the incident description.
    Results are ranked by similarity score (higher = more relevant).
    """
    hits = retrieve_topk(q, k=k)
    for h in hits:
        console.print(h)


@app.command()
def plan(q: str = typer.Option(..., "--q", help="Incident text")):
    """
    Run the full pipeline: retrieve SOP → generate plan.
    
    Takes an incident description, finds the relevant SOP, and generates
    a structured JSON plan with robot skills.
    """
    p = run_pipeline(q, k=5)
    console.print_json(data=json.loads(p.model_dump_json()))


@app.command()
def exec(q: str = typer.Option(..., "--q", help="Incident text")):
    """
    Run the full pipeline with mock execution.
    
    Same as `plan`, but also simulates executing the plan using a dummy
    robot adapter. This lets you test the plan without a real robot.
    """
    p = run_pipeline(q, k=5)
    api = DummyMuJoCoAdapter()
    summary = execute_plan(p, api)
    console.print_json(data={"plan": json.loads(p.model_dump_json()), "summary": summary})


if __name__ == "__main__":
    app()


