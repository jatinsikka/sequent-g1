"""
Compare Ollama RAG retrieval performance with BERT dual-encoder baseline.

This script evaluates both retrieval systems on the same test set and compares:
- Recall@1, Recall@5, MRR
- Latency (time per query)
- Overall performance metrics

Usage:
    python src/eval/compare_ollama_rag.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.schemas import IncidentEntry, load_json
from src.eval.metrics import mrr_score, retrieval_recall_at_k
from src.retrieval.infer_retrieve import retrieve_topk as bert_retrieve_topk
from src.retrieval.ollama_rag import OllamaRAGRetriever

console = Console()


def evaluate_retriever(
    retriever_name: str,
    retrieve_fn,
    incidents: List[IncidentEntry],
    k: int = 5,
) -> dict:
    """
    Evaluate a retriever on test incidents.
    
    Args:
        retriever_name: Name of the retriever (for display)
        retrieve_fn: Function that takes incident_text and k, returns list of dicts with 'sop_id'
        incidents: List of test incidents
        k: Number of top results to retrieve
        
    Returns:
        Dictionary with metrics: recall@1, recall@5, mrr, latency_ms
    """
    console.log(f"[dim]Evaluating {retriever_name}...[/dim]")
    
    golds = [inc.labels.get("sop_id", "") if inc.labels else "" for inc in incidents]
    ranked_ids: List[List[str]] = []
    
    # Measure latency
    start_time = time.time()
    
    for i, inc in enumerate(incidents):
        try:
            hits = retrieve_fn(inc.text, k=k)
            ranked_ids.append([str(h["sop_id"]) for h in hits])
        except Exception as e:
            console.log(f"[red]Error retrieving for incident {i+1}: {e}[/red]")
            ranked_ids.append([])
    
    elapsed_time = time.time() - start_time
    latency_ms = (elapsed_time / len(incidents)) * 1000
    
    # Compute metrics
    recall_1 = retrieval_recall_at_k(golds, ranked_ids, k=1)
    recall_5 = retrieval_recall_at_k(golds, ranked_ids, k=5)
    mrr = mrr_score(golds, ranked_ids)
    
    return {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "mrr": mrr,
        "latency_ms": latency_ms,
    }


def main() -> None:
    """Main evaluation function."""
    console.print("[bold]Ollama RAG vs BERT Dual-Encoder Comparison[/bold]\n")
    
    # Load test data
    console.log("[dim]Loading test data...[/dim]")
    incidents = load_json("src/data/incident_examples.json", IncidentEntry, key="incident_examples")
    console.log(f"[green]Loaded {len(incidents)} test incidents[/green]\n")
    
    # Evaluate BERT dual-encoder (baseline)
    console.print("[bold]1. Evaluating BERT Dual-Encoder (Baseline)[/bold]")
    bert_metrics = evaluate_retriever(
        "BERT Dual-Encoder",
        bert_retrieve_topk,
        incidents,
        k=5,
    )
    
    # Evaluate Ollama RAG
    console.print("\n[bold]2. Evaluating Ollama RAG[/bold]")
    try:
        # Use nomic-embed-text which is a proper embedding model
        ollama_retriever = OllamaRAGRetriever(embedding_model="nomic-embed-text")
        ollama_metrics = evaluate_retriever(
            "Ollama RAG",
            lambda text, k: ollama_retriever.retrieve_topk(text, k=k),
            incidents,
            k=5,
        )
    except Exception as e:
        console.log(f"[red]Error initializing Ollama RAG: {e}[/red]")
        console.log("[yellow]Make sure Ollama is running and the embedding model is available[/yellow]")
        console.log("[yellow]You can start Ollama with: ollama serve[/yellow]")
        console.log("[yellow]And pull the model with: ollama pull nomic-embed-text[/yellow]")
        return
    
    # Display comparison table
    console.print("\n[bold]Performance Comparison[/bold]\n")
    
    table = Table(title="Retrieval Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("BERT Dual-Encoder", style="green", justify="right")
    table.add_column("Ollama RAG", style="yellow", justify="right")
    table.add_column("Difference", style="magenta", justify="right")
    
    # Recall@1
    diff_r1 = ollama_metrics["recall@1"] - bert_metrics["recall@1"]
    table.add_row(
        "Recall@1",
        f"{bert_metrics['recall@1']:.4f}",
        f"{ollama_metrics['recall@1']:.4f}",
        f"{diff_r1:+.4f}",
    )
    
    # Recall@5
    diff_r5 = ollama_metrics["recall@5"] - bert_metrics["recall@5"]
    table.add_row(
        "Recall@5",
        f"{bert_metrics['recall@5']:.4f}",
        f"{ollama_metrics['recall@5']:.4f}",
        f"{diff_r5:+.4f}",
    )
    
    # MRR
    diff_mrr = ollama_metrics["mrr"] - bert_metrics["mrr"]
    table.add_row(
        "MRR",
        f"{bert_metrics['mrr']:.4f}",
        f"{ollama_metrics['mrr']:.4f}",
        f"{diff_mrr:+.4f}",
    )
    
    # Latency
    diff_latency = ollama_metrics["latency_ms"] - bert_metrics["latency_ms"]
    table.add_row(
        "Latency (ms/query)",
        f"{bert_metrics['latency_ms']:.2f}",
        f"{ollama_metrics['latency_ms']:.2f}",
        f"{diff_latency:+.2f}",
    )
    
    console.print(table)
    
    # Summary
    console.print("\n[bold]Summary[/bold]")
    if ollama_metrics["recall@1"] > bert_metrics["recall@1"]:
        console.print("[green]✓ Ollama RAG outperforms BERT on Recall@1[/green]")
    elif ollama_metrics["recall@1"] < bert_metrics["recall@1"]:
        console.print("[yellow]⚠ BERT outperforms Ollama RAG on Recall@1[/yellow]")
    else:
        console.print("[dim]→ Both models achieve the same Recall@1[/dim]")
    
    if ollama_metrics["latency_ms"] < bert_metrics["latency_ms"]:
        console.print("[green]✓ Ollama RAG is faster than BERT[/green]")
    elif ollama_metrics["latency_ms"] > bert_metrics["latency_ms"]:
        console.print("[yellow]⚠ BERT is faster than Ollama RAG[/yellow]")
    else:
        console.print("[dim]→ Both models have similar latency[/dim]")


if __name__ == "__main__":
    main()

