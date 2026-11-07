from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import SOPEntry, load_json
from src.retrieval.index_utils import build_and_save_index, load_index, search

console = Console()


def _synthesize_text(sop: SOPEntry) -> str:
    return f"{sop.title}. {sop.condition}. Steps: " + " ; ".join(sop.steps)


def _keyword_boost(incident_text: str, sop_text: str) -> float:
    """Boost score if key words in incident match SOP text."""
    incident_words = set(incident_text.lower().split())
    sop_words = set(sop_text.lower().split())
    
    # Extract keywords (remove stopwords and short tokens)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has", "do", "does", "did", "i", "we", "he", "she", "it", "might", "may", "could", "would", "should", "think", "at", "on", "in", "of", "to", "for", "and", "or", "but"}
    keywords = {w for w in incident_words if len(w) > 3 and w not in stopwords}
    
    # Count matches
    matches = len(keywords & sop_words)
    boost = 1.0 + (0.1 * matches)  # +10% per keyword match
    return boost


def retrieve_topk(
    incident_text: str,
    sops_path: str = None,
    index_dir: str = None,
    k: int = 5,
) -> List[Dict[str, str | float]]:
    # Set default paths relative to project root if not provided
    if sops_path is None:
        project_root = Path(__file__).parent.parent.parent
        sops_path = str(project_root / "src" / "data" / "sop_examples.json")
    if index_dir is None:
        project_root = Path(__file__).parent.parent.parent
        index_dir = str(project_root / "artifacts" / "retriever_bert" / "index")
    
    sops = load_json(sops_path, SOPEntry, key="sop_examples")
    texts = [_synthesize_text(s) for s in sops]
    ids = [s.sop_id for s in sops]
    out = Path(index_dir)
    if not (out / "embeddings.npy").exists():
        console.log("[yellow]Index not found. Building now...[/yellow]")
        build_and_save_index(texts, ids, out)
    emb, id_list, faiss_index, embed_type, vectorizer = load_index(out)
    results = search([incident_text], emb, id_list, faiss_index, top_k=k*2, embed_type=embed_type, vectorizer=vectorizer)[0]  # Get 2x more, then re-rank
    
    id_to_text = {s.sop_id: _synthesize_text(s) for s in sops}
    id_to_sop = {s.sop_id: s for s in sops}
    
    # Apply keyword boosting and re-rank
    boosted_results = []
    for sop_id, score in results:
        boost = _keyword_boost(incident_text, id_to_text.get(sop_id, ""))
        boosted_score = score * boost
        boosted_results.append((sop_id, boosted_score))
    
    # Sort by boosted score and return top-k
    boosted_results.sort(key=lambda x: -x[1])
    return [{"sop_id": sop_id, "score": score, "text": id_to_text.get(sop_id, "")} for sop_id, score in boosted_results[:k]]


def build_parser() -> argparse.ArgumentParser:
    # Get the project root directory (three levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser(description="Retrieve top-k SOPs for an incident.")
    p.add_argument("--q", type=str, default="Machine A red light; pressure low")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sops", type=str, default=str(project_root / "src" / "data" / "sop_examples.json"))
    p.add_argument("--index_dir", type=str, default=str(project_root / "artifacts" / "retriever_bert" / "index"))
    p.add_argument("--rebuild_index", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.rebuild_index:
        sops = load_json(args.sops, SOPEntry, key="sop_examples")
        texts = [_synthesize_text(s) for s in sops]
        ids = [s.sop_id for s in sops]
        build_and_save_index(texts, ids, args.index_dir)
    hits = retrieve_topk(args.q, sops_path=args.sops, index_dir=args.index_dir, k=args.k)
    for h in hits:
        console.print(h)


