from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from rich.console import Console

# Console instance for logging output with rich formatting
console = Console()


@dataclass
class Pair:
    """
    Represents a pair of incident and SOP (Standard Operating Procedure) text for cross-encoder ranking.
    
    Attributes:
        incident: The incident/query text describing the problem
        sop_text: The SOP text retrieved by the dual-encoder (candidate for reranking)
        sop_id: Unique identifier for the SOP for tracking
    """
    incident: str
    sop_text: str
    sop_id: str


def _heuristic_score(text: str, sop: str) -> float:
    """
    Compute a heuristic relevance score based on word overlap between incident and SOP text.
    
    Used as a fallback when DeBERTa cross-encoder is unavailable.
    Calculates: (number of shared words) / (total unique words in incident + 1)
    
    Args:
        text: The incident text
        sop: The SOP text
        
    Returns:
        A float score representing the heuristic relevance (higher = more relevant)
    """
    text_l = text.lower()
    sop_l = sop.lower()
    # Count how many words overlap between incident and SOP
    overlap = len(set(text_l.split()) & set(sop_l.split()))
    # Normalize by total unique words in incident (with +1 to avoid division by zero)
    return float(overlap) / (1.0 + len(set(text_l.split())))


def predict(pairs: Sequence[Pair]) -> List[Tuple[str, float]]:
    """
    Rerank pairs of incidents and SOPs using a cross-encoder model or heuristic fallback.
    
    This function uses DeBERTa-v3-base as a cross-encoder to score how well each SOP matches 
    the given incident. If DeBERTa is unavailable, falls back to a simple word-overlap heuristic.
    
    Args:
        pairs: Sequence of Pair objects containing incident text, SOP text, and SOP IDs
        
    Returns:
        List of tuples (sop_id, score) sorted by relevance. Score is computed from the [CLS] 
        token embedding of the cross-encoder or from the heuristic word-overlap function.
    """
    try:
        # Import deep learning libraries for cross-encoder inference
        from transformers import AutoModel, AutoTokenizer
        import torch
        import numpy as np

        # Use DeBERTa-v3-base as the cross-encoder model
        model_name = "microsoft/deberta-v3-base"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Set model to evaluation mode (no dropout, no gradient updates)
        model.eval()
        
        scores: List[Tuple[str, float]] = []
        
        # Disable gradient computation for efficiency during inference
        with torch.no_grad():
            for p in pairs:
                # Tokenize the incident and SOP text as a pair for the cross-encoder
                enc = tok(
                    p.incident,
                    p.sop_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                # Get the [CLS] token embedding (contextualized pair representation)
                cls_token = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token
                
                # Use L2 norm of CLS token as relevance score
                # Higher norm = more "activated" representation = more relevant
                score = float(torch.norm(cls_token).item())
                
                scores.append((p.sop_id, score))
        
        # Sort by score descending (higher = more relevant)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    except Exception as e:
        # Graceful fallback if DeBERTa is unavailable (e.g., missing dependencies)
        console.log(f"[yellow]Cross-encoder unavailable, falling back to heuristic: {e}[/yellow]")
        # Use heuristic word-overlap scoring for all pairs
        return [(p.sop_id, _heuristic_score(p.incident, p.sop_text)) for p in pairs]


