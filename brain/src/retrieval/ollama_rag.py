"""
Ollama-based RAG (Retrieval-Augmented Generation) system for SOP retrieval.

This module implements a RAG system using Ollama embeddings API to:
1. Embed SOP documents using Ollama's embedding model
2. Store SOP embeddings for efficient retrieval
3. Retrieve top-k SOPs for a given incident query using cosine similarity

Usage:
    python src/retrieval/ollama_rag.py --q "Yellow light flashing" --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console

import requests

# Try to import ollama Python library (optional, fallback to API)
try:
    import ollama
    HAS_OLLAMA_LIB = True
except ImportError:
    HAS_OLLAMA_LIB = False

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import SOPEntry, load_json

console = Console()


class OllamaRAGRetriever:
    """RAG-based retriever using Ollama embeddings."""
    
    def __init__(
        self,
        host: str = None,
        embedding_model: str = None,
        sops_path: str = None,
        cache_dir: str = None,
    ):
        """
        Initialize Ollama RAG Retriever.
        
        Args:
            host: Ollama host URL (default: http://localhost:11434)
            embedding_model: Ollama embedding model name (default: nomic-embed-text)
            sops_path: Path to SOP examples JSON file
            cache_dir: Directory to cache embeddings (default: artifacts/ollama_rag)
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", "gemma3:4b")
        
        # Verify Ollama is accessible
        self._verify_ollama_connection()
        
        # Set default paths
        if sops_path is None:
            project_root = Path(__file__).parent.parent.parent
            sops_path = str(project_root / "src" / "data" / "sop_examples.json")
        
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = str(project_root / "artifacts" / "ollama_rag")
        
        self.sops_path = sops_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load SOPs
        self.sops = load_json(self.sops_path, SOPEntry, key="sop_examples")
        self.sop_texts = [self._synthesize_text(s) for s in self.sops]
        self.sop_ids = [s.sop_id for s in self.sops]
        
        # Load or build embeddings
        self.embeddings = self._load_or_build_embeddings()
    
    def _verify_ollama_connection(self) -> None:
        """Verify that Ollama server is running and accessible."""
        try:
            # Try to connect to Ollama
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            console.log(f"[red]Error: Cannot connect to Ollama at {self.host}[/red]")
            console.log(f"[yellow]Make sure Ollama is running. On Windows, you can:[/yellow]")
            console.log(f"[cyan]  1. Check if Ollama is running in the background[/cyan]")
            console.log(f"[cyan]  2. Start it manually: C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Ollama\\ollama.exe serve[/cyan]")
            console.log(f"[cyan]  3. Or just run: ollama serve[/cyan]")
            raise
        except requests.exceptions.RequestException as e:
            console.log(f"[yellow]Warning: Could not verify Ollama connection: {e}[/yellow]")
            console.log(f"[yellow]Continuing anyway...[/yellow]")
    
    def _synthesize_text(self, sop: SOPEntry) -> str:
        """Convert SOP to text representation."""
        return f"{sop.title}. {sop.condition}. Steps: " + " ; ".join(sop.steps)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using Ollama API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # First try using ollama Python library if available (recommended)
        if HAS_OLLAMA_LIB:
            try:
                result = ollama.embeddings(model=self.embedding_model, prompt=text)
                embedding = np.array(result["embedding"], dtype=np.float32)
                if len(embedding) > 0:
                    return embedding
            except Exception as e:
                console.log(f"[dim]ollama library failed ({e}), trying HTTP API...[/dim]")
        
        # Fallback to HTTP API
        # Try different API formats for compatibility
        formats_to_try = [
            {"model": self.embedding_model, "prompt": text},
            {"model": self.embedding_model, "input": text},
        ]
        
        last_error = None
        for json_data in formats_to_try:
            try:
                resp = requests.post(
                    f"{self.host}/api/embeddings",
                    json=json_data,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = np.array(data.get("embedding", []), dtype=np.float32)
                
                if len(embedding) == 0:
                    raise ValueError(f"Empty embedding returned for text: {text[:50]}...")
                
                return embedding
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_error = e
                    continue  # Try next format
                else:
                    raise
            except requests.exceptions.RequestException as e:
                last_error = e
                continue  # Try next format
        
        # If all formats failed, provide helpful error message
        if last_error:
            if isinstance(last_error, requests.exceptions.HTTPError) and last_error.response.status_code == 404:
                error_msg = (
                    f"\n[red]Error: Ollama embeddings endpoint not found (404)[/red]\n"
                    f"[yellow]Your Ollama version may not support /api/embeddings endpoint.[/yellow]\n"
                    f"[yellow]Recommended solution: Install the ollama Python library:[/yellow]\n"
                    f"[cyan]  pip install ollama[/cyan]\n"
                    f"[yellow]Then make sure the model is available:[/yellow]\n"
                    f"[cyan]  ollama pull {self.embedding_model}[/cyan]\n"
                    f"[yellow]The Python library will handle embeddings automatically.[/yellow]"
                )
                console.print(error_msg)
            else:
                console.log(f"[red]Error calling Ollama API: {last_error}[/red]")
            raise last_error
        
        raise RuntimeError("Failed to get embedding with all attempted formats")
    
    def _load_or_build_embeddings(self) -> np.ndarray:
        """
        Load embeddings from cache or build them using Ollama.
        
        Returns:
            Array of SOP embeddings (n_sops, embedding_dim)
        """
        embeddings_file = self.cache_dir / "sop_embeddings.npy"
        metadata_file = self.cache_dir / "metadata.json"
        
        # Check if cached embeddings exist and match current SOPs
        if embeddings_file.exists() and metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Check if model and SOP count match
                if (metadata.get("model") == self.embedding_model and 
                    metadata.get("n_sops") == len(self.sops) and
                    metadata.get("sops_path") == self.sops_path):
                    console.log(f"[dim]Loading cached embeddings from {embeddings_file}[/dim]")
                    embeddings = np.load(embeddings_file)
                    console.log(f"[green]✓ Loaded {len(embeddings)} SOP embeddings[/green]")
                    return embeddings
            except Exception as e:
                console.log(f"[yellow]Error loading cache: {e}, rebuilding...[/yellow]")
        
        # Build embeddings
        console.log(f"[dim]Building embeddings for {len(self.sops)} SOPs using {self.embedding_model}...[/dim]")
        embeddings = []
        
        for i, text in enumerate(self.sop_texts):
            console.log(f"[dim]Embedding SOP {i+1}/{len(self.sop_texts)}[/dim]", end="\r")
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        console.log()  # New line after progress
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Cache embeddings
        np.save(embeddings_file, embeddings)
        with open(metadata_file, "w") as f:
            json.dump({
                "model": self.embedding_model,
                "n_sops": len(self.sops),
                "sops_path": self.sops_path,
                "embedding_dim": embeddings.shape[1],
            }, f, indent=2)
        
        console.log(f"[green]✓ Built and cached {len(embeddings)} SOP embeddings[/green]")
        return embeddings
    
    def retrieve_topk(
        self,
        incident_text: str,
        k: int = 5,
    ) -> List[Dict[str, str | float]]:
        """
        Retrieve top-k SOPs for a given incident.
        
        Args:
            incident_text: Incident description text
            k: Number of top SOPs to retrieve
            
        Returns:
            List of dictionaries with 'sop_id', 'score', and 'text' keys
        """
        # Get query embedding
        query_embedding = self._get_embedding(incident_text)
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        id_to_text = {s.sop_id: self._synthesize_text(s) for s in self.sops}
        
        for idx in top_indices:
            sop_id = self.sop_ids[idx]
            score = float(similarities[idx])
            results.append({
                "sop_id": sop_id,
                "score": score,
                "text": id_to_text.get(sop_id, ""),
            })
        
        return results


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser(description="Ollama RAG-based SOP retrieval.")
    p.add_argument("--q", type=str, default="Yellow warning light is flashing on the machine", help="Incident query text")
    p.add_argument("--k", type=int, default=5, help="Number of top SOPs to retrieve")
    p.add_argument("--sops", type=str, default=str(project_root / "src" / "data" / "sop_examples.json"), help="Path to SOP examples JSON")
    p.add_argument("--host", type=str, default=None, help="Ollama host URL (default: http://localhost:11434)")
    p.add_argument("--model", type=str, default=None, help="Ollama embedding model (default: gemma3:4b)")
    p.add_argument("--cache_dir", type=str, default=None, help="Cache directory for embeddings")
    p.add_argument("--rebuild", action="store_true", help="Rebuild embeddings cache")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    
    # Rebuild cache if requested
    if args.rebuild:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path(__file__).parent.parent.parent / "artifacts" / "ollama_rag"
        cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = cache_dir / "sop_embeddings.npy"
        metadata_file = cache_dir / "metadata.json"
        if embeddings_file.exists():
            embeddings_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        console.log("[yellow]Cache cleared, will rebuild on next retrieval[/yellow]")
    
    # Initialize retriever
    retriever = OllamaRAGRetriever(
        host=args.host,
        embedding_model=args.model,
        sops_path=args.sops,
        cache_dir=args.cache_dir,
    )
    
    # Retrieve
    hits = retriever.retrieve_topk(args.q, k=args.k)
    
    # Display results
    console.print(f"\n[bold]Top {args.k} SOPs for query:[/bold] [dim]{args.q}[/dim]\n")
    for i, hit in enumerate(hits, 1):
        console.print(f"[bold]{i}. SOP ID: {hit['sop_id']}[/bold] (score: {hit['score']:.4f})")
        console.print(f"   {hit['text'][:150]}...")
        console.print()

