from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v / norms


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def _build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def _search_faiss(index, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, idx = index.search(query_vecs.astype(np.float32), top_k)
    return idx, scores


def _tfidf_vectorize(texts: Sequence[str], vectorizer=None):
    """Vectorize texts using TF-IDF. If vectorizer is provided, use it; otherwise fit a new one."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        return X.toarray().astype(np.float32), vectorizer
    else:
        X = vectorizer.transform(texts)
        return X.toarray().astype(np.float32), vectorizer


@dataclass
class IndexPaths:
    root: Path

    @property
    def faiss_index(self) -> Path:
        return self.root / "index.faiss"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings.npy"

    @property
    def ids(self) -> Path:
        return self.root / "ids.json"

    @property
    def vectorizer(self) -> Path:
        return self.root / "vectorizer.pkl"

    @property
    def embed_type(self) -> Path:
        return self.root / "embed_type.json"


def build_and_save_index(
    texts: Sequence[str],
    ids: Sequence[str],
    out_dir: str | Path,
) -> None:
    """Builds a cosine FAISS index (if available) with normalized embeddings, else saves for brute-force."""
    out = IndexPaths(Path(out_dir))
    out.root.mkdir(parents=True, exist_ok=True)

    console.log(f"[bold]Indexing {len(texts)} SOP texts[/bold]")
    try:
        # Try to produce embeddings via transformers if available
        from transformers import AutoModel, AutoTokenizer
        import torch

        # Try to load trained sop_encoder first; fallback to pre-trained bert-base-uncased
        # out.root is artifacts/retriever_bert/index, so parent.parent gives us project root
        artifact_dir = out.root.parent  # This is artifacts/retriever_bert/
        sop_encoder_path = artifact_dir / "sop_encoder"
        tokenizer_path = artifact_dir / "tokenizer"
        
        console.log(f"[dim]Looking for trained models in: {artifact_dir}[/dim]")
        console.log(f"[dim]SOP encoder path: {sop_encoder_path}[/dim]")
        console.log(f"[dim]Tokenizer path: {tokenizer_path}[/dim]")
        console.log(f"[dim]SOP encoder exists: {sop_encoder_path.exists()}[/dim]")
        console.log(f"[dim]Tokenizer exists: {tokenizer_path.exists()}[/dim]")
        
        if sop_encoder_path.exists() and tokenizer_path.exists():
            console.log(f"[dim]Loading trained SOP encoder and tokenizer[/dim]")
            model_name = str(sop_encoder_path)
            tokenizer_name = str(tokenizer_path)
        else:
            console.log(f"[dim]No trained SOP encoder found; using pre-trained bert-base-uncased[/dim]")
            model_name = "bert-base-uncased"
            tokenizer_name = "bert-base-uncased"
        
        console.log(f"[dim]Loading tokenizer and model (may download on first run)...[/dim]")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        all_vecs: List[np.ndarray] = []
        with torch.no_grad():
            for i in track(range(0, len(texts), 8), description="Embedding (BERT)"):
                batch = texts[i : i + 8]
                tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                out_hidden = model(**tok).last_hidden_state  # [B, T, H]
                cls = out_hidden[:, 0, :]  # [CLS]
                vec = cls.cpu().numpy().astype(np.float32)
                all_vecs.append(vec)
        embeddings = _normalize(np.vstack(all_vecs))
        console.log("Built embeddings with BERT.")
        # Save embed type for BERT
        with open(out.embed_type, "w") as f:
            json.dump({"type": "bert"}, f)
    except Exception as e:  # Offline or no weights
        import traceback
        import pickle
        console.log(f"[yellow]Falling back to TF-IDF embeddings due to: {type(e).__name__}: {e}[/yellow]")
        console.log(f"[dim]{traceback.format_exc()}[/dim]")
        embeddings, vectorizer = _tfidf_vectorize(texts)
        embeddings = _normalize(embeddings)
        # Save vectorizer for use during search
        with open(out.vectorizer, "wb") as f:
            pickle.dump(vectorizer, f)
        # Save embed type
        with open(out.embed_type, "w") as f:
            json.dump({"type": "tfidf"}, f)
        console.log("[dim]Saved TF-IDF vectorizer for search[/dim]")

    np.save(out.embeddings, embeddings)
    with out.ids.open("w", encoding="utf-8") as f:
        json.dump(list(ids), f, ensure_ascii=False, indent=2)

    if _faiss_available():
        try:
            index = _build_faiss_index(embeddings)
            # Persist FAISS index
            import faiss

            faiss.write_index(index, str(out.faiss_index))
            console.log(f"FAISS index saved to {out.faiss_index}")
        except Exception as e:
            console.log(f"[yellow]FAISS save failed (fallback to numpy search only): {e}[/yellow]")
    else:
        console.log("[yellow]faiss-cpu not available, will use numpy brute-force search.[/yellow]")


def load_index(out_dir: str | Path):
    """Loads embeddings and ids. If FAISS is available and index file exists, load it too.
    Also loads vectorizer and embed_type if available."""
    import pickle
    out = IndexPaths(Path(out_dir))
    emb = np.load(out.embeddings)
    with out.ids.open("r", encoding="utf-8") as f:
        ids = json.load(f)

    faiss_index = None
    if _faiss_available() and out.faiss_index.exists():
        import faiss

        faiss_index = faiss.read_index(str(out.faiss_index))
    
    # Load embed type and vectorizer if available
    embed_type = "bert"
    vectorizer = None
    if out.embed_type.exists():
        with open(out.embed_type, "r") as f:
            embed_type = json.load(f).get("type", "bert")
    if embed_type == "tfidf" and out.vectorizer.exists():
        with open(out.vectorizer, "rb") as f:
            vectorizer = pickle.load(f)
    
    return emb, ids, faiss_index, embed_type, vectorizer


def search(
    query_texts: Sequence[str],
    corpus_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    faiss_index,
    top_k: int = 5,
    embed_type: str = "bert",
    vectorizer=None,
) -> List[List[Tuple[str, float]]]:
    """Search top-k cosine similar items."""
    # Compute query embeddings
    if embed_type == "tfidf" and vectorizer is not None:
        # Use the saved TF-IDF vectorizer
        query_vecs, _ = _tfidf_vectorize(query_texts, vectorizer)
        query_vecs = _normalize(query_vecs)
    else:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            from pathlib import Path

            # Find the project root by going up from this file
            # This file is at: project_root/src/retrieval/index_utils.py
            project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
            artifact_dir = project_root / "artifacts" / "retriever_bert"
            incident_encoder_path = artifact_dir / "incident_encoder"
            tokenizer_path = artifact_dir / "tokenizer"
            
            console.log(f"[dim]Looking for trained models in: {artifact_dir}[/dim]")
            console.log(f"[dim]Incident encoder path: {incident_encoder_path}[/dim]")
            console.log(f"[dim]Incident encoder exists: {incident_encoder_path.exists()}[/dim]")
            
            if incident_encoder_path.exists() and tokenizer_path.exists():
                console.log(f"[dim]Loading trained incident encoder[/dim]")
                model_name = str(incident_encoder_path)
                tokenizer_name = str(tokenizer_path)
            else:
                console.log(f"[dim]No trained incident encoder found; using pre-trained bert-base-uncased[/dim]")
                model_name = "bert-base-uncased"
                tokenizer_name = "bert-base-uncased"
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            all_vecs: List[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(query_texts), 8):
                    batch = query_texts[i : i + 8]
                    tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                    out_hidden = model(**tok).last_hidden_state
                    cls = out_hidden[:, 0, :]
                    vec = cls.cpu().numpy().astype(np.float32)
                    all_vecs.append(vec)
            query_vecs = _normalize(np.vstack(all_vecs))
        except Exception as e:
            # TF-IDF fallback - but warn that results may be poor without saved vectorizer
            console.log(f"[yellow]Warning: Using TF-IDF without saved vectorizer - results may be inconsistent[/yellow]")
            texts = list(query_texts)
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Fit on union of query and a small subset of corpus for cosine space alignment
            sample_corpus = [" ".join(map(str, range(32)))] + ["sample baseline"]  # stabilizer
            vec = TfidfVectorizer(max_features=min(2048, corpus_embeddings.shape[1]))
            vec.fit(texts + sample_corpus)
            query_vecs = _normalize(vec.transform(texts).toarray().astype(np.float32))

    if faiss_index is not None:
        idx, scores = _search_faiss(faiss_index, query_vecs, top_k)
    else:
        # brute-force cosine = dot of normalized
        sims = query_vecs @ corpus_embeddings.T
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, idx, axis=1)

    # Clip scores to [0, 1] range for interpretability
    # With dual-encoders, inner products can slightly exceed 1.0
    scores = np.clip(scores, 0.0, 1.0)

    results: List[List[Tuple[str, float]]] = []
    for r, s in zip(idx, scores):
        results.append([(corpus_ids[int(i)], float(sc)) for i, sc in zip(r, s)])
    return results


