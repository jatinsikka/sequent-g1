from __future__ import annotations

from src.retrieval.infer_retrieve import retrieve_topk


def test_retrieval_non_empty():
    hits = retrieve_topk("Machine A red light; pressure low", k=5)
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert "sop_id" in hits[0]


