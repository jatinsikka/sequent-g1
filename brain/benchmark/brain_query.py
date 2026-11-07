"""Machine-readable retrieval endpoint for the SIM repo (Gate 7 ticket demo).

Usage (from the SIM side, via subprocess):
    .venv_brain\\Scripts\\python.exe brain_query.py "<incident ticket text>" [k]

Prints EXACTLY ONE JSON object on stdout (all model/progress noise goes to stderr):
    {
      "query":  "<the ticket text>",
      "top5":   [{"sop_id","title","score"}, ...],
      "top_sop": {"sop_id","title","condition","steps":[...]}
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: brain_query.py <incident text> [k]"}))
        sys.exit(2)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    from src.data.schemas import SOPEntry, load_json
    from src.retrieval.st_retrieve import retrieve_topk_st

    hits = retrieve_topk_st(query, k=k)

    sops_path = Path(__file__).resolve().parent / "src" / "data" / "plant_sops.json"
    sops = {s.sop_id: s for s in load_json(str(sops_path), SOPEntry, key="sop_examples")}
    top = sops[hits[0]["sop_id"]]

    out = {
        "query": query,
        "top5": [{"sop_id": h["sop_id"], "title": h["title"], "score": round(h["score"], 4)} for h in hits],
        "top_sop": {
            "sop_id": top.sop_id,
            "title": top.title,
            "condition": top.condition,
            "steps": list(top.steps),
        },
    }
    # stdout carries ONLY the JSON payload; sentence-transformers progress goes to stderr.
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
