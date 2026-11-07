"""Shared loaders for the brain reproduction. Reuses the repo's own schemas and the
exact synth()/metrics() from src/retrieval/train_eval_sentence_encoder.py."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.schemas import SOPEntry, IncidentEntry, load_json  # noqa: E402
from src.retrieval.train_eval_sentence_encoder import synth, metrics  # noqa: E402,F401

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
FINALE_ID = "SOP-1013"  # added after the benchmark (commit 224370b) for the robot finale


def load(exclude_finale: bool = False):
    d = ROOT / "src" / "data"
    sops = load_json(str(d / "plant_sops.json"), SOPEntry, key="sop_examples")
    if exclude_finale:
        sops = [s for s in sops if s.sop_id != FINALE_ID]
    split = {k: load_json(str(d / f"plant_incidents_{k}.json"), IncidentEntry, key="incident_examples")
             for k in ("train", "val", "test")}
    return sops, split


def family_map() -> dict:
    """sop_id -> family slug, replaying data_gen/merge_corpus.py's ID assignment exactly."""
    import json
    out, n = {}, 0
    for fp in sorted((ROOT / "data_gen" / "raw").glob("*.json")):
        seen = set()
        for s in json.loads(fp.read_text(encoding="utf-8"))["sops"]:
            ref = str(s.get("ref", "")).strip()
            if not ref or not s.get("title") or not s.get("steps"):
                continue
            n += 1
            if ref in seen:
                continue
            seen.add(ref)
            out[f"SOP-{n:04d}"] = fp.stem
    out[FINALE_ID] = "finale"
    return out
