"""Shared paths for the demo scripts (g6_onetake, final_ticket_demo, ticket_demo, end_to_end_demo).

Everything is repo-relative. Override with environment variables when the brain lives elsewhere:
  SEQUENT_BRAIN_DIR  folder containing brain_query.py (default: brain/benchmark in this repo)
  SEQUENT_BRAIN_PY   python interpreter for the brain subprocess (default: this interpreter)
"""
from __future__ import annotations

import os
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.environ.get("SEQUENT_BRAIN_DIR", os.path.join(PROJ, "brain", "benchmark"))
BRAIN_PY = os.environ.get("SEQUENT_BRAIN_PY", sys.executable)
BRAIN_QUERY = os.path.join(BRAIN_DIR, "brain_query.py")

_HOW = {
    ".zip": "RL policy checkpoints are not released in this repo (see README, 'What is released').",
    ".mjb": "build the scene first: python build_unified_model.py (needs mujoco_menagerie/robotiq_2f85).",
}


def require(*paths: str) -> None:
    """Exit with a clear message if any file the demo needs is missing."""
    missing = [p for p in paths if not os.path.exists(os.path.join(PROJ, p))]
    if missing:
        lines = [f"  - {p}   ({_HOW.get(os.path.splitext(p)[1], 'missing')})" for p in missing]
        sys.exit("This demo needs files that are not present:\n" + "\n".join(lines))
