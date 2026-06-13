from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

# Allowed skills across the project
# Updated for Unitree G1 humanoid robot with manufacturing machine environment
SKILLS: List[str] = [
    "walk_to",        # Navigate to a location (machine, table, etc.)
    "press_button",   # Press buttons (red_button, green_button, blue_button, yellow_button)
    "wait",           # Wait for a duration
    "read_sensor",    # Read sensor values (pressure, temperature, vibration, light)
    "pick",           # Pick up objects from table
    "place",          # Place objects at a location
    "notify",         # Notify technician/operator
]


class SOPEntry(BaseModel):
    """Standard Operating Procedure entry."""

    sop_id: str
    title: str
    condition: str
    steps: List[str]
    equipment: List[str] = Field(default_factory=list)


class IncidentEntry(BaseModel):
    """Incident/Query entry."""

    incident_id: str
    text: str
    labels: Optional[Dict[str, str]] = None


class PlanStep(BaseModel):
    """One step in the high-level plan."""

    skill: str
    args: Dict[str, Any]


class Plan(BaseModel):
    """Structured plan for execution."""

    goal: str
    steps: List[PlanStep]
    fallback: List[PlanStep] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def load_json(path: str | Path, model: Type[T], key: str = None) -> List[T]:
    """Load a JSON file containing an array and validate each item with the given Pydantic model.

    Args:
        path: Path to JSON file.
        model: Pydantic model class to validate each record.
        key: Optional key to extract array from (e.g., 'sop_examples'). 
             If None, assumes the root is an array.
    Returns:
        List of validated model instances.
    """
    import json

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Extract array from key if provided
    if key:
        if not isinstance(raw, dict) or key not in raw:
            raise ValueError(f"Expected key '{key}' in JSON file {p}")
        items = raw[key]
    else:
        items = raw
    
    if not isinstance(items, list):
        raise ValueError(f"Expected array in JSON file {p}, got {type(items)}")
    
    data: List[T] = []
    for i, obj in enumerate(items):
        try:
            data.append(model.model_validate(obj))
        except ValidationError as e:
            raise ValueError(f"Invalid record at index {i} in {p}: {e}") from e
    return data


def load_jsonl(path: str | Path, model: Type[T]) -> List[T]:
    """Load a JSONL file and validate each line with the given Pydantic model.
    
    DEPRECATED: Use load_json() instead for JSON array files.

    Args:
        path: Path to JSONL file.
        model: Pydantic model class to validate each record.
    Returns:
        List of validated model instances.
    """
    import json

    data: List[T] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                data.append(model.model_validate(obj))
            except ValidationError as e:
                raise ValueError(f"Invalid record in {p}: {e}") from e
    return data


