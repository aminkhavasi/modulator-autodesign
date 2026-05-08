"""step2_journal.jsonl reader/writer.

One row per FDTD evaluation.  An evaluation is identified by:
  c_target_index : which of the 10 C values is being optimized (0..9)
  batch_id       : human-readable batch label (e.g. "lhs", "bo_1", "bo_2")
  geometry_hash  : hash of the CPSGeometry (so re-evaluations are visible)

Plus the geometry, the loaded-line scalars, the objective, the run metadata,
and an `agent_notes` free-text field where the agent records its analysis
after each batch.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

JOURNAL = Path("step2_journal.jsonl")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load(path: Path = JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_evaluation(entry: dict, *, path: Path = JOURNAL) -> None:
    """Append one row.  No deduplication -- caller is responsible for not
    submitting the same hash twice (use cache instead)."""
    entry = dict(entry)
    entry.setdefault("timestamp", now_iso())
    with path.open("a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


def _json_default(o):
    """Fallback for numpy types in entries."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    raise TypeError(f"can't serialize {type(o)}")


def per_C_history(c_target_index: int, *, path: Path = JOURNAL) -> list[dict]:
    """All journal rows for one outer-loop C target."""
    return [e for e in load(path) if e.get("c_target_index") == c_target_index]


def filter_successes(rows: list[dict]) -> list[dict]:
    """Rows that didn't fail the FDTD sanity check."""
    return [r for r in rows if not r.get("failed", False)]


def best_so_far(c_target_index: int, *, path: Path = JOURNAL) -> dict | None:
    """Lowest-objective successful row for a given C target.  None if none yet."""
    rows = filter_successes(per_C_history(c_target_index, path=path))
    if not rows:
        return None
    return min(rows, key=lambda r: r.get("objective", float("inf")))


def attach_agent_notes(c_target_index: int, batch_id: str,
                       notes: str, *, path: Path = JOURNAL) -> None:
    """Write a 'meta' row carrying the agent's analysis after a batch.

    These rows have geometry=None and are identified by batch_id + a 'meta'
    flag.  Read with `agent_notes(c_target_index)`.
    """
    append_evaluation({
        "meta": True,
        "c_target_index": c_target_index,
        "batch_id": batch_id,
        "agent_notes": notes,
    }, path=path)


def agent_notes(c_target_index: int, *, path: Path = JOURNAL) -> list[dict]:
    """All `meta` rows for a C target, in time order."""
    return [e for e in per_C_history(c_target_index, path=path)
            if e.get("meta")]
