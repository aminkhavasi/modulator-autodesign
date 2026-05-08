"""Adaptive sweep orchestrator for PN junction modulator.

Bracket-and-fill strategy (Option A):
  1. Start with 4 anchor points: mult in {0.2, 1.0, 5.0, 20.0}.
  2. After all anchors are evaluated, repeatedly insert mults at the
     midpoint (geometric) of the largest gap on the (VpiL, C) Pareto
     frontier until the budget is spent.

A "run" = one charge sim (9 voltages) + one mode-solver batch (9 voltages).
Cost is paid once per mult; the 7 interior voltages become 7 trade-off
points in the journal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from pn_junction import (
    INTERIOR_MASK,
    VOLTAGES,
    DesignResult,
    evaluate_design,
)

JOURNAL = Path("journal.jsonl")
ANCHORS = (0.2, 1.0, 5.0, 20.0)
MULT_MIN = 0.2
MULT_MAX = 20.0


# ---------------------------------------------------------------------------
# Journal I/O
# ---------------------------------------------------------------------------
def load_journal(path: Path = JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluated_mults(history: list[dict]) -> set[float]:
    """Set of `mult` values that have at least one journal entry."""
    return {round(e["mult"], 6) for e in history}


def append_result(result: DesignResult, *,
                  path: Path = JOURNAL,
                  notes: str = "") -> int:
    """Write the 7 interior-voltage rows to the journal. Returns rows written."""
    rows = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with path.open("a") as f:
        for i, v in enumerate(VOLTAGES):
            if not INTERIOR_MASK[i]:
                continue
            entry = {
                "timestamp": now,
                "run_id": f"mult_{result.mult:.4f}",
                "mult": result.mult,
                "p_doping_cm3": result.p_doping,
                "n_doping_cm3": result.n_doping,
                "target_v": float(v),
                "C_pF_per_cm": float(result.C_pF_cm[i]),
                "R_ohm_cm": result.R_ohm_cm,
                "f3dB_GHz": float(result.f3dB_GHz[i]),
                "VpiL_V_cm": (None if np.isnan(result.VpiL_V_cm[i])
                              else float(result.VpiL_V_cm[i])),
                "loss_dB_per_cm": float(result.loss_dB_cm[i]),
                "n_eff_re": float(result.n_eff_baseline.real),
                "n_eff_im": float(result.n_eff_baseline.imag),
                "charge_cache": result.charge_task_id,
                "mode_batch_dir": result.mode_solver_batch_dir,
                "notes": notes,
            }
            f.write(json.dumps(entry) + "\n")
            rows += 1
    return rows


# ---------------------------------------------------------------------------
# Bracket-and-fill strategy
# ---------------------------------------------------------------------------
def pick_next_mult(history: list[dict], *,
                   anchors: tuple[float, ...] = ANCHORS,
                   mult_min: float = MULT_MIN,
                   mult_max: float = MULT_MAX) -> float | None:
    """Return the next mult to evaluate, or None if no more should be run."""
    done = evaluated_mults(history)

    # Phase 1: place all anchors first
    for a in anchors:
        if round(a, 6) not in done:
            return a

    # Phase 2: fill the largest gap on the (VpiL, C) Pareto frontier.
    pts = _pareto_points(history)
    if len(pts) < 2:
        # Degenerate: not enough valid points yet, fall back to log midpoint
        sorted_done = sorted(done)
        gaps = [(sorted_done[i + 1] / sorted_done[i], i)
                for i in range(len(sorted_done) - 1)]
        if not gaps:
            return None
        ratio, i = max(gaps)
        return float(math.sqrt(sorted_done[i] * sorted_done[i + 1]))

    # Geometric-midpoint fill on the Pareto-frontier mults
    pareto_mults = sorted({p["mult"] for p in pts})
    gaps = [(pareto_mults[i + 1] / pareto_mults[i], i)
            for i in range(len(pareto_mults) - 1)]
    if not gaps:
        return None
    ratio, i = max(gaps)
    candidate = math.sqrt(pareto_mults[i] * pareto_mults[i + 1])
    candidate = float(np.clip(candidate, mult_min, mult_max))

    # Avoid resubmitting an already-evaluated mult
    if round(candidate, 6) in done:
        return None
    return candidate


def _pareto_points(history: list[dict]) -> list[dict]:
    """Pareto-optimal points minimizing both VpiL and C."""
    valid = [e for e in history if e.get("VpiL_V_cm") is not None]
    if not valid:
        return []
    pts = sorted(valid, key=lambda e: (e["VpiL_V_cm"], e["C_pF_per_cm"]))
    pareto = []
    best_C = math.inf
    for p in pts:
        if p["C_pF_per_cm"] < best_C:
            pareto.append(p)
            best_C = p["C_pF_per_cm"]
    return pareto


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(args):
    """Evaluate one mult (given) or one auto-picked next mult."""
    history = load_journal()
    if args.mult is not None:
        mult = args.mult
    else:
        mult = pick_next_mult(history)
        if mult is None:
            print("Strategy says: no further mult to evaluate.")
            return
    print(f"=== Evaluating mult = {mult} ===")
    result = evaluate_design(mult)
    n = append_result(result, notes=args.notes)
    print(f"Wrote {n} rows to {JOURNAL}")


def cmd_next(args):
    """Print the next-to-evaluate mult without running anything."""
    history = load_journal()
    mult = pick_next_mult(history)
    print(mult if mult is not None else "DONE")


def cmd_status(args):
    """Print summary of current journal."""
    history = load_journal()
    done = sorted(evaluated_mults(history))
    print(f"{len(history)} journal rows, {len(done)} unique mults evaluated:")
    for m in done:
        rows = [e for e in history if round(e["mult"], 6) == round(m, 6)]
        valid = [r for r in rows if r.get("VpiL_V_cm") is not None]
        print(f"  mult={m:7.3f}  rows={len(rows):2d}  valid_VpiL={len(valid):2d}")


def cmd_sweep(args):
    """Run autonomously until budget is spent or strategy says DONE."""
    for _ in range(args.budget):
        history = load_journal()
        mult = pick_next_mult(history)
        if mult is None:
            print("Strategy says: no further mult to evaluate.")
            break
        print(f"\n=== Evaluating mult = {mult} ===")
        try:
            result = evaluate_design(mult)
            n = append_result(result, notes=f"auto-sweep ({args.budget} budget)")
            print(f"Wrote {n} rows to {JOURNAL}")
        except Exception as exc:
            print(f"!! mult={mult} failed: {exc!r}")
            print("   Continuing with next mult.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="evaluate one mult")
    p_run.add_argument("--mult", type=float, default=None,
                       help="explicit mult; omit to auto-pick next")
    p_run.add_argument("--notes", default="", help="free-text notes for journal")
    p_run.set_defaults(func=cmd_run)

    p_next = sub.add_parser("next", help="print next mult without running")
    p_next.set_defaults(func=cmd_next)

    p_status = sub.add_parser("status", help="print journal summary")
    p_status.set_defaults(func=cmd_status)

    p_sweep = sub.add_parser("sweep", help="run autonomously up to budget")
    p_sweep.add_argument("--budget", type=int, default=10,
                         help="max number of runs (default 10)")
    p_sweep.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
