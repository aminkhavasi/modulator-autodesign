"""Select the 10 outer-loop (C, R, VpiL) targets from Step-1's journal.

For each of 10 linearly-spaced target C values from C_min to C_max of Step-1's
*interior* points, find the Step-1 row with the lowest VpiL whose
C_pF_per_cm is closest to (and <= ) the target.  Tie-breaker: closest C, then
min VpiL.

Result is written to `step2_targets.json` so subsequent batches see a stable
target list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEP1_JOURNAL = Path("journal.jsonl")
TARGETS_FILE = Path("step2_targets.json")
N_TARGETS = 10


def load_step1_interior() -> list[dict]:
    if not STEP1_JOURNAL.exists():
        raise FileNotFoundError(f"{STEP1_JOURNAL} not found.  Run Step 1 first.")
    rows = []
    with STEP1_JOURNAL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (row.get("is_interior", True)
                    and row.get("VpiL_V_cm") is not None):
                rows.append(row)
    if not rows:
        raise RuntimeError("No interior Step-1 rows found.")
    return rows


def pick_targets(rows: list[dict], n: int = N_TARGETS) -> list[dict]:
    """Pick `n` (C, R, VpiL) targets, linearly spaced over C, min VpiL per C."""
    Cs = np.array([r["C_pF_per_cm"] for r in rows])
    C_targets = np.linspace(Cs.min(), Cs.max(), n)

    targets = []
    used_keys = set()  # avoid duplicate Step-1 rows
    for i, c_t in enumerate(C_targets):
        # Candidates: those with C close to target.  Use a soft window of
        # 10% of the C range so we always have something to pick from.
        window = (Cs.max() - Cs.min()) * 0.1
        idx_close = np.where(np.abs(Cs - c_t) <= window)[0]
        if len(idx_close) == 0:
            # Fall back to the single nearest row
            idx_close = np.array([np.argmin(np.abs(Cs - c_t))])
        # Among those, pick the row with minimum VpiL (winning the trade-off)
        candidates = [rows[j] for j in idx_close]
        candidates.sort(key=lambda r: (r["VpiL_V_cm"], abs(r["C_pF_per_cm"] - c_t)))
        chosen = candidates[0]

        target = {
            "c_target_index": i,
            "C_target_pF_per_cm": float(c_t),
            "C_pF_per_cm": chosen["C_pF_per_cm"],
            "R_ohm_cm": chosen["R_ohm_cm"],
            "VpiL_V_cm": chosen["VpiL_V_cm"],
            "loss_dB_per_cm": chosen.get("loss_dB_per_cm"),
            "step1_mult": chosen["mult"],
            "step1_target_v": chosen["target_v"],
        }
        targets.append(target)
    return targets


def write_targets(targets: list[dict], path: Path = TARGETS_FILE) -> None:
    with path.open("w") as f:
        json.dump(targets, f, indent=2)


def read_targets(path: Path = TARGETS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.  Run `python -m step2.select_C_targets` first."
        )
    with path.open() as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--n-targets", type=int, default=N_TARGETS)
    p.add_argument("-o", "--output", default=str(TARGETS_FILE))
    args = p.parse_args()

    rows = load_step1_interior()
    targets = pick_targets(rows, n=args.n_targets)
    write_targets(targets, path=Path(args.output))

    print(f"Picked {len(targets)} C targets (saved to {args.output}):")
    print(f"  {'idx':>3}  {'C_target':>10}  {'C_actual':>10}  "
          f"{'VpiL':>8}  {'R':>10}  {'mult':>6}  {'V_bias':>8}")
    for t in targets:
        print(f"  {t['c_target_index']:>3d}  "
              f"{t['C_target_pF_per_cm']:>10.3f}  "
              f"{t['C_pF_per_cm']:>10.3f}  "
              f"{t['VpiL_V_cm']:>8.3f}  "
              f"{t['R_ohm_cm']:>10.4f}  "
              f"{t['step1_mult']:>6.3f}  "
              f"{t['step1_target_v']:>8.3f}")


if __name__ == "__main__":
    main()
