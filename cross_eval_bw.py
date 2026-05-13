"""Cross-evaluation pass — FREE.

CPS FDTD results are geometry-only (junction loading is applied analytically
in post-processing).  So every cached CPSResult can be re-scored against
every c_target's junction without any new FDTD spend.

We:
  1. Collect every unique cached geometry across all c_targets.
  2. For each c_target_index 0..9, compute analytic BW of every geometry
     under that c_target's junction.
  3. Identify the best-BW design for each c_target (which may be a
     geometry first simulated for a different c_target).
  4. Print the cross-evaluated leaderboard.
  5. Append journal entries (batch_id='cross_eval') for any new (c_idx,
     geometry_hash) pair we hadn't seen before, so the BO history is rich.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from step2.junction import (Junction, bandwidth_3dB_GHz, loaded_at_f0,
                            mzm_length_um)


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))
            and r["objective"] < 1e9]

    # Build {geometry_hash -> (geometry_dict, cps_pickle_path)}
    cache_dir = Path("cache_step2")
    geom_by_hash = {}
    cps_by_hash = {}
    for r in rows:
        gh = r["geometry_hash"]
        if gh in geom_by_hash:
            continue
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        if not pkls:
            continue
        try:
            with open(pkls[0], "rb") as f:
                cps_by_hash[gh] = pickle.load(f)
            geom_by_hash[gh] = r["geometry"]
        except Exception as e:
            print(f"  skip {gh}: {e}")
    print(f"Unique cached geometries: {len(geom_by_hash)}")

    # Build junction per c_target
    targets = {}
    targets_meta = {}
    for r in rows:
        ci = r["c_target_index"]
        if ci not in targets:
            targets[ci] = Junction(
                C_pF_per_cm=r["junction_C_pF_per_cm"],
                R_ohm_cm=r["junction_R_ohm_cm"],
                VpiL_V_cm=r["junction_VpiL_V_cm"],
                n_group_opt=3.88,
            )
            targets_meta[ci] = {
                "C_target_pF_per_cm": r["target_C_pF_per_cm"],
                "junction_C_pF_per_cm": r["junction_C_pF_per_cm"],
                "junction_R_ohm_cm": r["junction_R_ohm_cm"],
                "junction_VpiL_V_cm": r["junction_VpiL_V_cm"],
            }
    L_um_by_c = {ci: mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
                 for ci, J in targets.items()}

    # Existing (c_idx, geometry_hash) coverage from journal
    seen_pairs = {(r["c_target_index"], r["geometry_hash"]) for r in rows}

    # Cross-evaluate
    new_rows = []
    for ci in sorted(targets.keys()):
        if ci < 0:
            continue
        J = targets[ci]
        L_um = L_um_by_c[ci]
        meta = targets_meta[ci]
        for gh, cps in cps_by_hash.items():
            if (ci, gh) in seen_pairs:
                continue
            try:
                bw = bandwidth_3dB_GHz(cps, J, L_um)
                if not np.isfinite(bw):
                    continue
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
            except Exception:
                continue
            new_rows.append({
                "c_target_index": ci,
                "batch_id": "cross_eval",
                "geometry": geom_by_hash[gh],
                "geometry_hash": gh,
                "objective": -float(bw),
                "BW_GHz": float(bw),
                "MZM_length_um": L_um,
                "Z0_re_f0": float(Z0_re),
                "n_eff_f0": float(n_eff),
                "target_n_group": 3.88,
                "target_C_pF_per_cm": meta["C_target_pF_per_cm"],
                "junction_C_pF_per_cm": meta["junction_C_pF_per_cm"],
                "junction_R_ohm_cm": meta["junction_R_ohm_cm"],
                "junction_VpiL_V_cm": meta["junction_VpiL_V_cm"],
                "failed": False,
                "failure_reasons": [],
                "wall_time_s": 0.0,
            })
    print(f"Cross-eval new (c_idx, geometry) pairs: {len(new_rows)}")

    # Compute new per-c_target BW best (union of journal + new cross-eval)
    all_by_c = defaultdict(list)
    for r in rows:
        gh = r["geometry_hash"]
        if gh not in cps_by_hash:
            continue
        # Existing rows already have correct (c_idx, geometry); their
        # objective field is J-phase or BW-phase mixed -- recompute BW here.
        J = targets[r["c_target_index"]]
        L_um = L_um_by_c[r["c_target_index"]]
        bw = bandwidth_3dB_GHz(cps_by_hash[gh], J, L_um)
        if np.isfinite(bw):
            all_by_c[r["c_target_index"]].append({
                **r, "BW_GHz": float(bw)})
    for r in new_rows:
        all_by_c[r["c_target_index"]].append(r)

    print(f"\n{'c':>3} {'C_t':>6} {'L_um':>5} {'old_BW':>6} {'new_BW':>6} "
          f"{'+GHz':>5}  {'new_Z0':>6} {'new_n':>5}  best geom (g/ws/wg)")
    for ci in sorted(all_by_c.keys()):
        if ci < 0:
            continue
        designs = all_by_c[ci]
        # Old best: from journal-original rows (exclude cross_eval)
        from_journal = [d for d in designs if d["batch_id"] != "cross_eval"]
        new_best = max(designs, key=lambda d: d["BW_GHz"])
        old_best = max(from_journal, key=lambda d: d["BW_GHz"])
        g = new_best["geometry"]
        print(f"{ci:>3d} {targets_meta[ci]['junction_C_pF_per_cm']:>6.2f} "
              f"{L_um_by_c[ci]:>5.0f} "
              f"{old_best['BW_GHz']:>6.2f} {new_best['BW_GHz']:>6.2f} "
              f"{new_best['BW_GHz']-old_best['BW_GHz']:>+5.2f}  "
              f"{new_best['Z0_re_f0']:>6.2f} {new_best['n_eff_f0']:>5.2f}  "
              f"{g['g']:>5.1f}/{g['ws']:>5.0f}/{g['wg']:>5.0f}")

    # Optionally append new_rows to journal
    if new_rows:
        with open("step2_journal.jsonl", "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nAppended {len(new_rows)} cross_eval rows to journal.")


if __name__ == "__main__":
    main()
