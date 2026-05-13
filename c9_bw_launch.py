"""Submit 4 hand-picked candidates to push BW above the current 27 GHz
plateau at c_target=9.

Each candidate tests one hypothesis:
  A:  Extrapolate top-BW direction — push g to max, narrow both rails.
  B:  Same geometry as current top-BW design, but with relaxed-rule fine
      T-bars (s=h=r=1).  Test whether finer T reduces alpha without moving
      Z0 much (decoupled BW boost).
  C:  Most extreme high-Z0 corner: g=max, both rails at min, everything
      else at relaxed minimums.  Probes the upper limit of Z0.
  D:  Asymmetric rails: signal narrow, ground wide.  Test whether ground
      cross-section dominates conductor loss while signal width sets Z0.

Submits as a single FDTD batch tagged `bw_hand_1`.  Computes the analytic
BW after the FDTDs land and journals BW_GHz alongside (Z0, n_eff).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from step2.fab_rules_relaxed import RELAXED_RULES
from step2.fab_rules import feasible, repair_geometry_constraint, clip_to_bounds
from step2.geom import CPSGeometry
from step2.journal import append_evaluation
from step2.junction import Junction, bandwidth_3dB_GHz, mzm_length_um
from step2.objective_bw import bw_objective
from step2.select_C_targets import read_targets
from step2.simulate import evaluate_cps_batch


CANDIDATES = [
    # A: push gap to max, both rails to min, mid-range T (extrapolate top-1)
    dict(name="A_pushG_narrowRails",
         g=200.0, ws=20.0, wg=20.0,
         s=20.0, r=60.0, h=12.0, t=2.0, c=6.0),
    # B: current top (g=145, ws=25, wg=29) + fine relaxed T-bars
    dict(name="B_topGeom_fineT",
         g=145.0, ws=25.0, wg=29.0,
         s=1.0, r=1.0, h=1.0, t=1.0, c=4.0),
    # C: extreme high-Z0 corner
    dict(name="C_extremeZ0",
         g=200.0, ws=20.0, wg=20.0,
         s=1.0, r=1.0, h=1.0, t=1.0, c=1.0),
    # D: asymmetric — signal narrow, ground wide
    dict(name="D_asym_groundWide",
         g=200.0, ws=20.0, wg=200.0,
         s=20.0, r=60.0, h=12.0, t=2.0, c=6.0),
]


def main():
    # Repair / clip to relaxed rules before submission
    fixed = []
    for c in CANDIDATES:
        name = c.pop("name")
        c = repair_geometry_constraint(c, rules=RELAXED_RULES)
        c = clip_to_bounds(c, rules=RELAXED_RULES)
        ok, viol = feasible(c, rules=RELAXED_RULES)
        if not ok:
            raise SystemExit(f"{name} infeasible after repair: {viol}")
        fixed.append((name, c))

    print("=== Hand-picked c_target=9 BW candidates ===")
    for n, c in fixed:
        print(f"  {n:30}  {c}")

    target = next(t for t in read_targets() if t["c_target_index"] == 9)
    junction = Junction(C_pF_per_cm=target["C_pF_per_cm"],
                        R_ohm_cm=target["R_ohm_cm"],
                        VpiL_V_cm=target["VpiL_V_cm"],
                        n_group_opt=3.88)
    L_um = mzm_length_um(5.0, junction.VpiL_V_cm, 2.0, push_pull=True)
    print(f"\nJunction: C={junction.C_pF_per_cm:.2f} pF/cm  "
          f"R={junction.R_ohm_cm:.4f} Ohm.cm  Vpi L={junction.VpiL_V_cm:.3f}  "
          f"-> L_mod={L_um:.0f} um")

    geom_dicts = [c for _, c in fixed]
    geoms = [CPSGeometry(**g) for g in geom_dicts]
    print(f"\n=== Submitting batch of {len(geoms)} FDTDs ===")
    t0 = time.time()
    cps_results = evaluate_cps_batch(geoms)
    print(f"Batch done in {time.time()-t0:.0f} s")

    for (name, g), cps in zip(fixed, cps_results):
        if cps.failed:
            print(f"  {name}: FAILED  {cps.failure_reasons}")
            entry = {
                "c_target_index": 9,
                "batch_id": "bw_hand_1",
                "name": name,
                "geometry": g,
                "geometry_hash": cps.geometry_hash,
                "objective": 1e3,
                "BW_GHz": float("nan"),
                "MZM_length_um": L_um,
                "Z0_re_f0": float("nan"),
                "n_eff_f0": float("nan"),
                "target_n_group": junction.n_group_opt,
                "target_C_pF_per_cm": target["C_target_pF_per_cm"],
                "junction_C_pF_per_cm": target["C_pF_per_cm"],
                "junction_R_ohm_cm": target["R_ohm_cm"],
                "junction_VpiL_V_cm": target["VpiL_V_cm"],
                "failed": True,
                "failure_reasons": cps.failure_reasons,
                "wall_time_s": cps.wall_time_s,
            }
            append_evaluation(entry)
            continue

        bw_comp = bw_objective(cps, junction)
        # Also compute Z0_f0 + n_eff_f0 for journal continuity
        from step2.junction import loaded_at_f0
        Z0_re, n_eff, _ = loaded_at_f0(cps, junction)

        bw = bw_comp["BW_GHz"]
        print(f"  {name:30}  BW={bw:6.2f} GHz  Z0={Z0_re:5.2f}  "
              f"n_eff={n_eff:5.2f}  hash={cps.geometry_hash}")

        entry = {
            "c_target_index": 9,
            "batch_id": "bw_hand_1",
            "name": name,
            "geometry": g,
            "geometry_hash": cps.geometry_hash,
            "objective": bw_comp["objective"],
            "BW_GHz": bw_comp["BW_GHz"],
            "MZM_length_um": L_um,
            "Z0_re_f0": Z0_re,
            "n_eff_f0": n_eff,
            "target_n_group": junction.n_group_opt,
            "target_C_pF_per_cm": target["C_target_pF_per_cm"],
            "junction_C_pF_per_cm": target["C_pF_per_cm"],
            "junction_R_ohm_cm": target["R_ohm_cm"],
            "junction_VpiL_V_cm": target["VpiL_V_cm"],
            "failed": False,
            "failure_reasons": [],
            "wall_time_s": cps.wall_time_s,
        }
        append_evaluation(entry)

    # Print updated leaderboard for c_target=9 by BW
    print("\n=== Updated top-5 by BW (analytic, from cache) ===")
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("c_target_index") == 9
            and r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))]
    # Compute BW for any row without BW_GHz column (older rows)
    import pickle
    cache_dir = Path("cache_step2")
    for r in rows:
        if r.get("BW_GHz") is not None:
            continue
        gh = r["geometry_hash"]
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        if pkls:
            try:
                with open(pkls[0], "rb") as f:
                    cps = pickle.load(f)
                r["BW_GHz"] = bandwidth_3dB_GHz(cps, junction, L_um)
            except Exception:
                r["BW_GHz"] = None
    rows = [r for r in rows
            if r.get("BW_GHz") is not None and np.isfinite(r["BW_GHz"])]
    rows.sort(key=lambda r: -r["BW_GHz"])
    for r in rows[:6]:
        g = r["geometry"]
        print(f"  BW={r['BW_GHz']:5.2f}  Z0={r['Z0_re_f0']:5.2f}  "
              f"n={r['n_eff_f0']:4.2f}  batch={r['batch_id']:14}  "
              f"g/ws/wg={g['g']:>5.1f}/{g['ws']:>5.0f}/{g['wg']:>5.0f}")


if __name__ == "__main__":
    main()
