"""For each c_target_index 0..9, compute analytic 3-dB BW for every cached
design and report:
  - current best BW
  - the geometry that produced it
  - whether the J-phase best is the same design as the BW best
  - per-c_target headroom analysis (Z0_max observed, Pareto slope n/Z, etc.)

Writes c9_bw_ranking.json -> c_all_bw_ranking.json with one entry per row.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from step2.junction import Junction, bandwidth_3dB_GHz, mzm_length_um


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))
            and r["objective"] < 1e9
            and r.get("c_target_index") is not None
            and r["c_target_index"] >= 0]
    print(f"Total successful rows across c_targets: {len(rows)}")

    cache_dir = Path("cache_step2")
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["c_target_index"]].append(r)

    summary = []
    enriched_all = []
    for c_idx in sorted(by_c.keys()):
        rs = by_c[c_idx]
        j_target = rs[0]
        J = Junction(C_pF_per_cm=j_target["junction_C_pF_per_cm"],
                     R_ohm_cm=j_target["junction_R_ohm_cm"],
                     VpiL_V_cm=j_target["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        rs_enr = []
        for r in rs:
            gh = r["geometry_hash"]
            pkls = [p for p in cache_dir.rglob(f"*{gh}*")
                    if p.suffix == ".pkl"]
            if not pkls:
                continue
            try:
                with open(pkls[0], "rb") as f:
                    cps = pickle.load(f)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
                alpha_f0 = float(cps.alpha_dB_cm_bare[len(cps.freqs) // 2])
            except Exception:
                continue
            if not np.isfinite(bw):
                continue
            r2 = dict(r, BW_GHz=float(bw), alpha_dB_cm_f0=alpha_f0,
                      MZM_length_um=L_um)
            rs_enr.append(r2)
            enriched_all.append(r2)
        if not rs_enr:
            continue

        # Best by J (existing objective) and best by BW
        best_J = min(rs_enr, key=lambda r: r["objective"])
        best_BW = max(rs_enr, key=lambda r: r["BW_GHz"])
        # Pareto slope across c_target
        Zs = np.array([r["Z0_re_f0"] for r in rs_enr])
        ns = np.array([r["n_eff_f0"] for r in rs_enr])
        slope = float(np.mean(ns / Zs))

        # Z0 range explored
        Z_max = float(Zs.max())
        Z_min = float(Zs.min())

        summary.append({
            "c_idx": c_idx,
            "C_target_pF_per_cm": j_target["target_C_pF_per_cm"],
            "VpiL_V_cm": J.VpiL_V_cm,
            "MZM_length_um": L_um,
            "n_runs": len(rs_enr),
            "best_J": best_J["objective"],
            "best_J_BW": best_J["BW_GHz"],
            "best_J_Z0": best_J["Z0_re_f0"],
            "best_J_n_eff": best_J["n_eff_f0"],
            "best_BW": best_BW["BW_GHz"],
            "best_BW_Z0": best_BW["Z0_re_f0"],
            "best_BW_n_eff": best_BW["n_eff_f0"],
            "best_BW_geometry": best_BW["geometry"],
            "best_BW_batch": best_BW["batch_id"],
            "Pareto_slope_n_over_Z": slope,
            "Z0_max_observed": Z_max,
            "Z0_min_observed": Z_min,
            "BW_uplift_GHz": best_BW["BW_GHz"] - best_J["BW_GHz"],
            "same_design": (best_J["geometry_hash"]
                            == best_BW["geometry_hash"]),
        })

    print(f"\n{'c':>3} {'C_t':>6} {'VpiL':>5} {'L_um':>5}  "
          f"{'n':>3} {'J_BW':>5} {'BW*':>5} {'+GHz':>5}  "
          f"{'Z0*':>4} {'n*':>4} {'slope':>5} {'Zrng':>9}  same?")
    for s in summary:
        print(f"{s['c_idx']:>3d} {s['C_target_pF_per_cm']:>6.2f} "
              f"{s['VpiL_V_cm']:>5.3f} {s['MZM_length_um']:>5.0f}  "
              f"{s['n_runs']:>3d} {s['best_J_BW']:>5.2f} "
              f"{s['best_BW']:>5.2f} {s['BW_uplift_GHz']:>+5.2f}  "
              f"{s['best_BW_Z0']:>4.1f} {s['best_BW_n_eff']:>4.1f} "
              f"{s['Pareto_slope_n_over_Z']:>5.3f} "
              f"{s['Z0_min_observed']:>4.1f}-{s['Z0_max_observed']:>4.1f}  "
              f"{'YES' if s['same_design'] else 'no'}")

    print("\nBest-BW geometry per c_target:")
    print(f"{'c':>3}  {'g':>5} {'ws':>5} {'wg':>5} {'s':>5} {'r':>5} "
          f"{'h':>5} {'t':>5} {'c':>5}")
    for s in summary:
        g = s["best_BW_geometry"]
        print(f"{s['c_idx']:>3d}  {g['g']:>5.1f} {g['ws']:>5.0f} "
              f"{g['wg']:>5.0f} {g['s']:>5.1f} {g['r']:>5.1f} "
              f"{g['h']:>5.1f} {g['t']:>5.1f} {g['c']:>5.1f}")

    json.dump(summary, open("c_all_bw_summary.json", "w"),
              indent=2, default=str)
    print("\nWrote c_all_bw_summary.json")


if __name__ == "__main__":
    main()
