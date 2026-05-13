"""Look for cached designs at c=0 near (Z=50, n_eff close to 3.88) and
compute their BW.  Also tabulate (Z, n, BW) for all c_targets.
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

    cache_dir = Path("cache_step2")
    cps_by_hash = {}
    for r in rows:
        gh = r["geometry_hash"]
        if gh in cps_by_hash:
            continue
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        if pkls:
            try:
                cps_by_hash[gh] = pickle.load(open(pkls[0], "rb"))
            except Exception:
                pass

    # Group by c_target, dedupe by hash
    by_c = defaultdict(dict)
    for r in rows:
        gh = r["geometry_hash"]
        if gh in cps_by_hash and gh not in by_c[r["c_target_index"]]:
            by_c[r["c_target_index"]][gh] = r

    print(f"{'c':>3} {'C':>5} {'L_um':>5}  "
          f"{'best_BW_geom':30}  "
          f"{'Z0':>5} {'n_eff':>5} {'alpha_f0':>5} {'walk-off deg':>11}")
    full_summary = []
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        designs = []
        for gh, r in by_c[ci].items():
            cps = cps_by_hash[gh]
            try:
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
                a_f0 = float(cps.alpha_dB_cm_bare[len(cps.freqs)//2])
            except Exception:
                continue
            if not (np.isfinite(bw) and np.isfinite(Z0_re)):
                continue
            designs.append({
                "Z0": float(Z0_re), "n_eff": float(n_eff),
                "BW": float(bw), "alpha": a_f0,
                "geometry": r["geometry"], "hash": gh,
                "batch": r["batch_id"],
            })

        best = max(designs, key=lambda d: d["BW"])
        # Walk-off in degrees at f=20 GHz for L_um
        f_test = 20e9
        c_ms = 3e8
        walk_deg = (2*np.pi*f_test*(best["n_eff"] - J.n_group_opt)*
                    L_um*1e-6 / c_ms) * 180/np.pi
        g = best["geometry"]
        print(f"{ci:>3} {J.C_pF_per_cm:>5.2f} {L_um:>5.0f}  "
              f"g={g['g']:>5.1f} ws={g['ws']:>4.0f} wg={g['wg']:>4.0f} "
              f"r={g['r']:>4.1f} c={g['c']:>4.1f}  "
              f"{best['Z0']:>5.1f} {best['n_eff']:>5.2f} "
              f"{best['alpha']:>5.2f} {walk_deg:>11.1f}")
        full_summary.append({
            "c_idx": ci, "C_pF_per_cm": J.C_pF_per_cm,
            "L_um": L_um, "best": best,
        })

    # Now: for c=0 specifically, look for designs with Z ∈ [45,55] AND
    # n_eff close to 3.88 (within +/- 1).
    print("\n=== c=0 designs with loaded Z in [45,55] sorted by BW ===")
    ci = 0
    r0 = next(iter(by_c[ci].values()))
    J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                 R_ohm_cm=r0["junction_R_ohm_cm"],
                 VpiL_V_cm=r0["junction_VpiL_V_cm"],
                 n_group_opt=3.88)
    L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
    designs = []
    for gh, r in by_c[ci].items():
        cps = cps_by_hash[gh]
        Z0_re, n_eff, _ = loaded_at_f0(cps, J)
        bw = bandwidth_3dB_GHz(cps, J, L_um)
        if 45 <= Z0_re <= 55 and np.isfinite(bw):
            designs.append({
                "Z0": float(Z0_re), "n_eff": float(n_eff),
                "BW": float(bw), "geometry": r["geometry"],
                "hash": gh, "batch": r["batch_id"],
            })
    designs.sort(key=lambda d: -d["BW"])
    print(f"  Found {len(designs)} designs with Z in [45,55]")
    for d in designs[:10]:
        g = d["geometry"]
        print(f"    BW={d['BW']:5.2f} Z={d['Z0']:5.2f} n={d['n_eff']:5.2f}  "
              f"g={g['g']:>5.1f} ws={g['ws']:>4.0f} wg={g['wg']:>4.0f} "
              f"r={g['r']:>4.1f} c={g['c']:>4.1f}  batch={d['batch']}")

    # Also look at lower-n_eff for c=0 (in case walk-off matters):
    print("\n=== c=0 designs with n_eff in [3.5, 4.2] sorted by BW ===")
    designs = []
    for gh, r in by_c[ci].items():
        cps = cps_by_hash[gh]
        Z0_re, n_eff, _ = loaded_at_f0(cps, J)
        bw = bandwidth_3dB_GHz(cps, J, L_um)
        if 3.5 <= n_eff <= 4.2 and np.isfinite(bw):
            designs.append({"Z0": float(Z0_re), "n_eff": float(n_eff),
                            "BW": float(bw), "geometry": r["geometry"],
                            "hash": gh, "batch": r["batch_id"]})
    designs.sort(key=lambda d: -d["BW"])
    print(f"  Found {len(designs)} designs with n_eff in [3.5, 4.2]")
    for d in designs[:10]:
        g = d["geometry"]
        print(f"    BW={d['BW']:5.2f} Z={d['Z0']:5.2f} n={d['n_eff']:5.2f}  "
              f"g={g['g']:>5.1f} ws={g['ws']:>4.0f} wg={g['wg']:>4.0f}  "
              f"batch={d['batch']}")


if __name__ == "__main__":
    main()
