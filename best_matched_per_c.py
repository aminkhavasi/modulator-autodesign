"""Per c_target, pick the design with the HIGHEST BW *constrained* to
loaded Z0 close to 50 Ω.  Tabulate (Z0, n_eff, alpha at f0, BW,
walk-off at 20 GHz, geometry) so the user can decide where to push more.

If no design satisfies the strict Z ∈ [45,55] window, widen to [40,60].
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
            and r["objective"] < 1e9 and "geometry" in r]

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

    by_c = defaultdict(dict)
    for r in rows:
        gh = r["geometry_hash"]
        if gh in cps_by_hash and gh not in by_c[r["c_target_index"]]:
            by_c[r["c_target_index"]][gh] = r

    # For each c_target, compute (Z, n, BW, alpha) for every cached design,
    # then pick best-BW with |Z-50| <= 5 (then 10).
    print(f"{'c':>3} {'C':>6} {'Lum':>5}  "
          f"{'BW':>5} {'Z0':>5} {'n_eff':>5} {'α@f0':>5} {'WO@20':>5}  "
          f"geom (g/ws/wg/s/r/h/t/c)  batch")
    print("-" * 110)
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        candidates = []
        for gh, r in by_c[ci].items():
            cps = cps_by_hash[gh]
            try:
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
                a_f0 = float(cps.alpha_dB_cm_bare[len(cps.freqs) // 2])
            except Exception:
                continue
            if not (np.isfinite(bw) and np.isfinite(Z0_re)):
                continue
            candidates.append({
                "Z": float(Z0_re), "n": float(n_eff), "BW": float(bw),
                "alpha": a_f0, "geom": r["geometry"], "batch": r["batch_id"],
                "hash": gh,
            })

        # Filter by Z within tightening windows
        for window in [(45, 55), (40, 60), (35, 65), (0, 200)]:
            lo, hi = window
            sub = [d for d in candidates if lo <= d["Z"] <= hi]
            if sub:
                best = max(sub, key=lambda d: d["BW"])
                tag = f"|Z-50|≤{(hi-lo)//2}" if hi - lo <= 30 else "no Z constraint"
                break
        # Walk-off at 20 GHz
        wo = (2 * np.pi * 20e9 * (best["n"] - J.n_group_opt) * L_um * 1e-6
              / 3e8) * 180 / np.pi
        g = best["geom"]
        print(f"{ci:>3d} {J.C_pF_per_cm:>6.2f} {L_um:>5.0f}  "
              f"{best['BW']:>5.2f} {best['Z']:>5.1f} {best['n']:>5.2f} "
              f"{best['alpha']:>5.2f} {wo:>+5.0f}  "
              f"g={g['g']:>5.1f} ws={g['ws']:>4.0f} wg={g['wg']:>4.0f} "
              f"s={g['s']:>4.1f} r={g['r']:>4.1f} h={g['h']:>4.1f} "
              f"t={g['t']:>4.1f} c={g['c']:>4.1f}  ({best['batch']})")

    # Also show stats: how many designs at each c_target have Z in [45,55]?
    print("\nDesign counts per c_target with Z near 50:")
    print(f"{'c':>3}  {'in [45,55]':>10} {'in [40,60]':>10} {'in [35,65]':>10}  total")
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        zlist = []
        for gh in by_c[ci]:
            cps = cps_by_hash[gh]
            try:
                Z, _, _ = loaded_at_f0(cps, J)
                zlist.append(float(Z))
            except Exception:
                continue
        n55 = sum(1 for z in zlist if 45 <= z <= 55)
        n60 = sum(1 for z in zlist if 40 <= z <= 60)
        n65 = sum(1 for z in zlist if 35 <= z <= 65)
        print(f"{ci:>3d}  {n55:>10d} {n60:>10d} {n65:>10d}  {len(zlist)}")


if __name__ == "__main__":
    main()
