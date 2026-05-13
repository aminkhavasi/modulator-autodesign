"""For each c_target, search the cache for the design closest to
(loaded Z0=50, loaded n_eff=3.88) and report its BW.  Compare to the
overall-best-BW design.  Also check whether (Z=50, n=3.88) is even
geometrically possible given the junction loading.
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

    by_c = defaultdict(dict)
    for r in rows:
        gh = r["geometry_hash"]
        if gh in cps_by_hash and gh not in by_c[r["c_target_index"]]:
            by_c[r["c_target_index"]][gh] = r

    print(f"{'c':>3} {'C_pp':>5} {'R_pp':>5} {'wRC':>5} "
          f"{'C_eff_junc':>10} {'C_lim':>6} {'C_loaded_target':>16}  "
          f"{'reachable?':>11}")
    print(f"{'':>3} {'pF/cm':>5} {'mΩcm':>5} {'':>5} "
          f"{'pF/cm':>10} {'pF/cm':>6} {'pF/cm':>16}  "
          f"{'':>11}")
    f0 = 25e9
    omega = 2 * np.pi * f0
    c_cm_s = 3e10
    c_loaded_target = 3.88 / (c_cm_s * 50.0) * 1e12  # pF/cm

    target_reachable = {}
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        C_pn = r0["junction_C_pF_per_cm"]
        R_pn = r0["junction_R_ohm_cm"]
        # SPP-loaded
        C_pp = 0.5 * C_pn
        R_pp = 2.0 * R_pn
        # In SI: R [Ω*m] = R_ohm_cm * 1e-2; C [F/m] = pF/cm * 1e-10
        R_pp_SI = R_pp * 1e-2
        C_pp_SI = C_pp * 1e-10
        # Effective C contribution at f0 = C_pp / (1 + (ωRC)²)
        wRC = omega * R_pp_SI * C_pp_SI
        C_eff_pF_cm = C_pp / (1 + wRC**2)
        # C_loaded_min = C_eff_junc (CPS native C >= 0)
        reachable = C_eff_pF_cm <= c_loaded_target
        target_reachable[ci] = reachable
        print(f"{ci:>3d} {C_pp:>5.2f} {R_pp*1e3:>5.1f} {wRC:>5.2f} "
              f"{C_eff_pF_cm:>10.2f} {C_eff_pF_cm:>6.2f} "
              f"{c_loaded_target:>16.2f}  "
              f"{'YES' if reachable else 'NO  (junction alone over-loads)':>11}")

    # For each c, find designs closest to (Z=50, n=3.88) and compare BW
    print("\n=== Per-c_target: matched design search ===")
    print(f"{'c':>3}  {'BW_best':>7} {'Z_best':>5} {'n_best':>5}  "
          f"{'BW_matched':>10} {'Z_m':>5} {'n_m':>5} {'dist':>5}  "
          f"BW gap")
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
        rows_enr = []
        for gh, r in by_c[ci].items():
            cps = cps_by_hash[gh]
            try:
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
            except Exception:
                continue
            if not (np.isfinite(bw) and np.isfinite(Z0_re)):
                continue
            dist = np.sqrt(((Z0_re-50)/50)**2 + ((n_eff-3.88)/3.88)**2)
            rows_enr.append({
                "Z": float(Z0_re), "n": float(n_eff),
                "BW": float(bw), "dist": float(dist),
                "geom": r["geometry"], "batch": r["batch_id"],
            })
        best_bw = max(rows_enr, key=lambda d: d["BW"])
        best_matched = min(rows_enr, key=lambda d: d["dist"])
        gap = best_bw["BW"] - best_matched["BW"]
        print(f"{ci:>3d}  {best_bw['BW']:>7.2f} {best_bw['Z']:>5.1f} "
              f"{best_bw['n']:>5.2f}  "
              f"{best_matched['BW']:>10.2f} {best_matched['Z']:>5.1f} "
              f"{best_matched['n']:>5.2f} {best_matched['dist']:>5.2f}  "
              f"{gap:+.2f}")
        g = best_matched["geom"]
        print(f"      matched-design geom: g={g['g']:5.1f} ws={g['ws']:4.0f} "
              f"wg={g['wg']:4.0f} s={g['s']:3.1f} r={g['r']:4.1f} "
              f"h={g['h']:3.1f} t={g['t']:3.1f} c={g['c']:4.1f}  "
              f"({best_matched['batch']})")


if __name__ == "__main__":
    main()
