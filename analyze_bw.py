"""Re-rank every c_target=9 design by analytic 3-dB EO bandwidth.

Reuses each design's cached CPSResult (free; no FDTD).  Computes BW
through the same eo_response pipeline used in the bandwidth_sweep.
Then sorts by BW and prints the top/bottom.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from step2.junction import Junction, bandwidth_3dB_GHz, mzm_length_um


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("c_target_index") == 9
            and r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))
            and r["objective"] < 1e9]
    print(f"Loaded {len(rows)} c_target=9 successful rows")

    cache_dir = Path("cache_step2")
    j_inc = Junction(C_pF_per_cm=rows[0]["junction_C_pF_per_cm"],
                     R_ohm_cm=rows[0]["junction_R_ohm_cm"],
                     VpiL_V_cm=rows[0]["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
    L_um = mzm_length_um(target_ER_dB=5.0,
                         VpiL_per_arm_V_cm=j_inc.VpiL_V_cm,
                         V_pp=2.0, push_pull=True)
    print(f"Junction:  C={j_inc.C_pF_per_cm:.2f} pF/cm  "
          f"R={j_inc.R_ohm_cm:.4f} Ohm.cm  L={L_um:.0f} um")

    enriched = []
    for r in rows:
        gh = r["geometry_hash"]
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        if not pkls:
            continue
        try:
            with open(pkls[0], "rb") as f:
                cps = pickle.load(f)
            bw = bandwidth_3dB_GHz(cps, j_inc, L_um)
            alpha_f0 = float(cps.alpha_dB_cm_bare[len(cps.freqs) // 2])
        except Exception:
            continue
        r2 = dict(r)
        r2["BW_GHz"] = bw
        r2["alpha_dB_cm_f0"] = alpha_f0
        enriched.append(r2)

    enriched.sort(key=lambda r: -(r["BW_GHz"] if np.isfinite(r["BW_GHz"]) else -1))

    print(f"\n{'rank':>4} {'BW':>6} {'Z0':>6} {'n_eff':>6} {'alpha':>6} "
          f"{'J':>6}  geom (g/ws/wg/s/r/h/t/c)")
    for i, r in enumerate(enriched[:15]):
        g = r["geometry"]
        print(f"{i+1:>4} {r['BW_GHz']:>6.2f} {r['Z0_re_f0']:>6.2f} "
              f"{r['n_eff_f0']:>6.2f} {r['alpha_dB_cm_f0']:>6.2f} "
              f"{r['objective']:>6.3f}  "
              f"{g['g']:>5.1f}/{g['ws']:>5.0f}/{g['wg']:>5.0f}/"
              f"{g['s']:>4.1f}/{g['r']:>4.1f}/{g['h']:>4.1f}/"
              f"{g['t']:>4.1f}/{g['c']:>4.1f}")
    print(f"\n... {len(enriched) - 15} more, bottom 3:")
    for r in enriched[-3:]:
        g = r["geometry"]
        print(f"     {r['BW_GHz']:>6.2f} {r['Z0_re_f0']:>6.2f} "
              f"{r['n_eff_f0']:>6.2f} {r['alpha_dB_cm_f0']:>6.2f} "
              f"{r['objective']:>6.3f}  "
              f"{g['g']:>5.1f}/{g['ws']:>5.0f}/{g['wg']:>5.0f}/"
              f"{g['s']:>4.1f}/{g['r']:>4.1f}/{g['h']:>4.1f}/"
              f"{g['t']:>4.1f}/{g['c']:>4.1f}")

    # Correlations: BW vs each parameter, and vs Z0/alpha
    print("\n=== Correlations of BW with design quantities ===")
    bws = np.array([r["BW_GHz"] for r in enriched
                    if np.isfinite(r["BW_GHz"])])
    for key in ["Z0_re_f0", "n_eff_f0", "alpha_dB_cm_f0"]:
        vals = np.array([r[key] for r in enriched
                         if np.isfinite(r["BW_GHz"])])
        c = np.corrcoef(vals, bws)[0, 1]
        print(f"  {key:>18}: r = {c:+.2f}")
    for k in ["g", "ws", "wg", "s", "r", "h", "t", "c"]:
        vals = np.array([r["geometry"][k] for r in enriched
                         if np.isfinite(r["BW_GHz"])])
        c = np.corrcoef(vals, bws)[0, 1]
        print(f"  geom.{k:>13}: r = {c:+.2f}")

    # Best design's parameter values
    print("\n=== Best-BW design ===")
    best = enriched[0]
    g = best["geometry"]
    print(f"  BW = {best['BW_GHz']:.2f} GHz, Z0={best['Z0_re_f0']:.2f}, "
          f"n_eff={best['n_eff_f0']:.2f}, alpha={best['alpha_dB_cm_f0']:.2f}")
    print(f"  geom: {g}")

    # Save the enriched ranking for reuse
    out = Path("c9_bw_ranking.json")
    json.dump([{**r, "BW_GHz": float(r["BW_GHz"]) if np.isfinite(r["BW_GHz"])
                else None}
               for r in enriched], open(out, "w"), indent=1, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
