"""Finalize the BW-pivot campaign: build the best-per-c_target table,
write a JSON, and plot BW-vs-C (J-phase vs BW-phase).
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from step2.junction import Junction, bandwidth_3dB_GHz, mzm_length_um


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))
            and r["objective"] < 1e9]

    # 1. Per-c_target build {best-BW design, best-J design}
    cache_dir = Path("cache_step2")
    cps_by_hash = {}
    for r in rows:
        gh = r["geometry_hash"]
        if gh in cps_by_hash:
            continue
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        if pkls:
            try:
                with open(pkls[0], "rb") as f:
                    cps_by_hash[gh] = pickle.load(f)
            except Exception:
                pass

    by_c = defaultdict(list)
    for r in rows:
        if r["geometry_hash"] in cps_by_hash:
            by_c[r["c_target_index"]].append(r)

    summary = []
    for ci in sorted(k for k in by_c.keys() if k >= 0):
        rs = by_c[ci]
        r0 = rs[0]
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        rs_bw = []
        for r in rs:
            bw = bandwidth_3dB_GHz(cps_by_hash[r["geometry_hash"]], J, L_um)
            if np.isfinite(bw):
                rs_bw.append({**r, "BW_GHz_final": float(bw)})
        if not rs_bw:
            continue

        # J-phase designs only (batch_id starts with lhs or bo_ — exclude
        # bw_*, relaxed_*, cross_eval):
        j_phase = [r for r in rs_bw
                   if (r.get("batch_id", "").startswith("lhs")
                       or r.get("batch_id", "").startswith("bo_"))]
        best_J = (min(j_phase, key=lambda r: r["objective"])
                  if j_phase else None)
        # Best BW overall
        best_BW = max(rs_bw, key=lambda r: r["BW_GHz_final"])

        summary.append({
            "c_idx": ci,
            "C_target_pF_per_cm": r0["target_C_pF_per_cm"],
            "junction_C_pF_per_cm": r0["junction_C_pF_per_cm"],
            "junction_R_ohm_cm": r0["junction_R_ohm_cm"],
            "VpiL_V_cm": r0["junction_VpiL_V_cm"],
            "MZM_length_um": L_um,
            "J_phase_best_BW_GHz": best_J["BW_GHz_final"] if best_J else None,
            "J_phase_best_Z0": best_J["Z0_re_f0"] if best_J else None,
            "J_phase_best_n_eff": best_J["n_eff_f0"] if best_J else None,
            "BW_phase_best_BW_GHz": best_BW["BW_GHz_final"],
            "BW_phase_best_Z0": best_BW["Z0_re_f0"],
            "BW_phase_best_n_eff": best_BW["n_eff_f0"],
            "BW_phase_best_geometry": best_BW["geometry"],
            "BW_phase_best_batch": best_BW["batch_id"],
            "BW_phase_best_hash": best_BW["geometry_hash"],
            "uplift_GHz": (best_BW["BW_GHz_final"]
                           - best_J["BW_GHz_final"]) if best_J else None,
        })

    json.dump(summary, open("step4_bw_summary.json", "w"),
              indent=2, default=str)
    print("Wrote step4_bw_summary.json\n")

    # Pretty print
    print(f"{'c':>3} {'C':>6} {'L_um':>5}  "
          f"{'J_BW':>5} {'BW_BW':>5} {'+GHz':>5}  "
          f"{'Z0':>5} {'n':>4}  geom (g/ws/wg/r/c)")
    for s in summary:
        g = s["BW_phase_best_geometry"]
        print(f"{s['c_idx']:>3} {s['junction_C_pF_per_cm']:>6.2f} "
              f"{s['MZM_length_um']:>5.0f}  "
              f"{s['J_phase_best_BW_GHz']:>5.2f} "
              f"{s['BW_phase_best_BW_GHz']:>5.2f} "
              f"{s['uplift_GHz']:>+5.2f}  "
              f"{s['BW_phase_best_Z0']:>5.1f} "
              f"{s['BW_phase_best_n_eff']:>4.1f}  "
              f"{g['g']:>5.1f}/{g['ws']:>4.0f}/{g['wg']:>4.0f}/"
              f"{g['r']:>4.1f}/{g['c']:>4.1f}")

    # Plot
    Cs = [s["junction_C_pF_per_cm"] for s in summary]
    bw_j = [s["J_phase_best_BW_GHz"] for s in summary]
    bw_w = [s["BW_phase_best_BW_GHz"] for s in summary]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(Cs, bw_j, "o--", label="J-phase best (objective: Z0+n_eff)",
            color="#7d7d7d", markerfacecolor="white", markersize=8, lw=1.5)
    ax.plot(Cs, bw_w, "s-", label="BW-phase best (objective: BW, V2 rules)",
            color="#cc7a33", markersize=9, lw=2.2)
    for s in summary:
        ax.annotate(f"+{s['uplift_GHz']:.1f}",
                    (s["junction_C_pF_per_cm"], s["BW_phase_best_BW_GHz"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color="#cc5500")
    ax.set_xlabel("Junction C (pF / cm)")
    ax.set_ylabel("3-dB EO bandwidth (GHz)")
    ax.set_title("BW vs C: pivoting from J to BW objective\n"
                 "(plus RULES_V2 5 μm rail/gap minimums)")
    ax.set_xlim(0, 17.5)
    ax.set_ylim(15, 45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("field_plots/step4_BW_vs_C_pivot.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote field_plots/step4_BW_vs_C_pivot.png")


if __name__ == "__main__":
    main()
