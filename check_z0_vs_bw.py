"""Sanity check: does BW peak at loaded-Z0 = 50 or elsewhere?

For each c_target, scatter loaded-Z0 (at f0, after junction) vs computed
BW for every cached geometry.  Mark Z0=50 to see where the peak is.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
        if gh not in cps_by_hash:
            pkls = [p for p in cache_dir.rglob(f"*{gh}*")
                    if p.suffix == ".pkl"]
            if pkls:
                try:
                    cps_by_hash[gh] = pickle.load(open(pkls[0], "rb"))
                except Exception:
                    pass

    by_c = defaultdict(list)
    for r in rows:
        if r["geometry_hash"] in cps_by_hash:
            by_c[r["c_target_index"]].append(r)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=False, sharey=False)
    for ax, ci in zip(axes.flat, sorted(by_c)):
        if ci < 0:
            continue
        r0 = by_c[ci][0]
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        # Compute (Z0_loaded, BW) for unique geometries at this c_target
        seen = set()
        Zs, BWs, ns = [], [], []
        for r in by_c[ci]:
            gh = r["geometry_hash"]
            if gh in seen:
                continue
            seen.add(gh)
            cps = cps_by_hash[gh]
            try:
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
                if not np.isfinite(bw) or not np.isfinite(Z0_re):
                    continue
                Zs.append(float(Z0_re))
                ns.append(float(n_eff))
                BWs.append(float(bw))
            except Exception:
                continue

        Zs = np.array(Zs); BWs = np.array(BWs); ns = np.array(ns)
        sc = ax.scatter(Zs, BWs, c=ns, s=12, cmap="viridis", alpha=0.7)
        ax.axvline(50, color="red", lw=1.2, ls="--", alpha=0.7)
        # Find best Z0 numerically
        if len(BWs) > 5:
            order = np.argsort(Zs)
            Zs_s, BWs_s = Zs[order], BWs[order]
            i_best = int(np.argmax(BWs_s))
            ax.plot(Zs_s[i_best], BWs_s[i_best], "r*",
                    markersize=14, markeredgecolor="black")
            best_Z = Zs_s[i_best]
            ax.set_title(f"c={ci}  C={J.C_pF_per_cm:.1f}\n"
                         f"L={L_um:.0f}um  best_Z={best_Z:.0f}",
                         fontsize=9)
        else:
            ax.set_title(f"c={ci}", fontsize=9)
        ax.set_xlabel("Loaded Z0 (Ω)", fontsize=8)
        ax.set_ylabel("BW (GHz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max(Zs.max(), 100) + 5)

    fig.colorbar(sc, ax=axes, label="n_eff_RF (loaded)", shrink=0.6)
    fig.suptitle("BW vs loaded-Z0 per c_target  "
                 "(red dashed = 50 Ω; red star = max BW point)",
                 y=0.99)
    fig.savefig("field_plots/z0_vs_bw_check.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote field_plots/z0_vs_bw_check.png")


if __name__ == "__main__":
    main()
