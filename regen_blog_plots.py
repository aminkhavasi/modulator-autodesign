"""Regenerate the two blog plots with the BW-phase matched (Z≈50) results:

  field_plots/step2_BW_vs_efficiency.png  — BW vs 1/VπL with new numbers
  field_plots/step2_EO_S21_best.png       — EO S21 magnitude of best per c_target
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from step2.junction import (Junction, apply_junction_loading, eo_response,
                            bandwidth_3dB_GHz, loaded_at_f0, mzm_length_um)


# Pick the best-BW design at each c_target with loaded Z0 close to 50 Ω.
Z_WINDOW = (45, 55)

OUT_BW   = Path("field_plots/step2_BW_vs_efficiency.png")
OUT_S21  = Path("field_plots/step2_EO_S21_best.png")


def pick_matched_best():
    rows = [json.loads(l) for l in open("step2_journal.jsonl", encoding="utf-8")]
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

    out = []
    for ci in sorted(by_c):
        if ci < 0:
            continue
        r0 = next(iter(by_c[ci].values()))
        J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                     R_ohm_cm=r0["junction_R_ohm_cm"],
                     VpiL_V_cm=r0["junction_VpiL_V_cm"],
                     n_group_opt=3.88)
        L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)

        cands = []
        for gh, r in by_c[ci].items():
            cps = cps_by_hash[gh]
            try:
                Z0_re, n_eff, _ = loaded_at_f0(cps, J)
                bw = bandwidth_3dB_GHz(cps, J, L_um)
            except Exception:
                continue
            if not (np.isfinite(bw) and np.isfinite(Z0_re)):
                continue
            cands.append({
                "Z": float(Z0_re), "n": float(n_eff), "BW": float(bw),
                "cps": cps, "geom": r["geometry"],
                "hash": gh, "batch": r["batch_id"],
            })
        # try tight Z window, then widen
        for window in [Z_WINDOW, (40, 60), (35, 65), (0, 200)]:
            lo, hi = window
            sub = [d for d in cands if lo <= d["Z"] <= hi]
            if sub:
                best = max(sub, key=lambda d: d["BW"])
                break
        out.append({
            "c_idx": ci,
            "C_pF_per_cm": J.C_pF_per_cm,
            "R_ohm_cm": J.R_ohm_cm,
            "VpiL_V_cm": J.VpiL_V_cm,
            "MZM_length_um": L_um,
            "Z0": best["Z"], "n_eff": best["n"], "BW_GHz": best["BW"],
            "geometry": best["geom"],
            "cps": best["cps"],
            "junction": J,
        })
    return out


def plot_BW_vs_eff(designs):
    Cs = np.array([d["C_pF_per_cm"] for d in designs])
    VpiL = np.array([d["VpiL_V_cm"] for d in designs])
    BW = np.array([d["BW_GHz"] for d in designs])
    L_um = np.array([d["MZM_length_um"] for d in designs])
    eff = 1.0 / VpiL

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    sc = ax.scatter(eff, BW, c=Cs, s=110, cmap="viridis",
                    edgecolor="black", linewidth=0.6, zorder=3)
    # Annotate
    offsets = {
        0: (10, 6), 1: (10, 6), 2: (8, -16), 3: (10, 6),
        4: (10, -10), 5: (-90, -10), 6: (10, 6),
        7: (10, 6), 8: (10, 6), 9: (-105, 4),
    }
    for d, (dx, dy) in zip(designs, [offsets[d["c_idx"]] for d in designs]):
        ax.annotate(
            f"C={d['C_pF_per_cm']:.1f}, L={d['MZM_length_um']:.0f} μm",
            (1.0 / d["VpiL_V_cm"], d["BW_GHz"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, color="#333", ha="left")

    ax.set_xlabel("1 / VπL  [(V·cm)⁻¹] — modulation efficiency")
    ax.set_ylabel("3-dB EO bandwidth (GHz)")
    ax.set_title("Achievable BW–efficiency frontier (10 modulators, "
                 "ER = 5 dB, V_pp = 2 V push-pull)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Junction capacitance C (pF/cm)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.4, max(eff) * 1.20)
    ax.set_ylim(min(BW) * 0.93, max(BW) * 1.05)
    fig.tight_layout()
    fig.savefig(OUT_BW, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_BW}")


def plot_EO_S21(designs):
    from scipy.ndimage import gaussian_filter1d

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    Cs = np.array([d["C_pF_per_cm"] for d in designs])
    norm = Normalize(vmin=Cs.min(), vmax=Cs.max())
    cmap = plt.get_cmap("viridis")

    seen_C = set()
    for d in designs:
        # Skip the duplicate C=14.07 (c=8 same as c=7)
        if d["C_pF_per_cm"] in seen_C:
            continue
        seen_C.add(d["C_pF_per_cm"])

        freqs_ext, H_total, _ = eo_response(d["cps"], d["junction"],
                                            d["MZM_length_um"])
        # Skip DC anchor for plotting
        f = freqs_ext[1:] / 1e9
        H_dB = 20.0 * np.log10(np.abs(H_total[1:]) / np.abs(H_total[0]))
        # Interpolate to dense grid and smooth lightly
        f_dense = np.linspace(f[0], f[-1], 400)
        H_dense = np.interp(f_dense, f, H_dB)
        H_dense = gaussian_filter1d(H_dense, sigma=8, mode="nearest")

        color = cmap(norm(d["C_pF_per_cm"]))
        ax.plot(f_dense, H_dense, color=color, lw=2.0,
                label=f"C={d['C_pF_per_cm']:.1f}, BW={d['BW_GHz']:.1f} GHz")

    ax.axhline(-3, color="black", ls="--", lw=1.0, alpha=0.6, label="−3 dB")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("EO |H(f)| / |H(DC)|  (dB)")
    ax.set_title("EO frequency response of the best matched design at each "
                 "operating point")
    ax.set_xlim(0, 50)
    ax.set_ylim(-9, 1.5)
    ax.grid(alpha=0.3)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("C (pF/cm)")
    ax.legend(fontsize=7.5, loc="lower left", ncols=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_S21, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_S21}")


def main():
    designs = pick_matched_best()
    print(f"Picked {len(designs)} c_targets")
    for d in designs:
        print(f"  c={d['c_idx']}  C={d['C_pF_per_cm']:.2f}  BW={d['BW_GHz']:.2f}  "
              f"Z={d['Z0']:.1f}  n={d['n_eff']:.2f}  L={d['MZM_length_um']:.0f} um")

    plot_BW_vs_eff(designs)
    plot_EO_S21(designs)

    # Save a JSON summary too
    summ = []
    for d in designs:
        e = {k: v for k, v in d.items() if k not in ("cps", "junction")}
        e["geometry"] = {k: float(v) for k, v in e["geometry"].items()}
        summ.append(e)
    json.dump(summ, open("step4_matched_summary.json", "w"),
              indent=2, default=str)
    print("Wrote step4_matched_summary.json")


if __name__ == "__main__":
    main()
