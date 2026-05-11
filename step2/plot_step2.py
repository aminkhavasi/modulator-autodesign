"""Step-2 plots: bandwidth-vs-C, EO S21, BO convergence per C.

Reads:
  step2_journal.jsonl
  step2_targets.json
  step2_bandwidth_sweep.json (produced by `run_batch.py bandwidth_sweep`)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geom import CPSGeometry
from .journal import (filter_successes, per_C_history)
from .junction import Junction, eo_response
from .select_C_targets import read_targets
from .simulate import evaluate_cps

OUTDIR = Path("field_plots")


def plot_BW_vs_C(sweep_path: Path = Path("step2_bandwidth_sweep.json"),
                 outpath: Path | None = None):
    """3-dB EO bandwidth vs junction capacitance (one point per C target)."""
    if not sweep_path.exists():
        print(f"{sweep_path} not found.  Run "
              "`python -m step2.run_batch bandwidth_sweep` first.")
        return
    data = json.loads(sweep_path.read_text())
    if not data:
        print("No bandwidth-sweep data.")
        return

    Cs = [d["C_pF_per_cm"] for d in data]
    BWs = [d["bandwidth_3dB_GHz"] for d in data]
    VpiLs = [d["VpiL_V_cm"] for d in data]
    Ls = [d["MZM_length_um"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(Cs, BWs, "o-")
    axes[0].set_xlabel("Junction C (pF/cm)")
    axes[0].set_ylabel("3-dB EO bandwidth (GHz)")
    axes[0].set_title("Bandwidth vs C")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Cs, VpiLs, "o-")
    axes[1].set_xlabel("Junction C (pF/cm)")
    axes[1].set_ylabel("|VpiL| (V.cm)")
    axes[1].set_title("VpiL vs C (Step-1 lower envelope)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(Cs, np.array(Ls) / 1000.0, "o-")
    axes[2].set_xlabel("Junction C (pF/cm)")
    axes[2].set_ylabel("MZM length (mm)")
    axes[2].set_title("MZM length for 5 dB ER vs C")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if outpath is None:
        outpath = OUTDIR / "step2_BW_vs_C.png"
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_BO_convergence(c_target: int, outpath: Path | None = None):
    """Best-so-far objective vs evaluation index for one C target."""
    rows = [r for r in per_C_history(c_target) if not r.get("meta")]
    successes = filter_successes(rows)
    if len(successes) < 2:
        print(f"Not enough data for c_target={c_target}.")
        return
    objs = [r["objective"] for r in successes]
    best = []
    running = math.inf
    for o in objs:
        running = min(running, o)
        best.append(running)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(objs) + 1), objs, "o", alpha=0.5,
            label="per-eval objective")
    ax.plot(np.arange(1, len(best) + 1), best, "k-", label="best so far")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.set_title(f"BO convergence: C target #{c_target}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if outpath is None:
        outpath = OUTDIR / f"step2_BO_convergence_c{c_target}.png"
    outpath.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_EO_S21_best(outpath: Path | None = None):
    """EO S21 magnitude for each C target's best design, overlaid."""
    from scipy.ndimage import gaussian_filter1d
    targets = read_targets()
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    seen_C: set[str] = set()
    for t in targets:
        c_idx = t["c_target_index"]
        rows = filter_successes(
            [r for r in per_C_history(c_idx) if not r.get("meta")]
        )
        if not rows:
            continue
        best = min(rows, key=lambda r: r["objective"])
        # Pull cached CPSResult
        cps = evaluate_cps(CPSGeometry(**best["geometry"]))
        junction = Junction(
            C_pF_per_cm=t["C_pF_per_cm"],
            R_ohm_cm=t["R_ohm_cm"],
            VpiL_V_cm=t["VpiL_V_cm"],
            n_group_opt=3.88,
        )
        # Compute MZM length for 5 dB ER
        from .junction import mzm_length_um as _mzm
        L_um = _mzm(5.0, junction.VpiL_V_cm, 2.0, push_pull=True)
        f_ext, H_total, _ = eo_response(cps, junction, L_um)
        H_dB = 20 * np.log10(np.abs(H_total) / np.abs(H_total[0]))
        # Cosmetic: drop the 1-Hz extrapolation point (causes a visible kink
        # to the first FDTD frequency since there is no data in between),
        # interpolate onto a dense uniform grid, then Savitzky-Golay smooth
        # to remove FDTD numerical ripple in gamma(f) that propagates through
        # the loaded-line transfer function. Physics is untouched.
        f_plot_GHz = f_ext[1:] / 1e9
        H_plot = H_dB[1:]
        f_dense = np.linspace(f_plot_GHz.min(), f_plot_GHz.max(), 400)
        H_dense = np.interp(f_dense, f_plot_GHz, H_plot)
        # Gaussian low-pass kills the ~5 GHz-period numerical ripple while
        # leaving the slow rolloff (>15 GHz feature scale) intact.
        H_dense = gaussian_filter1d(H_dense, sigma=15, mode="nearest")
        color = cmap(c_idx / max(1, len(targets) - 1))
        c_label = f"C={junction.C_pF_per_cm:.2f}"
        if c_label in seen_C:
            ax.plot(f_dense, H_dense, color=color)
        else:
            ax.plot(f_dense, H_dense, color=color, label=c_label)
            seen_C.add(c_label)
    ax.axhline(-3, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|EO S21| (dB, normalized)")
    ax.set_title("EO frequency response of best design per C")
    ax.set_xlim(0, 60)
    ax.set_ylim(-10, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    if outpath is None:
        outpath = OUTDIR / "step2_EO_S21_best.png"
    outpath.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_BW_vs_efficiency(outpath: Path | None = None):
    """Bandwidth vs modulation efficiency (1/VπL), colored by junction C."""
    sweep_path = Path("step2_bandwidth_sweep.json")
    rows = json.loads(sweep_path.read_text())
    # Dedupe c_target rows that share a junction (e.g. c_idx 7 == 8)
    seen: set[tuple[float, float]] = set()
    pts = []
    for r in rows:
        key = (round(r["C_pF_per_cm"], 4), round(r["VpiL_V_cm"], 4))
        if key in seen:
            continue
        seen.add(key)
        pts.append(r)

    inv_VpiL = np.array([1.0 / r["VpiL_V_cm"] for r in pts])
    bw = np.array([r["bandwidth_3dB_GHz"] for r in pts])
    C = np.array([r["C_pF_per_cm"] for r in pts])
    L_um = np.array([r["MZM_length_um"] for r in pts])
    order = np.argsort(inv_VpiL)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(inv_VpiL[order], bw[order], color="gray", lw=1.5, alpha=0.6, zorder=1)
    sc = ax.scatter(inv_VpiL, bw, c=C, cmap="viridis", s=140,
                    edgecolor="black", linewidth=0.8, zorder=2)
    # Hand-tuned label placement (offset_x_pts, offset_y_pts, ha) keyed by
    # C, so each point's annotation steers clear of neighbours and the
    # "higher BW" / "higher efficiency" callouts.
    # For each close vertical pair, place the higher-BW point's label
    # ABOVE and on one side, the lower-BW point's label BELOW and on the
    # opposite side, so the texts are vertically and horizontally apart.
    def _offset(c_val: float) -> tuple[int, int, str]:
        if c_val < 3.5:    return ( 10,   0, "left")   # C=2.9  right
        if c_val < 5:      return (-10,   0, "right")  # C=4.0  left
        if c_val < 7:      return ( 10,   0, "left")   # C=6.3  right
        if c_val < 8:      return (-10,  12, "right")  # C=7.6  above-left
        if c_val < 10:     return ( 10, -14, "left")   # C=9.0  below-right
        if c_val < 11:     return (-10,  12, "right")  # C=10.4 above-left
        if c_val < 13:     return ( 10, -14, "left")   # C=12.1 below-right
        if c_val < 15:     return (-10,  12, "right")  # C=14.1 above-left
        return ( 10, -14, "left")                      # C=16.5 below-right
    for x, y, L, c in zip(inv_VpiL, bw, L_um, C):
        dx, dy, ha = _offset(c)
        ax.annotate(f"C = {c:.1f}, L = {L:.0f} μm", (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, va="center", ha=ha)
    ax.annotate("higher BW\nlonger device,\nlower efficiency",
                xy=(inv_VpiL[order][0], bw[order][0]),
                xytext=(inv_VpiL[order][0] + 0.4, bw[order][0] - 4),
                fontsize=10, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.annotate("higher efficiency\nshorter device,\nlower BW",
                xy=(inv_VpiL[order][-1], bw[order][-1]),
                xytext=(inv_VpiL[order][-1] - 1.1, bw[order][-1] + 5),
                fontsize=10, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))
    ax.set_xlim(inv_VpiL.min() - 0.15, inv_VpiL.max() + 0.7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Junction $C$ [pF/cm]")
    ax.set_xlabel(r"Modulation efficiency  $1/V_\pi L$   [(V·cm)$^{-1}$]")
    ax.set_ylabel("EO 3-dB bandwidth  [GHz]")
    ax.set_title("The bandwidth-efficiency trade-off: each point is one\n"
                 "autonomously optimized segmented CPS electrode on its own PN junction")
    ax.grid(True, alpha=0.3)
    if outpath is None:
        outpath = OUTDIR / "step2_BW_vs_efficiency.png"
    outpath.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bw", action="store_true", help="bandwidth-vs-C plot")
    p.add_argument("--convergence", type=int, metavar="C_IDX",
                   help="BO convergence for one C target")
    p.add_argument("--eo", action="store_true", help="EO S21 of best designs")
    p.add_argument("--bweff", action="store_true",
                   help="bandwidth-vs-efficiency Pareto plot")
    p.add_argument("--all", action="store_true", help="all of the above")
    args = p.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    if args.all or args.bw:
        plot_BW_vs_C()
    if args.all or args.eo:
        plot_EO_S21_best()
    if args.all or args.bweff:
        plot_BW_vs_efficiency()
    if args.convergence is not None:
        plot_BO_convergence(args.convergence)


if __name__ == "__main__":
    main()
