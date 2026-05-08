"""Trade-off plots from journal.jsonl.

Generates four PNGs in field_plots/:
  tradeoff_VpiL_C.png       Primary plot: VpiL vs C, with Pareto frontier.
  tradeoff_VpiL_loss.png    VpiL vs loss.
  tradeoff_C_bandwidth.png  C vs RC bandwidth.
  mult_sweep.png            Each FoM vs mult at target_v=0 (closest available).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

JOURNAL = Path("journal.jsonl")
OUTDIR = Path("field_plots")


def load_journal(path: Path = JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _valid(history: list[dict]) -> list[dict]:
    return [e for e in history if e.get("VpiL_V_cm") is not None]


def _pareto_min_min(points: list[tuple[float, float]]
                    ) -> list[tuple[float, float]]:
    """Pareto front minimizing both x and y."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_y = math.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def plot_VpiL_vs_C(history: list[dict], outpath: Path):
    valid = _valid(history)
    if not valid:
        print("No valid VpiL points -- skipping VpiL vs C plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    mults = sorted({e["mult"] for e in valid})
    cmap = plt.get_cmap("viridis")
    for m in mults:
        pts = [e for e in valid if e["mult"] == m]
        xs = [p["C_pF_per_cm"] for p in pts]
        ys = [p["VpiL_V_cm"] for p in pts]
        # log-normalize mult for coloring
        norm = (math.log10(m) - math.log10(min(mults))
                ) / max(1e-9, math.log10(max(mults)) - math.log10(min(mults)))
        ax.scatter(xs, ys, color=cmap(norm), s=40,
                   label=f"mult={m:g}", edgecolor="k", linewidth=0.4)

    front = _pareto_min_min(
        [(e["C_pF_per_cm"], e["VpiL_V_cm"]) for e in valid]
    )
    if len(front) >= 2:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "k--", lw=1, label="Pareto frontier")

    ax.set_xlabel("Capacitance (pF/cm)")
    ax.set_ylabel("VπL (V·cm)")
    ax.set_title("VπL vs C trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_VpiL_vs_loss(history: list[dict], outpath: Path):
    valid = _valid(history)
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in sorted({e["mult"] for e in valid}):
        pts = [e for e in valid if e["mult"] == m]
        ax.scatter([p["loss_dB_per_cm"] for p in pts],
                   [p["VpiL_V_cm"] for p in pts],
                   s=40, label=f"mult={m:g}", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Optical loss (dB/cm)")
    ax.set_ylabel("VπL (V·cm)")
    ax.set_title("VπL vs loss trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_C_vs_bandwidth(history: list[dict], outpath: Path):
    valid = _valid(history)
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in sorted({e["mult"] for e in valid}):
        pts = [e for e in valid if e["mult"] == m]
        ax.scatter([p["C_pF_per_cm"] for p in pts],
                   [p["f3dB_GHz"] for p in pts],
                   s=40, label=f"mult={m:g}", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Capacitance (pF/cm)")
    ax.set_ylabel("RC bandwidth (GHz)")
    ax.set_title("C vs RC-limited bandwidth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_mult_sweep(history: list[dict], outpath: Path,
                    target_v: float = 0.0):
    """Each FoM vs mult, at the journal point closest to target_v."""
    valid = _valid(history)
    if not valid:
        return

    # For each mult, pick the row whose target_v is closest to the requested
    rows_per_mult = {}
    for e in valid:
        m = e["mult"]
        if m not in rows_per_mult or (abs(e["target_v"] - target_v)
                                      < abs(rows_per_mult[m]["target_v"] - target_v)):
            rows_per_mult[m] = e
    if not rows_per_mult:
        return

    items = sorted(rows_per_mult.items())
    mults = [m for m, _ in items]
    rows = [r for _, r in items]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()
    series = [
        ("VπL (V·cm)",          [r["VpiL_V_cm"]      for r in rows]),
        ("Capacitance (pF/cm)", [r["C_pF_per_cm"]    for r in rows]),
        ("RC bandwidth (GHz)",  [r["f3dB_GHz"]       for r in rows]),
        ("Loss (dB/cm)",        [r["loss_dB_per_cm"] for r in rows]),
    ]
    for ax, (label, ys) in zip(axes, series):
        ax.plot(mults, ys, "o-")
        ax.set_xscale("log")
        ax.set_xlabel("Doping multiplier")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"FoMs vs doping multiplier (target_v ≈ {target_v} V)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    OUTDIR.mkdir(exist_ok=True)
    history = load_journal()
    if not history:
        print(f"No journal entries in {JOURNAL} -- run an evaluation first.")
        return
    plot_VpiL_vs_C(history, OUTDIR / "tradeoff_VpiL_C.png")
    plot_VpiL_vs_loss(history, OUTDIR / "tradeoff_VpiL_loss.png")
    plot_C_vs_bandwidth(history, OUTDIR / "tradeoff_C_bandwidth.png")
    plot_mult_sweep(history, OUTDIR / "mult_sweep.png", target_v=0.0)


if __name__ == "__main__":
    main()
