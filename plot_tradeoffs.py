"""Trade-off plots from journal.jsonl.

Generates four PNGs in field_plots/:
  tradeoff_VpiL_C.png       Primary plot: VpiL vs C, with Pareto frontier.
  tradeoff_VpiL_loss.png    VpiL vs loss.
  tradeoff_C_bandwidth.png  C vs RC bandwidth.
  mult_sweep.png            Each FoM vs mult at target_v=0 (closest available).

Styling convention for scatter plots:
  - color  = `mult` (log-normalized, viridis colormap)
  - marker = `target_v` (one symbol per discrete bias point)

By default only interior voltages are plotted (the 7 trade-off points).
Pass --include-endpoints to also show -0.5 and 1.5 V (useful for diagnostics).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

JOURNAL = Path("journal.jsonl")
OUTDIR = Path("field_plots")

# Marker shapes for discrete target_v values.  Up to 9 distinct shapes.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def load_journal(path: Path = JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _filter(history: list[dict], *, include_endpoints: bool) -> list[dict]:
    out = [e for e in history if e.get("VpiL_V_cm") is not None]
    if not include_endpoints:
        out = [e for e in out if e.get("is_interior", True)]
    return out


def _color_norm(history: list[dict]):
    """Return a Normalize and cmap for log10(mult) coloring."""
    mults = sorted({e["mult"] for e in history})
    if not mults:
        return None, None
    log_mults = np.log10(mults)
    norm = mcolors.Normalize(vmin=float(np.min(log_mults)),
                             vmax=float(np.max(log_mults)))
    return norm, plt.get_cmap("viridis")


def _marker_for(target_v: float, voltage_list: list[float]) -> str:
    idx = voltage_list.index(round(target_v, 6))
    return _MARKERS[idx % len(_MARKERS)]


def _scatter_with_legend(ax, history, x_key, y_key, *, x_label, y_label, title):
    norm, cmap = _color_norm(history)
    if norm is None:
        return
    voltages = sorted({round(e["target_v"], 6) for e in history})

    for e in history:
        ax.scatter(
            e[x_key], e[y_key],
            color=cmap(norm(np.log10(e["mult"]))),
            marker=_marker_for(e["target_v"], voltages),
            s=55, edgecolor="k", linewidth=0.4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Colorbar for mult
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label("log10(mult)")

    # Marker legend for target_v
    handles = [plt.Line2D([0], [0], marker=_MARKERS[i % len(_MARKERS)],
                          color="w", markerfacecolor="grey",
                          markeredgecolor="k", markersize=8,
                          label=f"V={v:+.2f}")
               for i, v in enumerate(voltages)]
    ax.legend(handles=handles, fontsize=8, loc="best",
              title="target_v", framealpha=0.85)


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


def plot_VpiL_vs_C(history: list[dict], outpath: Path, *,
                   include_endpoints: bool):
    pts = _filter(history, include_endpoints=include_endpoints)
    if not pts:
        print("No valid points -- skipping VpiL vs C plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_with_legend(
        ax, pts,
        x_key="C_pF_per_cm", y_key="VpiL_V_cm",
        x_label="Capacitance (pF/cm)",
        y_label="VpiL (V.cm)",
        title="VpiL vs C trade-off",
    )

    # Pareto frontier overlay (interior points only, regardless of flag)
    interior = [e for e in pts if e.get("is_interior", True)]
    front = _pareto_min_min([(e["C_pF_per_cm"], e["VpiL_V_cm"])
                             for e in interior])
    if len(front) >= 2:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "k--", lw=1.2, label="Pareto frontier", zorder=3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_VpiL_vs_loss(history: list[dict], outpath: Path, *,
                      include_endpoints: bool):
    pts = _filter(history, include_endpoints=include_endpoints)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_with_legend(
        ax, pts,
        x_key="loss_dB_per_cm", y_key="VpiL_V_cm",
        x_label="Optical loss (dB/cm)",
        y_label="VpiL (V.cm)",
        title="VpiL vs loss trade-off",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_C_vs_bandwidth(history: list[dict], outpath: Path, *,
                        include_endpoints: bool):
    pts = _filter(history, include_endpoints=include_endpoints)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_with_legend(
        ax, pts,
        x_key="C_pF_per_cm", y_key="f3dB_GHz",
        x_label="Capacitance (pF/cm)",
        y_label="RC bandwidth (GHz)",
        title="C vs RC-limited bandwidth",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_mult_sweep(history: list[dict], outpath: Path,
                    target_v: float = 0.0, *,
                    include_endpoints: bool):
    """Each FoM vs mult, at the journal point closest to target_v."""
    pts = _filter(history, include_endpoints=include_endpoints)
    if not pts:
        return

    rows_per_mult = {}
    for e in pts:
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
        ("VpiL (V.cm)",         [r["VpiL_V_cm"]      for r in rows]),
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
    fig.suptitle(f"FoMs vs doping multiplier (target_v ~ {target_v} V)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-endpoints", action="store_true",
                   help="include the -0.5 V and 1.5 V endpoint rows")
    args = p.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    history = load_journal()
    if not history:
        print(f"No journal entries in {JOURNAL} -- run an evaluation first.")
        return
    inc = args.include_endpoints
    plot_VpiL_vs_C(history, OUTDIR / "tradeoff_VpiL_C.png",
                   include_endpoints=inc)
    plot_VpiL_vs_loss(history, OUTDIR / "tradeoff_VpiL_loss.png",
                      include_endpoints=inc)
    plot_C_vs_bandwidth(history, OUTDIR / "tradeoff_C_bandwidth.png",
                        include_endpoints=inc)
    plot_mult_sweep(history, OUTDIR / "mult_sweep.png", target_v=0.0,
                    include_endpoints=inc)


if __name__ == "__main__":
    main()
