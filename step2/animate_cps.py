"""Animate the BO trajectory for one c_target as a GIF.

Top-down view of the segmented CPS T-rail electrode at each BO iteration,
with the loaded Z0 and n_eff_RF annotated.  Frames are read directly from
step2_journal.jsonl.

Usage:
    python -m step2.animate_cps --c-target 1 --out field_plots/cps_evolution.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _load_rows(c_target: int) -> list[dict]:
    rows = []
    for line in open("step2_journal.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("meta") or r.get("c_target_index") != c_target:
            continue
        rows.append(r)
    # Sort by timestamp so the animation follows the actual BO trajectory
    rows.sort(key=lambda r: r.get("timestamp", ""))
    return rows


def _draw_frame(ax, geom: dict, *, n_periods: int = 3, target_n: float):
    """Draw a top-down view of `n_periods` of the segmented CPS."""
    ax.clear()
    g, ws, wg = geom["g"], geom["ws"], geom["wg"]
    s, r, h, t, c = geom["s"], geom["r"], geom["h"], geom["t"], geom["c"]
    period = r + c

    # Rails (drawn as long rectangles along y)
    y_span = n_periods * period
    ymin, ymax = -y_span / 2.0, +y_span / 2.0
    rail_color = "#cc7a33"
    ax.add_patch(mpatches.Rectangle(
        (g / 2.0, ymin), ws, ymax - ymin,
        facecolor=rail_color, edgecolor="black", linewidth=0.5))
    ax.add_patch(mpatches.Rectangle(
        (-g / 2.0 - wg, ymin), wg, ymax - ymin,
        facecolor=rail_color, edgecolor="black", linewidth=0.5))

    # T fingers
    for ii in range(n_periods + 1):
        yy = -y_span / 2.0 + ii * period
        # Right rail T (extends in -x direction toward gap)
        # bar (s wide along x, r long along y) centered at base_x - (h + s/2)
        ax.add_patch(mpatches.Rectangle(
            (g / 2.0 - h - s, yy - r / 2.0), s, r,
            facecolor=rail_color, edgecolor="black", linewidth=0.4))
        # neck (h long along x, t wide along y)
        ax.add_patch(mpatches.Rectangle(
            (g / 2.0 - h, yy - t / 2.0), h, t,
            facecolor=rail_color, edgecolor="black", linewidth=0.4))
        # Left rail T (mirror)
        ax.add_patch(mpatches.Rectangle(
            (-g / 2.0 + h, yy - r / 2.0), s, r,
            facecolor=rail_color, edgecolor="black", linewidth=0.4))
        ax.add_patch(mpatches.Rectangle(
            (-g / 2.0, yy - t / 2.0), h, t,
            facecolor=rail_color, edgecolor="black", linewidth=0.4))

    # Faint optical waveguide marker along x=0
    ax.axvline(0.0, color="#3370cc", lw=1.0, alpha=0.5, ls="--")

    # Fixed extent so frame size doesn't jitter
    ax.set_xlim(-300, 300)
    ax.set_ylim(-200, 200)
    ax.set_aspect("equal")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y (propagation) [μm]")


def animate(c_target: int, outpath: Path, *, fps: float = 1.6):
    rows = _load_rows(c_target)
    if not rows:
        raise SystemExit(f"No rows for c_target={c_target}")

    # Target n_group (optical) — printed so reader sees what the agent is
    # trying to match.  Always 3.88 for this project.
    target_n = 3.88

    fig, ax = plt.subplots(figsize=(9, 5.2))
    title = fig.suptitle("", fontsize=12)

    def update(idx: int):
        r = rows[idx]
        _draw_frame(ax, r["geometry"], target_n=target_n)
        z0 = r.get("Z0_re_f0", float("nan"))
        neff = r.get("n_eff_f0", float("nan"))
        obj = r.get("objective", float("nan"))
        batch = r.get("batch_id", "?")
        failed = r.get("failed", False)
        status = " [FAILED]" if failed else ""
        title.set_text(
            f"Iteration {idx + 1}/{len(rows)}   "
            f"({batch}){status}\n"
            f"Z₀ = {z0:5.1f} Ω    "
            f"n_eff,RF = {neff:4.2f}   (target {target_n})    "
            f"J = {obj:.3f}"
        )
        return ()

    anim = FuncAnimation(fig, update, frames=len(rows), blit=False)
    outpath.parent.mkdir(exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(outpath, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Wrote {outpath} ({len(rows)} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-target", type=int, default=1)
    p.add_argument("--out", type=Path,
                   default=Path("field_plots/cps_evolution.gif"))
    p.add_argument("--fps", type=float, default=1.6)
    args = p.parse_args()
    animate(args.c_target, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
