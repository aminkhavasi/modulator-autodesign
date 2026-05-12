"""Two-stage design-loop schematic for the blog post.

Renders a flow diagram that mirrors the structure of the autonomous run:
 - Step 1: doping sweep (charge sim -> mode solve -> (VpiL, C) envelope)
 - Step 2 (per C target): propose electrode -> DRC -> RF FDTD ->
   ABCD + junction load -> Z0, n_eff, BW; loop until budget spent.

Saves to field_plots/design_loop.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("field_plots/design_loop.png")

# Palette
C_STEP1 = "#3d6fb2"     # blue for Step 1 (junction)
C_STEP2 = "#cc7a33"     # orange for Step 2 (electrode)
C_GATE  = "#888888"     # grey for DRC/gates
C_OUT   = "#2b8a3e"     # green for outputs
C_TEXT  = "#1a1a1a"


def _box(ax, x, y, w, h, label, color, *, fc=None, text_color="white",
         fontsize=10, italic=False):
    fc = fc if fc is not None else color
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.2, edgecolor=color, facecolor=fc)
    ax.add_patch(box)
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontstyle=style, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.5, style="->",
           connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 5.6))

    # --- STEP 1 column (left) ---
    x1 = 0.4   # left edge
    w  = 2.6   # box width
    bh = 0.55  # box height
    gap = 0.30
    # vertical positions top->bottom
    y_top = 4.7
    y1a = y_top
    y1b = y1a - bh - gap
    y1c = y1b - bh - gap
    y1d = y1c - bh - gap

    ax.text(x1 + w / 2, y_top + bh + 0.30, "Step 1: junction envelope",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_STEP1)

    _box(ax, x1, y1a, w, bh, "Pick doping mult × bias",          C_STEP1)
    _box(ax, x1, y1b, w, bh, "Charge sim  (Tidy3D)",             C_STEP1)
    _box(ax, x1, y1c, w, bh, "Optical mode solve  (PhotonForge)", C_STEP1)
    _box(ax, x1, y1d, w, bh, "Journal (VπL, C, R)",              C_OUT)

    # arrows down
    for ya, yb in [(y1a, y1b), (y1b, y1c), (y1c, y1d)]:
        _arrow(ax, x1 + w / 2, ya, x1 + w / 2, yb + bh, color=C_STEP1)

    # loopback (bracket-and-fill)
    _arrow(ax, x1, y1d + bh / 2, x1 - 0.30, y1d + bh / 2,
           color=C_STEP1, style="-", lw=1.2)
    _arrow(ax, x1 - 0.30, y1d + bh / 2, x1 - 0.30, y1a + bh / 2,
           color=C_STEP1, style="-", lw=1.2)
    _arrow(ax, x1 - 0.30, y1a + bh / 2, x1, y1a + bh / 2,
           color=C_STEP1, lw=1.2)
    ax.text(x1 - 0.45, (y1a + y1d) / 2 + bh / 2, "bracket-and-fill\n×10",
            ha="right", va="center", fontsize=9, color=C_STEP1, style="italic")

    # --- HANDOFF arrow + box ---
    x_handoff = x1 + w + 0.45
    _box(ax, x_handoff, y1b + 0.15, 1.85, bh, "Pick 10 C-targets\n(min VπL ±10 %)",
         C_GATE, fc="white", text_color=C_TEXT, fontsize=9.5)
    _arrow(ax, x1 + w + 0.05, y1d + bh / 2, x_handoff, y1b + 0.15 + bh / 2,
           color=C_GATE, lw=1.5, connectionstyle="arc3,rad=-0.2")

    # --- STEP 2 column (right) ---
    x2 = x_handoff + 2.1
    y_top2 = y_top
    y2a = y_top2
    y2b = y2a - bh - gap
    y2c = y2b - bh - gap
    y2d = y2c - bh - gap
    y2e = y2d - bh - gap

    ax.text(x2 + w / 2, y_top2 + bh + 0.30,
            "Step 2: electrode design (per C-target)",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_STEP2)

    _box(ax, x2, y2a, w, bh, "Propose CPS geometry\n(LHS → Bayesian opt)",
         C_STEP2, fontsize=9.5)
    _box(ax, x2, y2b, w, bh, "DRC + sanity checks",                C_GATE, fc="white",
         text_color=C_TEXT)
    _box(ax, x2, y2c, w, bh, "RF FDTD  (Tidy3D, GPU cloud)",       C_STEP2)
    _box(ax, x2, y2d, w, bh, "ABCD + junction loading\n(analytic)", C_STEP2,
         fontsize=9.5)
    _box(ax, x2, y2e, w, bh, "Journal (Z₀, n_eff_RF, BW)",         C_OUT)

    for ya, yb in [(y2a, y2b), (y2b, y2c), (y2c, y2d), (y2d, y2e)]:
        _arrow(ax, x2 + w / 2, ya, x2 + w / 2, yb + bh, color=C_STEP2)

    # loopback
    _arrow(ax, x2 + w, y2e + bh / 2, x2 + w + 0.30, y2e + bh / 2,
           color=C_STEP2, style="-", lw=1.2)
    _arrow(ax, x2 + w + 0.30, y2e + bh / 2, x2 + w + 0.30, y2a + bh / 2,
           color=C_STEP2, style="-", lw=1.2)
    _arrow(ax, x2 + w + 0.30, y2a + bh / 2, x2 + w, y2a + bh / 2,
           color=C_STEP2, lw=1.2)
    ax.text(x2 + w + 0.45, (y2a + y2e) / 2 + bh / 2, "20 ×\nper target",
            ha="left", va="center", fontsize=9, color=C_STEP2, style="italic")

    # DRC failed -> back to propose (red dashed)
    _arrow(ax, x2, y2b + bh / 2, x2 - 0.30, y2b + bh / 2,
           color="#aa3333", style="-", lw=1.0,
           connectionstyle="arc3,rad=0.0")
    _arrow(ax, x2 - 0.30, y2b + bh / 2, x2 - 0.30, y2a + bh / 2,
           color="#aa3333", style="-", lw=1.0)
    _arrow(ax, x2 - 0.30, y2a + bh / 2, x2, y2a + bh / 2,
           color="#aa3333", lw=1.0)
    ax.text(x2 - 0.45, (y2a + y2b) / 2 + bh / 2, "reject\n(no FDTD billed)",
            ha="right", va="center", fontsize=8, color="#aa3333", style="italic")

    ax.set_xlim(-1.2, x2 + w + 1.9)
    ax.set_ylim(y2e - 0.4, y_top + bh + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
