"""Animate the Step-1 PN-junction doping cross-section vs `mult`.

Builds a GIF showing how p- and n-doping concentrations grow as the
agent's scalar doping multiplier sweeps from 0.2 to 20.  Heavy access
contacts (p+, p++, n+, n++) are fixed by the process and stay constant;
only the core/slab dopings move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

# Geometry constants (microns), mirroring pn_junction.py
W_CORE = 0.5
H_CORE = 0.22
H_CLEARANCE = 0.09
W_SIDE = 1.0
W_CLEARANCE = 2.0
W_TOT = 2 * W_SIDE + 2 * W_CLEARANCE + W_CORE

Y_P_P  = -W_CORE / 2 - 0.2     # -0.45
Y_N_P  = +W_CORE / 2 + 0.2     # +0.45
Y_P_PP = -W_CORE / 2 - 0.9     # -1.15
Y_N_PP = +W_CORE / 2 + 0.9     # +1.15

P_P_DOPING  = 1.5e19
N_P_DOPING  = 1.2e19
P_PP_DOPING = 1.0e20
N_PP_DOPING = 1.0e20

P_NOMINAL = 5e17
N_NOMINAL = 3e17

# Fixed colour limits in log space so colour stays comparable across frames
LOG_MIN, LOG_MAX = 16.0, 20.5  # log10(N_doping cm^-3)


def _color(conc: float, kind: str) -> tuple:
    """Map a doping concentration to an RGBA colour.

    `kind` is 'p' or 'n' to pick the colour family.
    """
    if conc <= 0:
        return (1, 1, 1, 1)
    val = (np.log10(conc) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    val = float(np.clip(val, 0.0, 1.0))
    if kind == "p":
        cmap = plt.get_cmap("Reds")
    else:
        cmap = plt.get_cmap("Blues")
    return cmap(0.20 + 0.7 * val)


def _draw_frame(ax_main, ax_bar, mult: float):
    ax_main.clear()
    p_core = P_NOMINAL * mult
    n_core = N_NOMINAL * mult

    # --- Background: SiO2 BOX below z=0, air above z=H_CORE (drawn as a
    # thin grey strip just to give context)
    ax_main.add_patch(mpatches.Rectangle(
        (-W_TOT / 2, -0.30), W_TOT, 0.30,
        facecolor="#dde", edgecolor="black", linewidth=0.4))
    ax_main.text(0, -0.18, "SiO$_2$ BOX", ha="center", va="center",
                 fontsize=9, color="#555")

    # --- Si layer regions (z=0 to H_CORE for the core, z=0 to H_CLEARANCE
    # for the slab/clearance on the sides) ---
    # We draw left to right.
    def _patch(y_lo, y_hi, z_lo, z_hi, conc, kind, label=None):
        h = z_hi - z_lo
        ax_main.add_patch(mpatches.Rectangle(
            (y_lo, z_lo), y_hi - y_lo, h,
            facecolor=_color(conc, kind), edgecolor="black", linewidth=0.4))
        if label:
            ax_main.text((y_lo + y_hi) / 2, z_hi + 0.03, label,
                         ha="center", va="bottom", fontsize=8, color="black")

    # p++ contact on far left
    _patch(-W_TOT / 2, Y_P_PP, 0.0, H_CLEARANCE, P_PP_DOPING, "p", "p⁺⁺")
    # p+ region (slab)
    _patch(Y_P_PP, Y_P_P, 0.0, H_CLEARANCE, P_P_DOPING, "p", "p⁺")
    # p slab + core p (varies)
    _patch(Y_P_P, -W_CORE / 2, 0.0, H_CLEARANCE, p_core, "p", "p")
    _patch(-W_CORE / 2, 0.0, 0.0, H_CORE, p_core, "p")
    # n slab + core n (varies)
    _patch(0.0, W_CORE / 2, 0.0, H_CORE, n_core, "n")
    _patch(W_CORE / 2, Y_N_P, 0.0, H_CLEARANCE, n_core, "n", "n")
    # n+
    _patch(Y_N_P, Y_N_PP, 0.0, H_CLEARANCE, N_P_DOPING, "n", "n⁺")
    # n++ contact on far right
    _patch(Y_N_PP, W_TOT / 2, 0.0, H_CLEARANCE, N_PP_DOPING, "n", "n⁺⁺")

    # Core outline emphasised
    ax_main.add_patch(mpatches.Rectangle(
        (-W_CORE / 2, 0.0), W_CORE, H_CORE,
        facecolor="none", edgecolor="black", linewidth=1.5))
    ax_main.text(0, H_CORE + 0.10,
                 f"core p={p_core:.2e}, n={n_core:.2e} cm$^{{-3}}$",
                 ha="center", va="bottom", fontsize=9)

    # PN junction line at y=0
    ax_main.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_main.set_xlim(-W_TOT / 2 - 0.1, W_TOT / 2 + 0.1)
    ax_main.set_ylim(-0.4, 0.5)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("y (lateral) [μm]")
    ax_main.set_ylabel("z (height) [μm]")
    ax_main.set_title(
        f"Step-1 PN junction cross-section — mult = {mult:.2f}\n"
        f"(p_core = {P_NOMINAL:.1e}·mult,   n_core = {N_NOMINAL:.1e}·mult)",
        fontsize=11
    )


def animate(outpath: Path, *, fps: float = 1.4):
    rows = [json.loads(l) for l in open("journal.jsonl", encoding="utf-8")]
    mults = sorted({r["mult"] for r in rows if not r.get("meta")})
    if not mults:
        raise SystemExit("No mults in journal.jsonl")

    fig = plt.figure(figsize=(9, 4.8))
    ax_main = fig.add_subplot(1, 1, 1)

    # Side colour-bar legend (two small log strips)
    cax_p = fig.add_axes([0.86, 0.18, 0.025, 0.62])
    cax_n = fig.add_axes([0.92, 0.18, 0.025, 0.62])
    z = np.linspace(0, 1, 256)[:, None]
    cax_p.imshow(z, aspect="auto", origin="lower",
                 cmap=plt.get_cmap("Reds"),
                 extent=(0, 1, LOG_MIN, LOG_MAX))
    cax_n.imshow(z, aspect="auto", origin="lower",
                 cmap=plt.get_cmap("Blues"),
                 extent=(0, 1, LOG_MIN, LOG_MAX))
    for cax, label in [(cax_p, "log₁₀ N$_A$"), (cax_n, "log₁₀ N$_D$")]:
        cax.set_xticks([])
        cax.set_yticks([16, 17, 18, 19, 20])
        cax.tick_params(labelsize=8)
        cax.set_title(label, fontsize=8, pad=4)
    fig.subplots_adjust(left=0.08, right=0.83, top=0.86, bottom=0.13)

    def update(idx: int):
        _draw_frame(ax_main, None, mults[idx])
        return ()

    anim = FuncAnimation(fig, update, frames=len(mults), blit=False)
    outpath.parent.mkdir(exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(outpath, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Wrote {outpath} ({len(mults)} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("field_plots/doping_sweep.gif"))
    p.add_argument("--fps", type=float, default=1.4)
    args = p.parse_args()
    animate(args.out, fps=args.fps)


if __name__ == "__main__":
    main()
