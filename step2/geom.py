"""CPS geometry construction.

Pure geometry -- no Tidy3D simulations are run here.  Builds a structure list
that simulate.py wraps into a Simulation/TerminalComponentModeler.

Direct port of the notebook's create_cps + create_T_structure +
create_segmented_cps logic, reorganized into a single class with the 8 free
parameters as inputs.

Also produces a stable hash of the geometry, used by simulate.py to key the
on-disk cache so re-runs don't re-bill Tidy3D.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np
import tidy3d as td
import tidy3d.rf as rf

from .fab_rules import DEFAULT_RULES, FabRules, feasible


# --- Fixed (process-determined, not optimized) -----------------------------

LEN_INF = 1e5         # effective infinity (microns)
TM = 2.0              # metal thickness  (process)
TSIO20 = 3.59         # SiO2 BOX thickness (process)
TSI = LEN_INF         # substrate (effective infinity for simulation)

W1 = 1000.0           # overall width of dielectric layers (microns)

# Target length of the segmented section (microns). Constant so every FDTD
# has comparable cost; the actual length is rounded to a whole number of
# unit cells.  Notebook default was 1000 um.
L_SEGMENTED_TARGET = 1000.0
# Lower bound on the number of unit cells: too few and the de-embedded
# segmented-line characteristics still feel feedline-launch transients.
NUM_UNITS_MIN = 8
# Input/output unloaded CPS lengths, expressed as multiples of period P
NUM_UNITS_FEEDLINE = 10
# Floor for feedline length so wave ports keep ≥2 mesh cells of clearance
# from the simulation boundary. NUM_UNITS_FEEDLINE * period drops below
# this when r and c are both small; without the floor, Tidy3D rejects the
# batch at upload-time validation. Smallest LHS draw that passed was
# ~229 µm — 300 µm gives a margin without ballooning the sim domain.
MIN_L_FEEDLINE_UM = 300.0

# Frequency band (Hz) -- fixed per notebook
F_MIN = 10e9
F_MAX = 40e9
N_FREQS = 51
F0 = (F_MIN + F_MAX) / 2.0


# --- Data class -------------------------------------------------------------

@dataclass(frozen=True)
class CPSGeometry:
    """8 free parameters of the segmented CPS T-rail geometry, in microns.

    g  : inner gap between rails (before T extensions)
    ws : signal trace width
    wg : ground trace width
    s  : T-bar (top-of-T) width
    r  : T-bar length
    h  : T-neck length
    t  : T-neck width
    c  : inter-T gap (period P = r + c)
    """
    g: float
    ws: float
    wg: float
    s: float
    r: float
    h: float
    t: float
    c: float

    @property
    def period(self) -> float:
        """Period of the T-rail array (microns)."""
        return self.r + self.c

    @property
    def num_units(self) -> int:
        """Number of T unit cells, derived to keep segmented length ≈ L_SEGMENTED_TARGET."""
        return max(NUM_UNITS_MIN, int(round(L_SEGMENTED_TARGET / self.period)))

    @property
    def L_segmented(self) -> float:
        """Length of the segmented (loaded) section (microns), an integer
        multiple of the period closest to L_SEGMENTED_TARGET."""
        return self.num_units * self.period

    @property
    def L_feedline(self) -> float:
        """Length of one input/output unloaded section (microns)."""
        return max(NUM_UNITS_FEEDLINE * self.period, MIN_L_FEEDLINE_UM)

    @property
    def w_cps(self) -> float:
        """Overall lateral span of the CPS (microns)."""
        return self.ws + self.g + self.wg

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Stable 12-char hex hash for cache filenames."""
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def label(self) -> str:
        """Human-readable label for logs."""
        return (f"g{self.g:.1f}_ws{self.ws:.1f}_wg{self.wg:.1f}_"
                f"s{self.s:.2f}_r{self.r:.1f}_h{self.h:.2f}_"
                f"t{self.t:.2f}_c{self.c:.2f}")


# --- Construction helpers ---------------------------------------------------

def _make_cps_pair(*, gap, w_left, w_right, thickness, y_start, y_end,
                   medium):
    """Two parallel rails along y, centered on x=0 with the given inner gap.

    `w_left` is the rail to the -x side, `w_right` to the +x side.  This split
    is needed because the user wants ws (right) and wg (left) independent.
    """
    length = y_end - y_start
    midpos = (y_end + y_start) / 2.0
    str_left = td.Structure(
        medium=medium,
        geometry=td.Box(
            size=(w_left, length, thickness),
            center=(-gap / 2 - w_left / 2, midpos, thickness / 2.0),
        ),
    )
    str_right = td.Structure(
        medium=medium,
        geometry=td.Box(
            size=(w_right, length, thickness),
            center=(gap / 2 + w_right / 2, midpos, thickness / 2.0),
        ),
    )
    return [str_left, str_right]


def _make_T(*, base_x, base_y, direction, r, t, s, h, thickness):
    """One T-shaped extension: top-bar (s wide, r long) + neck (h long, t wide).

    `direction` is "+" or "-": which way along x the T extends from base_x.
    """
    sgn = 1.0 if direction == "+" else -1.0
    return [
        td.Box(
            size=(s, r, thickness),
            center=(base_x + sgn * (h + s / 2), base_y, thickness / 2.0),
        ),
        td.Box(
            size=(h, t, thickness),
            center=(base_x + sgn * h / 2, base_y, thickness / 2.0),
        ),
    ]


def build_structures(geom: CPSGeometry, *,
                     med_air, med_SiO2, med_Si, med_Al) -> list[td.Structure]:
    """Build the full Tidy3D structure list for one segmented CPS geometry.

    Returns a list of td.Structure ready to feed into td.Simulation.
    """
    # Dielectric stack
    str_sio2_box = td.Structure(
        medium=med_SiO2,
        geometry=td.Box(center=(0, 0, -TSIO20 / 2.0),
                        size=(W1, td.inf, TSIO20)),
    )
    str_si = td.Structure(
        medium=med_Si,
        geometry=td.Box(center=(0, 0, -TSI / 2.0 - TSIO20),
                        size=(W1, td.inf, TSI)),
    )
    structures_layers = [str_si, str_sio2_box]

    L_seg = geom.L_segmented

    # Wide CPS through the segmented region (the main rails, not the T's).
    # gap = `g` (inner clearance, before T-extensions reach inward).
    cps_wide = _make_cps_pair(
        gap=geom.g, w_left=geom.wg, w_right=geom.ws,
        thickness=TM, y_start=-LEN_INF, y_end=LEN_INF, medium=med_Al,
    )

    # Narrow CPS for input/output feedlines.  Inner gap is reduced by
    # 2*(s+h) so the rails approach where the T-tips would have been.  The
    # outer rail width grows by (s+h) to compensate (matches notebook).
    narrow_gap = geom.g - 2.0 * (geom.s + geom.h)
    cps_narrow_in = _make_cps_pair(
        gap=narrow_gap,
        w_left=geom.wg + geom.s + geom.h,
        w_right=geom.ws + geom.s + geom.h,
        thickness=TM, y_start=-LEN_INF, y_end=-L_seg / 2.0, medium=med_Al,
    )
    cps_narrow_out = _make_cps_pair(
        gap=narrow_gap,
        w_left=geom.wg + geom.s + geom.h,
        w_right=geom.ws + geom.s + geom.h,
        thickness=TM, y_start=L_seg / 2.0, y_end=LEN_INF, medium=med_Al,
    )

    # T-extensions
    P = geom.period
    N = geom.num_units
    T_geoms = []
    for ii in range(N + 1):
        yy = -P * N / 2.0 + ii * P
        # Right rail: T extends in -x direction (toward gap)
        T_geoms += _make_T(base_x=geom.g / 2.0, base_y=yy, direction="-",
                           r=geom.r, t=geom.t, s=geom.s, h=geom.h, thickness=TM)
        # Left rail: T extends in +x direction (toward gap)
        T_geoms += _make_T(base_x=-geom.g / 2.0, base_y=yy, direction="+",
                           r=geom.r, t=geom.t, s=geom.s, h=geom.h, thickness=TM)

    T_struct = td.Structure(
        medium=med_Al,
        geometry=td.GeometryGroup(geometries=T_geoms),
    )

    return (structures_layers
            + cps_wide
            + cps_narrow_in
            + cps_narrow_out
            + [T_struct])


def sim_box_size(geom: CPSGeometry) -> tuple[tuple[float, float, float],
                                             tuple[float, float, float]]:
    """Return ((cx, cy, cz), (Lx, Ly, Lz)) of the simulation domain.

    Includes a quarter-wavelength padding from PML on the lateral sides.
    """
    padding = (td.C_0 / F_MAX) / 4.0  # microns (C_0 is in micron units)
    Lx = geom.w_cps + 2 * padding
    Ly = geom.L_segmented + 2 * geom.L_feedline
    Lz = Lx
    cy = -(geom.L_feedline - geom.L_feedline) / 2.0  # = 0 (symmetric)
    return (0.0, cy, 0.0), (Lx, Ly, Lz)


def wave_port_centers(geom: CPSGeometry, wp_offset: float = 100.0
                      ) -> tuple[tuple[float, float, float],
                                 tuple[float, float, float]]:
    """Centers of WP1 (input) and WP2 (output)."""
    L_seg_half = geom.L_segmented / 2.0
    return ((0.0, -L_seg_half - wp_offset, 1.0),
            (0.0, +L_seg_half + wp_offset, 0.0))
