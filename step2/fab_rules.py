"""Fab-rule defaults for the segmented CPS T-rail geometry.

These are placeholder values pulled from generic silicon-photonics + thin-metal
process expectations.  **OVERRIDE THESE WITH YOUR FAB'S ACTUAL PDK RULES.**

All dimensions are in micrometers.

Used by:
  geom.py     -- to validate / clip geometry candidates before submission
  propose.py  -- as the search-space bounds for LHS and BO
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FabRules:
    """Minimum and maximum feature sizes (micrometers)."""
    # Inner CPS gap (between signal and ground rails, before T extensions)
    g_min: float = 20.0
    g_max: float = 200.0

    # Signal trace width
    ws_min: float = 20.0
    ws_max: float = 250.0

    # Ground trace width (independent of ws per user choice)
    wg_min: float = 20.0
    wg_max: float = 250.0

    # T-bar (top of the T) width (called `s` in the notebook)
    s_min: float = 2.0
    s_max: float = 30.0

    # T-bar length (`r`)
    r_min: float = 10.0
    r_max: float = 80.0

    # T-neck length (`h`)
    h_min: float = 2.0
    h_max: float = 20.0

    # T-neck width (`t`)
    t_min: float = 1.0
    t_max: float = 5.0

    # Inter-T gap (`c`); period P = r + c
    c_min: float = 1.5
    c_max: float = 10.0

    # Geometric constraint: 2*(s + h) + safety_gap <= g
    # (The two T-rails extending into the gap from each side must leave at
    # least `safety_gap` of bare gap in the middle for the optical mode.)
    safety_gap_min: float = 1.0


DEFAULT_RULES = FabRules()


def feasible(geom: dict, rules: FabRules = DEFAULT_RULES) -> tuple[bool, list[str]]:
    """Check whether a candidate geometry satisfies all fab rules.

    Returns (is_feasible, list_of_violations).  Violations are human-readable
    strings; an empty list means is_feasible is True.
    """
    v = []
    if not (rules.g_min <= geom["g"] <= rules.g_max):
        v.append(f"g={geom['g']:.2f} outside [{rules.g_min}, {rules.g_max}]")
    if not (rules.ws_min <= geom["ws"] <= rules.ws_max):
        v.append(f"ws={geom['ws']:.2f} outside [{rules.ws_min}, {rules.ws_max}]")
    if not (rules.wg_min <= geom["wg"] <= rules.wg_max):
        v.append(f"wg={geom['wg']:.2f} outside [{rules.wg_min}, {rules.wg_max}]")
    if not (rules.s_min <= geom["s"] <= rules.s_max):
        v.append(f"s={geom['s']:.2f} outside [{rules.s_min}, {rules.s_max}]")
    if not (rules.r_min <= geom["r"] <= rules.r_max):
        v.append(f"r={geom['r']:.2f} outside [{rules.r_min}, {rules.r_max}]")
    if not (rules.h_min <= geom["h"] <= rules.h_max):
        v.append(f"h={geom['h']:.2f} outside [{rules.h_min}, {rules.h_max}]")
    if not (rules.t_min <= geom["t"] <= rules.t_max):
        v.append(f"t={geom['t']:.2f} outside [{rules.t_min}, {rules.t_max}]")
    if not (rules.c_min <= geom["c"] <= rules.c_max):
        v.append(f"c={geom['c']:.2f} outside [{rules.c_min}, {rules.c_max}]")

    # T-rails must fit inside the gap with safety margin.
    inner_clearance = geom["g"] - 2.0 * (geom["s"] + geom["h"])
    if inner_clearance < rules.safety_gap_min:
        v.append(
            f"2*(s+h)={2*(geom['s']+geom['h']):.2f} >= g - safety = "
            f"{geom['g'] - rules.safety_gap_min:.2f} "
            f"(inner clearance {inner_clearance:.2f} < {rules.safety_gap_min})"
        )

    return (len(v) == 0), v


def clip_to_bounds(geom: dict, rules: FabRules = DEFAULT_RULES) -> dict:
    """Clamp each parameter into its bounds.  Does NOT enforce the
    g >= 2*(s+h) + safety_gap constraint -- caller must check `feasible`
    after clipping.
    """
    out = dict(geom)
    for k, lo, hi in [
        ("g", rules.g_min, rules.g_max),
        ("ws", rules.ws_min, rules.ws_max),
        ("wg", rules.wg_min, rules.wg_max),
        ("s", rules.s_min, rules.s_max),
        ("r", rules.r_min, rules.r_max),
        ("h", rules.h_min, rules.h_max),
        ("t", rules.t_min, rules.t_max),
        ("c", rules.c_min, rules.c_max),
    ]:
        out[k] = float(max(lo, min(hi, out[k])))
    return out


def repair_geometry_constraint(geom: dict,
                               rules: FabRules = DEFAULT_RULES) -> dict:
    """Repair `g >= 2*(s+h) + safety_gap` by inflating g if needed.

    Used by `propose.py` when LHS/BO suggests an infeasible point: we'd rather
    bump g up to fit the proposed s,h,c than reject the candidate outright
    (which loses the BO suggestion's information).  Returns a new dict.
    """
    out = dict(geom)
    needed = 2 * (out["s"] + out["h"]) + rules.safety_gap_min
    if out["g"] < needed:
        out["g"] = min(needed, rules.g_max)
    return out
