"""Relaxed fab-rule set for exploring the high-C corner.

Lithography-bound minimum feature sizes (s, r, h, c) are dropped to 1.0 μm
to test whether denser, finer T-bar segments can recover bandwidth at the
heaviest junction loading.  All other bounds (CPS gap g, signal/ground
trace widths ws/wg, maxima) are unchanged: those are set by impedance and
optical-mode placement, not by litho.

This is an *experiment* rules-set.  Sub-2 μm thin-metal features need a
revisit of the conductor model (skin depth, surface roughness, grain
boundaries) before any silicon-level commitment.
"""

from __future__ import annotations

from .fab_rules import FabRules

RELAXED_RULES = FabRules(
    g_min=20.0,   g_max=200.0,
    ws_min=20.0,  ws_max=250.0,
    wg_min=20.0,  wg_max=250.0,
    s_min=1.0,    s_max=30.0,    # was 2.0
    r_min=1.0,    r_max=80.0,    # was 10.0
    h_min=1.0,    h_max=20.0,    # was 2.0
    t_min=1.0,    t_max=5.0,
    c_min=1.0,    c_max=10.0,    # was 1.5
    safety_gap_min=1.0,
)
