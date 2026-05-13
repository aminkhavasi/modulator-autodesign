"""Step 3 BW-objective rule set.

User authorized relaxing CPS rail/gap minimums to 5 μm.  T-feature
minimums (set by process litho, not by CPS impedance design) are
left at near-strict values to keep the search space realistic.

Compared to the original FabRules:
  g_min:  20.0 -> 5.0   (CPS inner gap)
  ws_min: 20.0 -> 5.0   (signal-trace width)
  wg_min: 20.0 -> 5.0   (ground-trace width)
  r_min:  10.0 -> 5.0   (T-bar length)
  others unchanged.

This opens the high-Z0 corner that was previously gated by g/ws/wg_min=20,
without rebadging the litho-bound small T-features.
"""

from __future__ import annotations

from .fab_rules import FabRules

RULES_V2 = FabRules(
    g_min=5.0,    g_max=200.0,
    ws_min=5.0,   ws_max=250.0,
    wg_min=5.0,   wg_max=250.0,
    s_min=2.0,    s_max=30.0,
    r_min=5.0,    r_max=80.0,
    h_min=2.0,    h_max=20.0,
    t_min=1.0,    t_max=5.0,
    c_min=1.5,    c_max=10.0,
    safety_gap_min=1.0,
)
