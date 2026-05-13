"""Bandwidth-maximizing objective for the c_target=9 optimization phase.

Where the original `objective.py` minimized

    J = ((Z0_re - 50)/50)**2 + ((n_eff_rf - n_group)/n_group)**2

we found (see analyze_bw.py) that J is poorly aligned with the actual 3-dB
EO bandwidth when the modulator is short.  For the heavy-loading c_target=9
the optimum L_mod is ~280 um, which makes velocity walk-off negligible and
leaves impedance-mismatch reflection as the dominant BW limiter.  BO
optimizing J therefore pushes Z0 DOWN (away from 50 Ohm) and ends up with
WORSE bandwidth than designs J considered bad.

This module provides a drop-in objective that lets BO minimize the negated
3-dB EO bandwidth directly.
"""

from __future__ import annotations

from .junction import Junction, bandwidth_3dB_GHz, mzm_length_um
from .simulate import CPSResult


# Default modulator-length spec (ER, Vpp) matching run_batch.py defaults.
DEFAULT_ER_DB = 5.0
DEFAULT_VPP = 2.0
DEFAULT_L_PARASITIC_H = 0.0


def bw_objective(cps: CPSResult, junction: Junction, *,
                 er_dB: float = DEFAULT_ER_DB,
                 vpp: float = DEFAULT_VPP,
                 L_parasitic_H: float = DEFAULT_L_PARASITIC_H) -> dict:
    """Return -BW (skopt minimizes) plus the components for journaling.

    Bandwidth depends on (Z0(f), n_eff(f), alpha(f)) from the LOADED line
    AND on the modulator length L (which is set by VpiL + the ER target).
    Length is computed from the same recipe `bandwidth_sweep` uses.
    """
    L_um = mzm_length_um(er_dB, junction.VpiL_V_cm, vpp, push_pull=True)
    bw_GHz = bandwidth_3dB_GHz(cps, junction, L_um,
                               L_parasitic_H=L_parasitic_H)
    # If BW is NaN (response broken / never above -3 dB), penalize hard.
    if bw_GHz != bw_GHz:   # NaN check
        return {"objective": 1e3, "BW_GHz": float("nan"),
                "MZM_length_um": L_um}
    return {"objective": -float(bw_GHz),  # skopt minimizes; we want max BW
            "BW_GHz": float(bw_GHz),
            "MZM_length_um": L_um}
