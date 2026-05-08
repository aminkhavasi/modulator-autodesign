"""MZM length calculation for a target extinction ratio at quadrature bias.

Cleanup of the original snippet:
  * `np.asin` -> `np.arcsin` (asin doesn't exist in numpy)
  * Push-pull factor made explicit (default True for MZM modulator)
  * Returns micrometers consistently

For an MZM at quadrature, output power is
    P_out / P_in = 0.5 * (1 + sin(phi))
so extinction ratio (max/min) is r = (1 + sin(phi)) / (1 - sin(phi)),
giving sin(phi) = (r - 1) / (r + 1).

In push-pull, both arms are driven oppositely -- each by V_pp/2 -- so the
*differential* phase shift per arm is doubled.  Equivalent V_pi reduction
by a factor of 2 versus single-arm drive.
"""

from __future__ import annotations

import numpy as np


def mzm_length_um(target_ER_dB: float,
                  VpiL_per_arm_V_cm: float,
                  V_pp: float, *,
                  push_pull: bool = True) -> float:
    """Length (micrometers) needed for `target_ER_dB` extinction at quadrature.

    target_ER_dB         desired Pmax/Pmin in dB
    VpiL_per_arm_V_cm    single-arm modulation efficiency (signed; we use abs)
    V_pp                 peak-to-peak drive voltage
    push_pull            True (default) for differential drive across two arms
    """
    er_lin = 10.0 ** (target_ER_dB / 10.0)
    sin_phi = (er_lin - 1.0) / (er_lin + 1.0)
    phi_rad = float(np.arcsin(sin_phi))
    if phi_rad == 0.0:
        return 0.0

    v_peak = V_pp / 2.0
    if push_pull:
        # Each arm produces phi/2 of the differential; required per-arm V_pi:
        v_pi_per_arm_required = 2.0 * (np.pi * v_peak) / phi_rad
    else:
        v_pi_per_arm_required = (np.pi * v_peak) / phi_rad

    length_cm = abs(VpiL_per_arm_V_cm) / v_pi_per_arm_required
    return float(length_cm * 1e4)


if __name__ == "__main__":
    # Example: typical Step-1 numbers
    L_um = mzm_length_um(target_ER_dB=5.0,
                         VpiL_per_arm_V_cm=1.5,  # V.cm
                         V_pp=2.0)
    print(f"MZM length for 5 dB ER, VpiL=1.5 V.cm, V_pp=2 V, push-pull: "
          f"{L_um:.0f} um = {L_um/1000:.2f} mm")
