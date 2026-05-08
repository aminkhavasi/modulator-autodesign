"""Scalar objective for CPS optimization.

The objective is a weighted sum of squared relative errors:

    J = ((Re Z0_loaded(f0) - 50) / 50) ** 2
      + ((n_eff_rf_loaded(f0) - n_group_opt) / n_group_opt) ** 2

evaluated AFTER junction loading (per user specification).
"""

from __future__ import annotations

from .junction import Junction, loaded_at_f0
from .simulate import CPSResult


TARGET_Z0 = 50.0  # ohms


def objective(cps: CPSResult, junction: Junction, *,
              w_Z0: float = 1.0, w_neff: float = 1.0) -> dict:
    """Return scalar objective + the components that built it.

    The returned dict has keys:
      objective : scalar to minimize
      Z0_re_f0  : real impedance at f0 after loading
      n_eff_f0  : RF effective index at f0 after loading
      Z0_term   : ((Z0-50)/50)**2 (un-weighted)
      n_term    : ((n_eff-n_group)/n_group)**2 (un-weighted)
    """
    Z0_re, n_eff, _alpha = loaded_at_f0(cps, junction)
    Z0_term = ((Z0_re - TARGET_Z0) / TARGET_Z0) ** 2
    n_term = ((n_eff - junction.n_group_opt) / junction.n_group_opt) ** 2
    return {
        "objective": w_Z0 * Z0_term + w_neff * n_term,
        "Z0_re_f0": Z0_re,
        "n_eff_f0": n_eff,
        "Z0_term": Z0_term,
        "n_term": n_term,
    }
