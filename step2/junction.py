"""Junction loading -> loaded line -> EO S21 -> bandwidth -> MZM length.

This module is the post-processing pipeline that takes a CPSResult and a
junction (R, C) plus a target ER and returns the EO bandwidth.

All of this is closed-form; no Tidy3D simulation runs here.  So sweeping
across many (R, C) points for a fixed CPSResult is essentially free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tidy3d as td

from .simulate import CPSResult


# --- Constants & defaults --------------------------------------------------

C_M_S = 299_792_458.0  # m/s

# EO model defaults (from your answers in chat)
DEFAULT_L_PARASITIC = 0.0   # set to 0 for clean comparison
DEFAULT_ZS = 50.0
DEFAULT_ZL = 50.0


@dataclass(frozen=True)
class Junction:
    """Single-junction (per-arm) characteristics from Step 1.

    Convention matches notebook Cell 2: ``c_pn`` and ``r_pn`` are *single-arm*
    values; SPP MZM topology is applied internally as
    ``c_eff = c_pn / 2`` and ``r_eff = r_pn * 2``.
    """
    C_pF_per_cm: float           # capacitance, single arm
    R_ohm_cm: float              # series resistance, single arm
    VpiL_V_cm: float             # modulation-efficiency product
    n_group_opt: float           # optical group index (target for n_eff_rf)

    # Conversion to SI per-unit-length values
    @property
    def c_pn_F_per_m(self) -> float:
        """SI per-meter capacitance (single arm), F/m."""
        return self.C_pF_per_cm * 1e-12 * 100.0

    @property
    def r_pn_ohm_m(self) -> float:
        """SI per-meter resistance (single arm), Ohm*m."""
        return self.R_ohm_cm * 1e-2

    def spp_load_per_m(self) -> tuple[float, float]:
        """Series push-pull effective per-unit-length (R, C).

        Two single-arm diodes in series across the rails: capacitance halves,
        resistance doubles.
        """
        return (2.0 * self.r_pn_ohm_m, 0.5 * self.c_pn_F_per_m)


# --- Loaded-line characteristics ------------------------------------------

def apply_junction_loading(cps: CPSResult, junction: Junction
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add y_junction(omega) shunt to the unloaded line and re-extract.

    Returns (Z0_loaded, n_eff_rf_loaded, alpha_dB_cm_loaded), each shape (N_freqs,).
    """
    omega = 2.0 * np.pi * cps.freqs
    r_eff, c_eff = junction.spp_load_per_m()

    # y_junction = 1 / (R + 1/(j omega C))
    # Engineering convention here matches notebook Cell 69
    # (notebook uses -1j*omega*c_pn explicitly).
    y_junction = 1.0 / (r_eff + 1.0 / (-1j * omega * c_eff))
    y_shunt_loaded = cps.y_shunt + y_junction
    z_series = cps.z_series

    # Inverse of _eq_circuit:
    #   gamma = sqrt(z * y);  Z0 = sqrt(z / y)
    gamma = np.sqrt(z_series * y_shunt_loaded)
    Z0_loaded = np.sqrt(z_series / y_shunt_loaded)
    alpha_np_m = np.real(gamma)
    beta = -np.imag(gamma)

    alpha_dB_cm_loaded = alpha_np_m * (20.0 * np.log10(np.e)) / 100.0
    n_eff_rf_loaded = beta * C_M_S / omega
    return Z0_loaded, n_eff_rf_loaded, alpha_dB_cm_loaded


def loaded_at_f0(cps: CPSResult, junction: Junction
                 ) -> tuple[float, float, float]:
    """(Re Z0, n_eff_rf, alpha_dB_cm) at f0 (band center) after loading."""
    Z0_l, n_l, a_l = apply_junction_loading(cps, junction)
    f0_idx = len(cps.freqs) // 2
    return float(np.real(Z0_l[f0_idx])), float(n_l[f0_idx]), float(a_l[f0_idx])


# --- EO S21 (port of notebook Cell 71) ------------------------------------

def eo_s21(rf_loss_dB_um, freqs, n_rf, n_group, length_um, ZL, ZS, Z0):
    """EO S21 (complex) of a traveling-wave modulator vs frequency.

    Inputs:
      rf_loss_dB_um : (Nf,) RF loss in dB/um
      freqs         : (Nf,) Hz
      n_rf          : (Nf,) microwave effective index (loaded)
      n_group       : scalar, optical group index
      length_um     : scalar, modulator length in um
      ZL, ZS        : scalars (ohms)
      Z0            : (Nf,) complex characteristic impedance (loaded)
    """
    c_ums = td.C_0  # microns/sec
    omega = 2.0 * np.pi * freqs
    alpha_np_um = rf_loss_dB_um / 8.686
    gamma_m = alpha_np_um + 1j * (omega / c_ums) * n_rf

    F_plus = ((1 - np.exp(gamma_m * length_um
                          - 1j * (omega / c_ums) * n_group * length_um))
              / (gamma_m * length_um
                 - 1j * (omega / c_ums) * n_group * length_um))
    F_minus = ((1 - np.exp(-gamma_m * length_um
                           - 1j * (omega / c_ums) * n_group * length_um))
               / (-gamma_m * length_um
                  - 1j * (omega / c_ums) * n_group * length_um))

    tanh_gL = np.tanh(gamma_m * length_um)
    Z_in = Z0 * (ZL + Z0 * tanh_gL) / (Z0 + ZL * tanh_gL)

    return -((2 * Z_in / (Z_in + ZS))
             * ((ZL + Z0) * F_plus + (ZL - Z0) * F_minus)
             / ((ZL + Z0) * np.exp(gamma_m * length_um)
                + (ZL - Z0) * np.exp(-gamma_m * length_um)))


def eo_response(cps: CPSResult, junction: Junction, length_um: float, *,
                L_parasitic_H: float = DEFAULT_L_PARASITIC,
                ZS: float = DEFAULT_ZS, ZL: float = DEFAULT_ZL
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full EO frequency response with the 1 Hz extrapolation.

    Returns (freqs_ext, H_total, H_tw_only).  Length is len(cps.freqs)+1.
    """
    Z0_l, n_l, alpha_l = apply_junction_loading(cps, junction)

    # Extrapolate down to ~DC for normalization
    freqs_ext = np.insert(cps.freqs, 0, 1.0)
    n_l_ext = np.insert(n_l, 0, n_l[0])
    Z0_l_ext = np.insert(Z0_l, 0, Z0_l[0])
    a_l_ext = np.insert(alpha_l, 0, 0.0)
    omega_ext = 2.0 * np.pi * freqs_ext

    # Lumped RLC voltage divider (notebook Cell 75)
    r_eff, c_eff = junction.spp_load_per_m()
    L_um = length_um  # microns
    L_m = L_um * 1e-6
    c_tot = c_eff * L_m
    r_tot = r_eff / L_m
    H_rlc = 1.0 / ((1.0 - omega_ext**2 * L_parasitic_H * c_tot)
                   + 1j * omega_ext * r_tot * c_tot)

    H_tw = eo_s21(
        rf_loss_dB_um=a_l_ext * 1e-4,  # dB/cm -> dB/um
        freqs=freqs_ext, n_rf=n_l_ext, n_group=junction.n_group_opt,
        length_um=L_um, ZL=ZL, ZS=ZS, Z0=Z0_l_ext,
    )

    H_total = H_tw * H_rlc
    return freqs_ext, H_total, H_tw


def bandwidth_3dB_GHz(cps: CPSResult, junction: Junction, length_um: float, **kw
                      ) -> float:
    """3-dB EO bandwidth (GHz) by interpolating |H_total|^2 vs frequency.

    Defined as the lowest f > 0 at which 20*log10(|H_total(f)|/|H_total(f=1Hz)|)
    crosses -3 dB.  Returns NaN if the response stays above -3 dB across the
    whole simulated range, or if it's never above -3 dB (broken response).
    """
    freqs_ext, H_total, _ = eo_response(cps, junction, length_um, **kw)
    H_dB = 20.0 * np.log10(np.abs(H_total) / np.abs(H_total[0]))

    # Look for first crossing of -3 dB
    above = H_dB > -3.0
    if not above[0]:
        return float("nan")  # response is never near 0 dB at low freq -- broken
    crossing_idx = np.argmax(~above)  # first False after a True streak
    if crossing_idx == 0:
        # Never crosses -- or starts already below.  Either way, no 3-dB BW.
        return float("nan")

    # Linear interpolate between freqs_ext[crossing_idx-1] and freqs_ext[crossing_idx]
    f_lo, f_hi = freqs_ext[crossing_idx - 1], freqs_ext[crossing_idx]
    H_lo, H_hi = H_dB[crossing_idx - 1], H_dB[crossing_idx]
    if H_hi == H_lo:
        return float(f_hi / 1e9)
    f_3dB = f_lo + (f_hi - f_lo) * (-3.0 - H_lo) / (H_hi - H_lo)
    return float(f_3dB / 1e9)


# --- MZM length from VpiL + ER spec (push-pull, quadrature bias) -----------

def mzm_length_um(target_ER_dB: float, VpiL_per_arm_V_cm: float, V_pp: float,
                  *, push_pull: bool = True) -> float:
    """Length needed to hit `target_ER_dB` extinction at quadrature bias.

    Push-pull (default) means each arm sees V_pp/2 of opposite sign, so the
    *differential* phase is 2 * (pi V_peak / V_pi_per_arm).  Equivalent to a
    single-drive modulator with V_pi_eff = V_pi_per_arm / 2.

    Returns length in **micrometers** (matches the notebook's `length` units).
    """
    er_lin = 10.0 ** (target_ER_dB / 10.0)
    sin_phi = (er_lin - 1.0) / (er_lin + 1.0)
    phi_rad = float(np.arcsin(sin_phi))   # use np.arcsin (your snippet had np.asin)
    if phi_rad == 0.0:
        return 0.0
    v_peak = V_pp / 2.0
    # Push-pull doubles the phase efficiency: V_pi_eff = V_pi / 2 if both arms drive
    if push_pull:
        # Effective single-drive equivalent: V_pi_required at the pair = (pi V_peak / phi) / 2
        # i.e. each arm need only achieve  V_pi = (pi V_peak / phi) * 2  on its own
        v_pi_per_arm_required = 2.0 * (np.pi * v_peak) / phi_rad
    else:
        v_pi_per_arm_required = (np.pi * v_peak) / phi_rad

    # length [cm] = VpiL [V cm] / V_pi [V]
    length_cm = abs(VpiL_per_arm_V_cm) / v_pi_per_arm_required
    return float(length_cm * 1e4)   # um
