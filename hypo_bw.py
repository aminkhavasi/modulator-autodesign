"""Analytic bandwidth estimate for a HYPOTHETICAL design that hits the
J-optimum at c_target=9.

No FDTD needed: we hand-pick (Z0, n_eff) values, borrow alpha(f) from the
incumbent's cached CPSResult, run them through the same eo_s21 + RLC
voltage divider as bandwidth_3dB_GHz uses, and read off the -3 dB point.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from step2.junction import Junction, eo_s21, mzm_length_um


def compute_bw(Z0_val, n_eff_val, alpha_curve, freqs, J, L_um):
    """Mimic bandwidth_3dB_GHz with synthetic (constant) Z0 & n_eff."""
    freqs_ext = np.insert(freqs, 0, 1.0)
    Z0_ext = np.insert(np.full_like(freqs, Z0_val, dtype=complex), 0, Z0_val)
    n_ext = np.insert(np.full_like(freqs, n_eff_val, dtype=float), 0, n_eff_val)
    a_ext = np.insert(alpha_curve.astype(float), 0, 0.0)

    H_tw = eo_s21(rf_loss_dB_um=a_ext * 1e-4,
                  freqs=freqs_ext, n_rf=n_ext, n_group=J.n_group_opt,
                  length_um=L_um, ZL=50.0, ZS=50.0, Z0=Z0_ext)

    omega = 2.0 * np.pi * freqs_ext
    r_eff, c_eff = J.spp_load_per_m()
    L_m = L_um * 1e-6
    c_tot = c_eff * L_m
    r_tot = r_eff / L_m
    H_rlc = 1.0 / (1.0 + 1j * omega * r_tot * c_tot)
    H_total = H_tw * H_rlc

    H_dB = 20.0 * np.log10(np.abs(H_total) / np.abs(H_total[0]))
    if len(H_dB) >= 5:
        H_dB[1:] = gaussian_filter1d(H_dB[1:], sigma=2.0, mode="nearest")
    above = H_dB > -3.0
    if not above[0]:
        return float("nan")
    cross = np.argmax(~above)
    if cross == 0:
        return float("nan")
    f_lo, f_hi = freqs_ext[cross - 1], freqs_ext[cross]
    H_lo, H_hi = H_dB[cross - 1], H_dB[cross]
    if H_hi == H_lo:
        return f_hi / 1e9
    return (f_lo + (f_hi - f_lo) * (-3.0 - H_lo) / (H_hi - H_lo)) / 1e9


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    rows = [r for r in rows
            if r.get("c_target_index") == 9
            and r.get("failed") is False
            and isinstance(r.get("objective"), (int, float))
            and r["objective"] < 1e9]
    inc = min(rows, key=lambda r: r["objective"])
    print("Incumbent (strict, best J=0.532):")
    print(f"  Z0={inc['Z0_re_f0']:.2f}  n_eff={inc['n_eff_f0']:.3f}")
    print(f"  C_pn={inc['junction_C_pF_per_cm']:.2f} pF/cm, "
          f"R={inc['junction_R_ohm_cm']:.4f} Ohm.cm, "
          f"VpiL={inc['junction_VpiL_V_cm']:.4f} V.cm")

    J = Junction(C_pF_per_cm=inc["junction_C_pF_per_cm"],
                 R_ohm_cm=inc["junction_R_ohm_cm"],
                 VpiL_V_cm=inc["junction_VpiL_V_cm"],
                 n_group_opt=3.88)
    L_um = mzm_length_um(target_ER_dB=5.0,
                         VpiL_per_arm_V_cm=J.VpiL_V_cm,
                         V_pp=2.0, push_pull=True)
    print(f"  MZM length: {L_um:.0f} um")

    # Try to load incumbent's cached CPSResult for its alpha(f) curve.
    incumbent_alpha = None
    inc_freqs = None
    for p in Path("cache_step2").rglob(f"*{inc['geometry_hash']}*"):
        if p.suffix == ".pkl":
            try:
                with open(p, "rb") as f:
                    cps = pickle.load(f)
                incumbent_alpha = np.asarray(cps.alpha_dB_cm_bare)
                inc_freqs = np.asarray(cps.freqs)
                print(f"  Loaded alpha(f) from {p.name}: "
                      f"alpha(f0={inc_freqs[len(inc_freqs)//2]/1e9:.0f}GHz)"
                      f" = {incumbent_alpha[len(inc_freqs)//2]:.2f} dB/cm")
                break
            except Exception as e:
                print(f"  (could not unpickle {p}: {e})")

    if incumbent_alpha is None:
        inc_freqs = np.linspace(10e9, 40e9, 51)
        incumbent_alpha = 1.0 * np.sqrt(inc_freqs / 10e9)
        print("  No cache loaded; assuming alpha = 1 dB/cm at 10 GHz, "
              "sqrt(f) scaling")

    freqs = inc_freqs

    print("\n=== Sanity check ===")
    bw_inc = compute_bw(inc["Z0_re_f0"], inc["n_eff_f0"],
                        incumbent_alpha, freqs, J, L_um)
    print(f"Incumbent (Z0={inc['Z0_re_f0']:.2f}, n_eff={inc['n_eff_f0']:.3f}, "
          f"alpha 1x): BW = {bw_inc:.2f} GHz")
    print("  (cross-check vs step2_bandwidth_sweep.json incumbent BW.)")

    print("\n=== Hypothetical 'J-optimum' (Z0=18, n_eff=4.64) "
          "vs alpha scale ===")
    for a_scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        bw = compute_bw(18.0, 4.64, incumbent_alpha * a_scale,
                        freqs, J, L_um)
        a_at_f0 = incumbent_alpha[len(freqs)//2] * a_scale
        print(f"  alpha = {a_scale:>4.2f}x ({a_at_f0:>5.2f} dB/cm at f0): "
              f"BW = {bw:.2f} GHz")

    print("\n=== Z0 sweep along the Pareto (n_eff = 0.258*Z0, "
          "alpha = 1x incumbent) ===")
    print(f"{'Z0':>5} {'n_eff':>6} {'J':>6} {'BW [GHz]':>9}")
    for Z in [15, 17, 18, 19, 20, 21, 22, 25, 30]:
        n = 0.258 * Z
        J_val = ((Z - 50) / 50) ** 2 + ((n - 3.88) / 3.88) ** 2
        bw = compute_bw(Z, n, incumbent_alpha, freqs, J, L_um)
        print(f"{Z:>5.1f} {n:>6.3f} {J_val:>6.4f} {bw:>9.2f}")


if __name__ == "__main__":
    main()
