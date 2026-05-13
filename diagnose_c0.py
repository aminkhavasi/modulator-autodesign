"""Side-by-side diagnostic for c=0: the Z=90 "winner" vs the Z=52 matched
design.  Why does matched lose by 0.33 GHz?

Look at:
  Z0_loaded(f), n_eff_loaded(f), alpha_loaded(f)
  |H_tw(f)| (traveling-wave), |H_rlc(f)| (RC divider), |H_total(f)|
  on the same axes.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from step2.junction import (Junction, apply_junction_loading, eo_response,
                            eo_s21, mzm_length_um)


# Two designs we want to compare (both for c=0)
HASH_Z90 = "01a12c56320e"   # R1_H_finest_at200 (Z=90, n=4.79, BW=39.89)
HASH_Z52 = None             # find from journal: cross_eval g=127.8 ws=242 wg=54


def find_hash_for_geom(rows, ci, g_match):
    for r in rows:
        if r.get("c_target_index") != ci or "geometry" not in r:
            continue
        g = r["geometry"]
        if (abs(g["g"] - g_match["g"]) < 0.5
                and abs(g["ws"] - g_match["ws"]) < 1
                and abs(g["wg"] - g_match["wg"]) < 1):
            return r["geometry_hash"]
    return None


def main():
    rows = [json.loads(l)
            for l in open("step2_journal.jsonl", encoding="utf-8")]
    cache_dir = Path("cache_step2")

    g_z52 = {"g": 127.8, "ws": 242, "wg": 54}
    h_z52 = find_hash_for_geom(rows, 0, g_z52)
    print(f"Z90 design hash: {HASH_Z90}")
    print(f"Z52 design hash: {h_z52}")

    def load_cps(gh):
        pkls = [p for p in cache_dir.rglob(f"*{gh}*") if p.suffix == ".pkl"]
        return pickle.load(open(pkls[0], "rb"))

    cps_z90 = load_cps(HASH_Z90)
    cps_z52 = load_cps(h_z52)

    # c=0 junction
    r0 = next(r for r in rows if r["c_target_index"] == 0
              and isinstance(r.get("objective"), (int, float)))
    J = Junction(C_pF_per_cm=r0["junction_C_pF_per_cm"],
                 R_ohm_cm=r0["junction_R_ohm_cm"],
                 VpiL_V_cm=r0["junction_VpiL_V_cm"], n_group_opt=3.88)
    L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
    print(f"c=0: C_pp={J.C_pF_per_cm:.2f} pF/cm, R_pp={J.R_ohm_cm:.3f} Ohm.cm, "
          f"L={L_um:.0f} um")

    # Loaded line params
    Z_z90, n_z90, a_z90 = apply_junction_loading(cps_z90, J)
    Z_z52, n_z52, a_z52 = apply_junction_loading(cps_z52, J)
    freqs = cps_z90.freqs

    print(f"\n{'f (GHz)':>9}  {'Z_z90':>8} {'n_z90':>5} {'a_z90':>5}   "
          f"{'Z_z52':>8} {'n_z52':>5} {'a_z52':>5}")
    for i in [0, 8, 16, 25, 33, 41, 50]:
        f_GHz = freqs[i] / 1e9
        print(f"  {f_GHz:>6.1f}   "
              f"{Z_z90[i].real:>6.2f}+{Z_z90[i].imag:>5.2f}j "
              f"{n_z90[i]:>5.2f} {a_z90[i]:>5.2f}   "
              f"{Z_z52[i].real:>6.2f}+{Z_z52[i].imag:>5.2f}j "
              f"{n_z52[i]:>5.2f} {a_z52[i]:>5.2f}")

    # Full EO response
    f_ext_z90, H_z90, Htw_z90 = eo_response(cps_z90, J, L_um)
    f_ext_z52, H_z52, Htw_z52 = eo_response(cps_z52, J, L_um)
    H_dB_z90 = 20.0 * np.log10(np.abs(H_z90)/np.abs(H_z90[0]))
    H_dB_z52 = 20.0 * np.log10(np.abs(H_z52)/np.abs(H_z52[0]))
    Htw_dB_z90 = 20.0 * np.log10(np.abs(Htw_z90)/np.abs(Htw_z90[0]))
    Htw_dB_z52 = 20.0 * np.log10(np.abs(Htw_z52)/np.abs(Htw_z52[0]))

    print("\n=== EO response (dB) ===")
    print(f"{'f (GHz)':>9}  {'H_tw_z90':>9} {'H_tot_z90':>10}   "
          f"{'H_tw_z52':>9} {'H_tot_z52':>10}")
    for i in range(1, len(f_ext_z90), 6):
        f_GHz = f_ext_z90[i] / 1e9
        print(f"  {f_GHz:>6.1f}   "
              f"{Htw_dB_z90[i]:>+8.2f}  {H_dB_z90[i]:>+8.2f}    "
              f"{Htw_dB_z52[i]:>+8.2f}  {H_dB_z52[i]:>+8.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(freqs/1e9, np.real(Z_z90), "C1-",
            label=f"Z=90 design (R1_H)")
    ax.plot(freqs/1e9, np.real(Z_z52), "C0-",
            label=f"Z=52 design (matched)")
    ax.axhline(50, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("Re(Z0_loaded) [Ω]")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Loaded characteristic impedance")

    ax = axes[0, 1]
    ax.plot(freqs/1e9, n_z90, "C1-")
    ax.plot(freqs/1e9, n_z52, "C0-")
    ax.axhline(3.88, color="r", ls="--", alpha=0.5,
               label="n_g_opt = 3.88")
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("n_eff_RF (loaded)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Loaded effective index")

    ax = axes[1, 0]
    ax.plot(freqs/1e9, a_z90, "C1-", label="Z=90")
    ax.plot(freqs/1e9, a_z52, "C0-", label="Z=52 matched")
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("alpha (dB/cm) loaded")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Loaded RF attenuation")

    ax = axes[1, 1]
    ax.plot(f_ext_z90[1:]/1e9, H_dB_z90[1:], "C1-",
            lw=2, label=f"Z=90 total")
    ax.plot(f_ext_z52[1:]/1e9, H_dB_z52[1:], "C0-",
            lw=2, label=f"Z=52 total")
    ax.plot(f_ext_z90[1:]/1e9, Htw_dB_z90[1:], "C1:",
            lw=1, label=f"Z=90 tw-only")
    ax.plot(f_ext_z52[1:]/1e9, Htw_dB_z52[1:], "C0:",
            lw=1, label=f"Z=52 tw-only")
    ax.axhline(-3, color="black", ls="--", alpha=0.4, label="-3 dB")
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("EO |H| (dB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"EO S21 (L_mod = {L_um:.0f} um)")
    ax.set_ylim(-10, 1)

    fig.suptitle("c=0 BW diagnostic: matched (Z=52) vs Z=90", y=1.00)
    fig.tight_layout()
    fig.savefig("field_plots/diag_c0_matched_vs_unmatched.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote field_plots/diag_c0_matched_vs_unmatched.png")


if __name__ == "__main__":
    main()
