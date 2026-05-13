"""Probe eo_s21 directly.

Test 1: matched lossless (Z0=ZS=ZL=50, alpha=0, n_rf=n_group) at f=10 GHz.
        Expected: |H| = 1 exactly (pure phase delay).
Test 2: matched lossy (Z0=50, n_rf=n_group, alpha=3.18 dB/cm) at f=10 GHz.
        Expected: |H| < 1 by some sinh-attenuation factor.
Test 3: the actual Z=90 design at 10 GHz numbers from the diagnostic.
        Expected: |H| < 1 if formula is correct; if |H| > 1, formula is wrong.
"""

from __future__ import annotations

import numpy as np

from step2.junction import eo_s21


def call(label, Z0, n_rf, n_group, alpha_dB_cm, f_GHz=10.0, length_um=1325):
    alpha_dB_um = alpha_dB_cm / 1e4
    f = f_GHz * 1e9
    H = eo_s21(rf_loss_dB_um=np.array([alpha_dB_um]),
               freqs=np.array([f]),
               n_rf=np.array([n_rf]),
               n_group=n_group,
               length_um=length_um,
               ZL=50.0, ZS=50.0,
               Z0=np.array([Z0]))
    mag = float(np.abs(H[0]))
    print(f"{label:55}  H = {H[0]:.4f}  |H| = {mag:.4f}  dB = {20*np.log10(mag):+.2f}")
    return mag


def main():
    print("=== Probe eo_s21 directly ===")
    print(f"\nTest 1: matched, lossless, n_rf=n_group (should give |H|=1)")
    call("Z0=50, n=3.88=n_g, alpha=0,  f=10 GHz",
         Z0=50.0, n_rf=3.88, n_group=3.88, alpha_dB_cm=0.0)
    call("Z0=50, n=3.88=n_g, alpha=0,  f=25 GHz",
         Z0=50.0, n_rf=3.88, n_group=3.88, alpha_dB_cm=0.0, f_GHz=25)
    call("Z0=50, n=3.88=n_g, alpha=0,  f=40 GHz",
         Z0=50.0, n_rf=3.88, n_group=3.88, alpha_dB_cm=0.0, f_GHz=40)

    print(f"\nTest 2: matched, lossy (Z=50, n=3.88, alpha=3.18 dB/cm — Z=52 design's α)")
    call("Z0=50, n=3.88, alpha=3.18,  f=10 GHz",
         Z0=50.0, n_rf=3.88, n_group=3.88, alpha_dB_cm=3.18)

    print(f"\nTest 3: actual Z=90 design at 10 GHz "
          "(Z0=98.05-4.26j, n=5.21, alpha=6.92 dB/cm)")
    call("Z0=98.05-4.26j, n=5.21, alpha=6.92,  f=10 GHz",
         Z0=98.05 - 4.26j, n_rf=5.21, n_group=3.88, alpha_dB_cm=6.92)

    print(f"\nTest 4: actual Z=52 design at 10 GHz "
          "(Z0=51.53-2.00j, n=3.85, alpha=3.18 dB/cm)")
    call("Z0=51.53-2.00j, n=3.85, alpha=3.18,  f=10 GHz",
         Z0=51.53 - 2.00j, n_rf=3.85, n_group=3.88, alpha_dB_cm=3.18)

    print(f"\nTest 5: extra — Z0=50, n=5.21 (walk-off only, no Z mismatch)")
    call("Z0=50, n=5.21, alpha=0,  f=10 GHz",
         Z0=50.0, n_rf=5.21, n_group=3.88, alpha_dB_cm=0.0)

    print(f"\nTest 6: Z0=98 (mismatch only, no walk-off, no loss)")
    call("Z0=98, n=3.88, alpha=0,  f=10 GHz",
         Z0=98.0, n_rf=3.88, n_group=3.88, alpha_dB_cm=0.0)


if __name__ == "__main__":
    main()
