"""Round 2: refine the high-Z0 corner between R1_B (ws=5) and R1_H (ws=20).

R1 found that for c=7-9, ws=5 is best, while for c=0-2 ws=20 is best.
The optimum ws is c_target-dependent.  Probe intermediate ws and verify
g_max=200 is correct.

Skip the low-Z0 direction (R1 confirmed it's strictly worse for BW).

8 candidates, each cross-evaluated against all 10 c_targets.
"""

from __future__ import annotations

import time

from step2.fab_rules_v2 import RULES_V2
from step2.fab_rules import feasible, repair_geometry_constraint, clip_to_bounds
from step2.geom import CPSGeometry
from step2.journal import append_evaluation
from step2.junction import Junction, bandwidth_3dB_GHz, loaded_at_f0, mzm_length_um
from step2.select_C_targets import read_targets
from step2.simulate import evaluate_cps_batch


CANDIDATES = [
    # ws scan at g=200 between B (5) and H (20):
    dict(name="R2_A_ws7",   g=200.0, ws=7.0,  wg=7.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    dict(name="R2_B_ws12",  g=200.0, ws=12.0, wg=12.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    dict(name="R2_C_ws15",  g=200.0, ws=15.0, wg=15.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    dict(name="R2_D_ws18",  g=200.0, ws=18.0, wg=18.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    # Verify g_max=200 isn't over-shooting:
    dict(name="R2_E_g150",  g=150.0, ws=10.0, wg=10.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    # Sparse T at narrow rails (saw R1_C work at ws=10):
    dict(name="R2_F_ws5_sparseT", g=200.0, ws=5.0,  wg=5.0,
         s=2.0, r=80.0, h=2.0, t=1.0, c=10.0),
    # Wider rails + sparse T (for c=0,1,2 where wider ws works):
    dict(name="R2_G_ws40_sparseT", g=200.0, ws=40.0, wg=40.0,
         s=2.0, r=80.0, h=2.0, t=1.0, c=10.0),
    # H: explore mid-g, mid-rail compromise
    dict(name="R2_H_g100_ws50", g=100.0, ws=50.0, wg=50.0,
         s=2.0, r=20.0, h=2.0, t=1.0, c=2.0),
]


def main():
    fixed = []
    for c in CANDIDATES:
        name = c.pop("name")
        c = repair_geometry_constraint(c, rules=RULES_V2)
        c = clip_to_bounds(c, rules=RULES_V2)
        ok, viol = feasible(c, rules=RULES_V2)
        if not ok:
            raise SystemExit(f"{name} infeasible: {viol}")
        fixed.append((name, c))

    print("=== Round 2 candidates ===")
    for n, c in fixed:
        print(f"  {n:24}  g={c['g']:>5.1f} ws={c['ws']:>5.1f} "
              f"wg={c['wg']:>5.1f}  r={c['r']:>4.1f} c={c['c']:>4.1f}")

    geoms = [CPSGeometry(**g) for _, g in fixed]
    print(f"\nSubmitting {len(geoms)} FDTDs ...")
    t0 = time.time()
    cps_results = evaluate_cps_batch(geoms)
    print(f"Batch done in {time.time()-t0:.0f} s\n")

    targets = read_targets()
    tbi = {t["c_target_index"]: t for t in targets}

    print(f"{'name':24}  " + " ".join(f"c{i:>1}" for i in range(10)))
    for (name, gd), cps in zip(fixed, cps_results):
        if cps.failed:
            print(f"  {name:22}  FAILED")
            continue
        bws = []
        for ci in range(10):
            t = tbi[ci]
            J = Junction(C_pF_per_cm=t["C_pF_per_cm"],
                         R_ohm_cm=t["R_ohm_cm"],
                         VpiL_V_cm=t["VpiL_V_cm"],
                         n_group_opt=3.88)
            L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
            bw = bandwidth_3dB_GHz(cps, J, L_um)
            Z0_re, n_eff, _ = loaded_at_f0(cps, J)
            bws.append(bw)
            append_evaluation({
                "c_target_index": ci,
                "batch_id": "bw_round2",
                "name": name,
                "geometry": gd,
                "geometry_hash": cps.geometry_hash,
                "objective": -float(bw) if bw == bw else 1e3,
                "BW_GHz": float(bw) if bw == bw else None,
                "MZM_length_um": L_um,
                "Z0_re_f0": float(Z0_re),
                "n_eff_f0": float(n_eff),
                "target_n_group": 3.88,
                "target_C_pF_per_cm": t["C_target_pF_per_cm"],
                "junction_C_pF_per_cm": t["C_pF_per_cm"],
                "junction_R_ohm_cm": t["R_ohm_cm"],
                "junction_VpiL_V_cm": t["VpiL_V_cm"],
                "failed": False,
                "failure_reasons": [],
                "wall_time_s": cps.wall_time_s,
            })
        print(f"  {name:22}  " + " ".join(
            f"{b:5.1f}" if b == b else "  NaN" for b in bws))


if __name__ == "__main__":
    main()
