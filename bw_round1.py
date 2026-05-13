"""Round 1 of the BW-objective optimization pass under RULES_V2.

Eight hand-picked candidates spanning the two new corners that V2 opens:

  HIGH-Z0 (heavy-loading, c=7-9 needs Z0 up):
    A: g=200, ws=10, wg=10, mid-T            (narrow rails to lift Z0)
    B: g=200, ws=5,  wg=5,  fine T           (narrowest rails + finest T)
    C: g=200, ws=10, wg=10, sparse T (r=80)  (sparse T to drop native C0)
    D: g=200, ws=5,  wg=200, fine T          (asymmetric, narrow signal)

  LOW-Z0 (light-loading, c=1 Z0 too high -> needs Z0 down):
    E: g=5,  ws=200, wg=200, mid-T   (smallest gap, biggest rails -> high C0)
    F: g=10, ws=200, wg=200, mid-T
    G: g=20, ws=200, wg=200, dense T (r=5, c=1.5)

  MID-LOAD (c=3-6 already close, try slight v2 tweak):
    H: g=200, ws=20, wg=20, finest T everywhere (s=r=h=c=1.5, t=1)

Each gets a single FDTD + automatic cross-evaluation against all 10
c_targets. Journals BW_GHz under each c_target.
"""

from __future__ import annotations

import json
import time

from step2.fab_rules_v2 import RULES_V2
from step2.fab_rules import feasible, repair_geometry_constraint, clip_to_bounds
from step2.geom import CPSGeometry
from step2.journal import append_evaluation
from step2.junction import Junction, bandwidth_3dB_GHz, loaded_at_f0, mzm_length_um
from step2.select_C_targets import read_targets
from step2.simulate import evaluate_cps_batch


CANDIDATES = [
    dict(name="R1_A_highZ_ws10",   g=200.0, ws=10.0, wg=10.0,
         s=2.0, r=20.0, h=2.0, t=1.0, c=2.0),
    dict(name="R1_B_highZ_ws5_fineT", g=200.0, ws=5.0,  wg=5.0,
         s=2.0, r=5.0,  h=2.0, t=1.0, c=1.5),
    dict(name="R1_C_highZ_sparseT",   g=200.0, ws=10.0, wg=10.0,
         s=2.0, r=80.0, h=2.0, t=1.0, c=10.0),
    dict(name="R1_D_highZ_asym",      g=200.0, ws=5.0,  wg=200.0,
         s=2.0, r=10.0, h=2.0, t=1.0, c=2.0),
    dict(name="R1_E_lowZ_g5",         g=5.0,   ws=200.0, wg=200.0,
         s=2.0, r=20.0, h=2.0, t=2.0, c=2.0),
    dict(name="R1_F_lowZ_g10",        g=10.0,  ws=200.0, wg=200.0,
         s=2.0, r=20.0, h=2.0, t=2.0, c=2.0),
    dict(name="R1_G_lowZ_g20_denseT", g=20.0,  ws=200.0, wg=200.0,
         s=2.0, r=5.0,  h=2.0, t=2.0, c=1.5),
    dict(name="R1_H_finest_at200",    g=200.0, ws=20.0, wg=20.0,
         s=2.0, r=5.0,  h=2.0, t=1.0, c=1.5),
]


def main():
    fixed = []
    for c in CANDIDATES:
        name = c.pop("name")
        c = repair_geometry_constraint(c, rules=RULES_V2)
        c = clip_to_bounds(c, rules=RULES_V2)
        ok, viol = feasible(c, rules=RULES_V2)
        if not ok:
            raise SystemExit(f"{name} infeasible after repair: {viol}")
        fixed.append((name, c))

    print("=== Round 1 candidates (RULES_V2, 5 μm rail/gap min) ===")
    for n, c in fixed:
        print(f"  {n:24}  "
              f"g={c['g']:>5.1f} ws={c['ws']:>5.1f} wg={c['wg']:>5.1f}  "
              f"s={c['s']:>4.1f} r={c['r']:>4.1f} h={c['h']:>4.1f} "
              f"t={c['t']:>4.1f} c={c['c']:>4.1f}")

    geoms = [CPSGeometry(**g) for _, g in fixed]
    print(f"\n=== Submitting batch of {len(geoms)} FDTDs ===")
    t0 = time.time()
    cps_results = evaluate_cps_batch(geoms)
    print(f"Batch done in {time.time()-t0:.0f} s")

    # Cross-evaluate each result against every c_target's junction.
    targets = read_targets()
    target_by_idx = {t["c_target_index"]: t for t in targets}
    print("\n=== Cross-eval BW per c_target (new BW vs prior best) ===")
    header = f"{'name':24}  " + "  ".join(f"c{i}" for i in range(10))
    print(header)

    for (name, gd), cps in zip(fixed, cps_results):
        line = f"  {name:22}  "
        if cps.failed:
            line += "FAILED  " + ",".join(cps.failure_reasons)
            print(line)
            continue
        bws_line = []
        for ci in range(10):
            t = target_by_idx[ci]
            J = Junction(C_pF_per_cm=t["C_pF_per_cm"],
                         R_ohm_cm=t["R_ohm_cm"],
                         VpiL_V_cm=t["VpiL_V_cm"],
                         n_group_opt=3.88)
            L_um = mzm_length_um(5.0, J.VpiL_V_cm, 2.0, push_pull=True)
            bw = bandwidth_3dB_GHz(cps, J, L_um)
            Z0_re, n_eff, _ = loaded_at_f0(cps, J)
            bws_line.append(bw)
            append_evaluation({
                "c_target_index": ci,
                "batch_id": "bw_round1",
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
        line += " ".join(f"{b:5.1f}" if b == b else "  NaN" for b in bws_line)
        print(line)


if __name__ == "__main__":
    main()
