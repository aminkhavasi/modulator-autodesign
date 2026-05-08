"""Step-2 batch orchestrator.

Each invocation of this CLI does ONE batch and stops, journaling the result.
The agent (or you) decide when to issue the next command.

Examples
--------
# Step 0: pick the 10 outer-loop C targets from Step-1's journal
python -m step2.select_C_targets

# Step 1 of inner loop for c_target_index=0: 8 LHS samples
python -m step2.run_batch lhs --c-target 0 --n 8

# Print review for c_target_index=0
python -m step2.run_batch review --c-target 0

# Next 4 BO samples for c_target_index=0
python -m step2.run_batch bo --c-target 0 --n 4

# Show all C targets and their current best-objective
python -m step2.run_batch overview

# Final: bandwidth sweep across all 10 C targets (no FDTD; post-processing)
python -m step2.run_batch bandwidth_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fab_rules import DEFAULT_RULES
from .geom import CPSGeometry
from .journal import (
    JOURNAL, append_evaluation, attach_agent_notes, best_so_far,
    filter_successes, per_C_history,
)
from .junction import (
    Junction, bandwidth_3dB_GHz, mzm_length_um,
)
from .objective import objective
from .propose import perturb, propose_bo, propose_lhs, PARAM_ORDER
from .review import review_C_target
from .select_C_targets import read_targets
from .simulate import evaluate_cps


# Soft / hard caps on per-C runs (matches user's choice in chat)
SOFT_BUDGET = 20
HARD_BUDGET = 40

# Default ER spec for MZM length calc (per user)
DEFAULT_ER_DB = 5.0
DEFAULT_VPP = 2.0


# --- Helpers ---------------------------------------------------------------

def _target_for_index(c_idx: int) -> dict:
    targets = read_targets()
    for t in targets:
        if t["c_target_index"] == c_idx:
            return t
    raise ValueError(f"No target with c_target_index={c_idx}")


def _junction_from_target(target: dict, *, n_group_opt: float = 3.88
                          ) -> Junction:
    return Junction(
        C_pF_per_cm=target["C_pF_per_cm"],
        R_ohm_cm=target["R_ohm_cm"],
        VpiL_V_cm=target["VpiL_V_cm"],
        n_group_opt=n_group_opt,
    )


def _budget_check(c_idx: int, requested_n: int, *, allow_hard: bool = False
                  ) -> int:
    """Refuse to run if budget would be exceeded.  Returns n actually allowed."""
    used = len([r for r in per_C_history(c_idx)
                if not r.get("meta")])
    cap = HARD_BUDGET if allow_hard else SOFT_BUDGET
    remaining = cap - used
    if remaining <= 0:
        msg = (f"C target {c_idx} has used {used}/{cap} runs.  "
               f"Re-run with --allow-hard to extend up to {HARD_BUDGET}.")
        if used >= HARD_BUDGET:
            msg = (f"C target {c_idx} has used {used} runs >= "
                   f"hard cap {HARD_BUDGET}.  No further runs allowed without "
                   f"explicit user override.")
        raise SystemExit(msg)
    return min(requested_n, remaining)


def _evaluate_batch_and_journal(geom_dicts: list[dict], target: dict,
                                batch_id: str, c_idx: int, *,
                                retry_on_failure: bool = True):
    """Run a batch of FDTD evaluations, retry failures once, journal everything.

    All non-cached designs in `geom_dicts` are submitted to Tidy3D as a single
    batch, so they run in parallel on the cloud.  After the main batch
    finishes, any failures are perturbed and re-submitted in a small retry
    batch.  Both the original (failed) and retry results are journaled.
    """
    from .simulate import evaluate_cps_batch  # imported here to avoid
                                              # photonforge slow-imports at
                                              # module load when running
                                              # journal-only commands

    # Main batch
    geoms = [CPSGeometry(**gd) for gd in geom_dicts]
    print(f"=== submitting batch of {len(geoms)} designs ===")
    cps_results = evaluate_cps_batch(geoms)

    # Identify failures for retry
    failed_indices = [i for i, c in enumerate(cps_results) if c.failed]
    if failed_indices and retry_on_failure:
        print(f"  ! {len(failed_indices)} design(s) failed; preparing retry batch")
        retry_dicts = []
        retry_owner = []  # parallel list: index in original batch
        for i in failed_indices:
            seed = int(cps_results[i].geometry_hash[:8], 16) % 2**32
            retry_dicts.append(perturb(geom_dicts[i], scale=0.02, seed=seed))
            retry_owner.append(i)
        retry_geoms = [CPSGeometry(**gd) for gd in retry_dicts]
        retry_results = evaluate_cps_batch(retry_geoms)

        # Log the original failures with "_failed" suffix
        for owner_idx, orig_cps in zip(retry_owner,
                                       [cps_results[i] for i in failed_indices]):
            _journal_one(geom_dicts[owner_idx], target,
                         batch_id, c_idx, orig_cps,
                         suffix="_failed")
        # Replace the failed slots with retry results & geometry
        for owner_idx, new_geom_dict, new_cps in zip(retry_owner, retry_dicts,
                                                     retry_results):
            cps_results[owner_idx] = new_cps
            geom_dicts[owner_idx] = new_geom_dict

    # Journal final results
    for gd, cps in zip(geom_dicts, cps_results):
        _journal_one(gd, target, batch_id, c_idx, cps)


def _journal_one(geom_dict, target, batch_id, c_idx, cps, *, suffix=""):
    junction = _junction_from_target(target)
    if cps.failed:
        obj = float("inf")
        components = {"objective": float("inf"), "Z0_re_f0": float("nan"),
                      "n_eff_f0": float("nan"),
                      "Z0_term": float("nan"), "n_term": float("nan")}
    else:
        components = objective(cps, junction)
        obj = components["objective"]

    entry = {
        "c_target_index": c_idx,
        "batch_id": batch_id + suffix,
        "geometry": geom_dict,
        "geometry_hash": cps.geometry_hash,
        "objective": obj,
        "Z0_re_f0": components["Z0_re_f0"],
        "n_eff_f0": components["n_eff_f0"],
        "target_n_group": junction.n_group_opt,
        "target_C_pF_per_cm": target["C_target_pF_per_cm"],
        "junction_C_pF_per_cm": target["C_pF_per_cm"],
        "junction_R_ohm_cm": target["R_ohm_cm"],
        "junction_VpiL_V_cm": target["VpiL_V_cm"],
        "failed": cps.failed,
        "failure_reasons": cps.failure_reasons,
        "wall_time_s": cps.wall_time_s,
    }
    append_evaluation(entry)


# --- LHS subcommand --------------------------------------------------------

def cmd_lhs(args):
    target = _target_for_index(args.c_target)
    n = _budget_check(args.c_target, args.n, allow_hard=args.allow_hard)
    candidates = propose_lhs(n, rules=DEFAULT_RULES, seed=args.seed)
    print(f"=== LHS batch: c_target={args.c_target}, n={n} ===")
    for i, geom in enumerate(candidates):
        print(f"  [lhs {i+1}/{n}] {geom}")
    _evaluate_batch_and_journal(candidates, target, batch_id="lhs",
                                c_idx=args.c_target,
                                retry_on_failure=not args.no_retry)
    notes = review_C_target(args.c_target,
                            target_C_pF_per_cm=target["C_target_pF_per_cm"],
                            target_VpiL_V_cm=target["VpiL_V_cm"])
    print("\n" + notes)
    attach_agent_notes(args.c_target, "lhs_complete", notes)


# --- BO subcommand ---------------------------------------------------------

def cmd_bo(args):
    target = _target_for_index(args.c_target)
    n = _budget_check(args.c_target, args.n, allow_hard=args.allow_hard)
    successes = filter_successes(per_C_history(args.c_target))
    successes = [r for r in successes if not r.get("meta")]
    if len(successes) < 4:
        raise SystemExit(
            f"Need >= 4 successful runs to fit BO; have {len(successes)}.  "
            "Run more LHS first."
        )
    history = [(r["geometry"], r["objective"]) for r in successes]
    candidates = propose_bo(history, n, rules=DEFAULT_RULES)

    # Determine batch label
    n_prior_bo = len({r["batch_id"] for r in per_C_history(args.c_target)
                      if r.get("batch_id", "").startswith("bo_")})
    batch_id = f"bo_{n_prior_bo + 1}"

    print(f"=== BO batch '{batch_id}': c_target={args.c_target}, n={n} ===")
    for i, geom in enumerate(candidates):
        print(f"  [{batch_id} {i+1}/{n}] {geom}")
    _evaluate_batch_and_journal(candidates, target, batch_id=batch_id,
                                c_idx=args.c_target,
                                retry_on_failure=not args.no_retry)
    notes = review_C_target(args.c_target,
                            target_C_pF_per_cm=target["C_target_pF_per_cm"],
                            target_VpiL_V_cm=target["VpiL_V_cm"])
    print("\n" + notes)
    attach_agent_notes(args.c_target, f"{batch_id}_complete", notes)


# --- Review subcommand -----------------------------------------------------

def cmd_review(args):
    target = _target_for_index(args.c_target)
    print(review_C_target(args.c_target,
                          target_C_pF_per_cm=target["C_target_pF_per_cm"],
                          target_VpiL_V_cm=target["VpiL_V_cm"]))


# --- Overview --------------------------------------------------------------

def cmd_overview(args):
    targets = read_targets()
    print(f"{'idx':>3}  {'C_targ':>8}  {'mult':>6}  {'#runs':>5}  "
          f"{'#succ':>5}  {'best_obj':>10}  {'Z0':>7}  {'n_eff':>7}")
    for t in targets:
        rows = [r for r in per_C_history(t["c_target_index"])
                if not r.get("meta")]
        succ = filter_successes(rows)
        best = best_so_far(t["c_target_index"])
        print(f"{t['c_target_index']:>3d}  "
              f"{t['C_target_pF_per_cm']:>8.3f}  "
              f"{t['step1_mult']:>6.2f}  "
              f"{len(rows):>5d}  {len(succ):>5d}  "
              f"{(best['objective'] if best else float('nan')):>10.5f}  "
              f"{(best['Z0_re_f0'] if best else float('nan')):>7.2f}  "
              f"{(best['n_eff_f0'] if best else float('nan')):>7.3f}")


# --- Final bandwidth sweep -------------------------------------------------

def cmd_bandwidth_sweep(args):
    """For each C target, take its best CPS design and compute bandwidth."""
    targets = read_targets()
    out = []
    for t in targets:
        c_idx = t["c_target_index"]
        best = best_so_far(c_idx)
        if best is None:
            print(f"!! C target {c_idx}: no successful runs yet")
            continue

        # Re-load CPSResult from cache (geometry_hash drives the cache key)
        from .simulate import evaluate_cps  # cache will hit
        cps = evaluate_cps(CPSGeometry(**best["geometry"]))
        junction = _junction_from_target(t)

        L_um = mzm_length_um(args.er_dB, junction.VpiL_V_cm, args.vpp,
                             push_pull=True)
        bw_GHz = bandwidth_3dB_GHz(cps, junction, L_um,
                                   L_parasitic_H=args.L_parasitic)

        record = {
            "c_target_index": c_idx,
            "C_pF_per_cm": junction.C_pF_per_cm,
            "VpiL_V_cm": junction.VpiL_V_cm,
            "best_obj": best["objective"],
            "Z0_re_f0": best["Z0_re_f0"],
            "n_eff_f0": best["n_eff_f0"],
            "MZM_length_um": L_um,
            "bandwidth_3dB_GHz": bw_GHz,
            "geometry": best["geometry"],
        }
        out.append(record)
        print(f"C={junction.C_pF_per_cm:.3f}  VpiL={junction.VpiL_V_cm:.3f}  "
              f"L={L_um:.0f}um  Z0={best['Z0_re_f0']:.1f}  "
              f"n_eff={best['n_eff_f0']:.3f}  BW={bw_GHz:.1f} GHz")

    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=lambda o: str(o))
    print(f"\nWrote {out_path}")


# --- Argparse plumbing -----------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_lhs = sub.add_parser("lhs")
    p_lhs.add_argument("--c-target", type=int, required=True)
    p_lhs.add_argument("--n", type=int, default=8)
    p_lhs.add_argument("--seed", type=int, default=42)
    p_lhs.add_argument("--allow-hard", action="store_true",
                       help="extend budget from soft (20) to hard (40) cap")
    p_lhs.add_argument("--no-retry", action="store_true",
                       help="disable auto-retry on FDTD failure")
    p_lhs.set_defaults(func=cmd_lhs)

    p_bo = sub.add_parser("bo")
    p_bo.add_argument("--c-target", type=int, required=True)
    p_bo.add_argument("--n", type=int, default=4)
    p_bo.add_argument("--allow-hard", action="store_true")
    p_bo.add_argument("--no-retry", action="store_true")
    p_bo.set_defaults(func=cmd_bo)

    p_rev = sub.add_parser("review")
    p_rev.add_argument("--c-target", type=int, required=True)
    p_rev.set_defaults(func=cmd_review)

    p_ov = sub.add_parser("overview")
    p_ov.set_defaults(func=cmd_overview)

    p_bw = sub.add_parser("bandwidth_sweep")
    p_bw.add_argument("--er-dB", type=float, default=DEFAULT_ER_DB)
    p_bw.add_argument("--vpp", type=float, default=DEFAULT_VPP)
    p_bw.add_argument("--L-parasitic", type=float, default=0.0)
    p_bw.add_argument("--output", default="step2_bandwidth_sweep.json")
    p_bw.set_defaults(func=cmd_bandwidth_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
