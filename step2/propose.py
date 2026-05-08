"""Candidate proposers for the inner-loop CPS optimization.

Two algorithms:
  propose_lhs(n_samples, rules, seed)
      Latin Hypercube sample of N points in the 8-D bounded box.  Used at
      the start of each per-C inner loop (8 samples by default).

  propose_bo(history, n_samples, rules)
      Bayesian-optimization proposal via skopt's GP + Expected Improvement.
      `history` is a list of (geom_dict, objective_scalar) pairs from
      successful previous evaluations.

Both functions auto-repair candidates that violate the geometric constraint
2*(s+h) + safety_gap <= g  (see fab_rules.repair_geometry_constraint),
since rejecting infeasible samples wastes information.
"""

from __future__ import annotations

import warnings

import numpy as np

from .fab_rules import (
    DEFAULT_RULES, FabRules, clip_to_bounds,
    feasible, repair_geometry_constraint,
)


# Stable parameter ordering -- skopt and LHS both rely on this.
PARAM_ORDER = ("g", "ws", "wg", "s", "r", "h", "t", "c")


# --- Latin Hypercube --------------------------------------------------------

def propose_lhs(n_samples: int, *,
                rules: FabRules = DEFAULT_RULES,
                seed: int = 42) -> list[dict]:
    """Generate `n_samples` Latin Hypercube points in the 8-D box."""
    try:
        from scipy.stats import qmc
    except ImportError as e:
        raise RuntimeError(
            "scipy.stats.qmc required for LHS.  scipy >= 1.7 should have it."
        ) from e

    bounds = _param_bounds(rules)
    sampler = qmc.LatinHypercube(d=len(PARAM_ORDER), seed=seed)
    raw = sampler.random(n_samples)  # shape (n_samples, 8) in [0, 1]
    lo = np.array([b[0] for b in bounds.values()])
    hi = np.array([b[1] for b in bounds.values()])
    pts = lo + raw * (hi - lo)

    candidates = []
    for row in pts:
        geom = dict(zip(PARAM_ORDER, row.tolist()))
        geom = repair_geometry_constraint(geom, rules=rules)
        geom = clip_to_bounds(geom, rules=rules)
        ok, _ = feasible(geom, rules=rules)
        if not ok:
            # Couldn't repair within bounds.  Fall back: bump g to its max.
            geom["g"] = rules.g_max
            geom = clip_to_bounds(geom, rules=rules)
        candidates.append(geom)
    return candidates


# --- Bayesian optimization -------------------------------------------------

def propose_bo(history: list[tuple[dict, float]], n_samples: int, *,
               rules: FabRules = DEFAULT_RULES,
               n_random_starts: int = 0,
               acq_func: str = "EI") -> list[dict]:
    """Propose `n_samples` next points via skopt Bayesian optimization.

    `history`: list of (geometry_dict, objective_value).  Failed runs should
    be EXCLUDED (skopt.GP-based BO assumes finite, smooth objective values).

    `acq_func`: skopt acquisition.  "EI" = Expected Improvement (recommended).
    """
    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as e:
        raise RuntimeError(
            "scikit-optimize required.  pip install scikit-optimize"
        ) from e

    bounds = _param_bounds(rules)
    space = [Real(lo, hi, name=k) for k, (lo, hi) in bounds.items()]
    optimizer = Optimizer(
        dimensions=space,
        base_estimator="GP",
        n_initial_points=n_random_starts,
        acq_func=acq_func,
        random_state=42,
    )

    # Tell the optimizer about prior history
    for geom, y in history:
        x = [geom[k] for k in PARAM_ORDER]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # skopt complains about unrepaired clipping
            optimizer.tell(x, float(y))

    # Ask for n_samples points all at once (kappa-batch heuristic in skopt
    # uses the constant-liar strategy).
    candidates_x = optimizer.ask(n_points=n_samples,
                                 strategy="cl_min")  # "constant liar - min"
    out = []
    for x in candidates_x:
        geom = dict(zip(PARAM_ORDER, [float(v) for v in x]))
        geom = repair_geometry_constraint(geom, rules=rules)
        geom = clip_to_bounds(geom, rules=rules)
        ok, _ = feasible(geom, rules=rules)
        if not ok:
            geom["g"] = rules.g_max
            geom = clip_to_bounds(geom, rules=rules)
        out.append(geom)
    return out


# --- Local perturbation (used for failure auto-retry) ----------------------

def perturb(geom: dict, *,
            rules: FabRules = DEFAULT_RULES,
            scale: float = 0.02,
            seed: int | None = None) -> dict:
    """Random multiplicative perturbation of all 8 params, repaired."""
    rng = np.random.default_rng(seed)
    out = dict(geom)
    for k in PARAM_ORDER:
        out[k] *= (1.0 + scale * rng.standard_normal())
    out = repair_geometry_constraint(out, rules=rules)
    out = clip_to_bounds(out, rules=rules)
    return out


# --- Helper ----------------------------------------------------------------

def _param_bounds(rules: FabRules) -> dict:
    return {
        "g":  (rules.g_min,  rules.g_max),
        "ws": (rules.ws_min, rules.ws_max),
        "wg": (rules.wg_min, rules.wg_max),
        "s":  (rules.s_min,  rules.s_max),
        "r":  (rules.r_min,  rules.r_max),
        "h":  (rules.h_min,  rules.h_max),
        "t":  (rules.t_min,  rules.t_max),
        "c":  (rules.c_min,  rules.c_max),
    }
