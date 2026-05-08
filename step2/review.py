"""Review utilities for the agent's per-batch checkpoints.

The agent calls these between batches to summarize what's known so far,
what worked, and where the optimization seems to be heading.  Output is a
text report (Markdown-friendly) that ends up in the agent_notes journal field.
"""

from __future__ import annotations

import math

import numpy as np

from .journal import per_C_history, filter_successes, best_so_far
from .propose import PARAM_ORDER


def review_C_target(c_target_index: int, *,
                    target_C_pF_per_cm: float | None = None,
                    target_VpiL_V_cm: float | None = None) -> str:
    """Plain-text review of progress at one outer-loop C value.

    Returns a Markdown-flavored multi-paragraph string.  Suitable for piping
    into agent_notes after a batch.
    """
    rows = per_C_history(c_target_index)
    eval_rows = [r for r in rows if not r.get("meta")]
    successes = filter_successes(eval_rows)

    n_total = len(eval_rows)
    n_fail = n_total - len(successes)

    lines = []
    lines.append(f"# C target #{c_target_index}: progress review")
    if target_C_pF_per_cm is not None:
        lines.append(f"- target C: {target_C_pF_per_cm:.3f} pF/cm")
    if target_VpiL_V_cm is not None:
        lines.append(f"- corresponding VpiL: {target_VpiL_V_cm:.3f} V.cm")
    lines.append("")
    lines.append(f"**Evaluations so far:** {n_total} ({n_fail} failed)")
    if not successes:
        lines.append("\nNo successful runs yet -- cannot report convergence.")
        return "\n".join(lines)

    best = best_so_far(c_target_index)
    lines.append("")
    lines.append("**Best so far:**")
    lines.append(f"- objective = {best['objective']:.5f}")
    lines.append(f"- Z0_re(f0) = {best['Z0_re_f0']:.2f} ohm  "
                 f"(target 50)")
    lines.append(f"- n_eff_rf(f0) = {best['n_eff_f0']:.3f}  "
                 f"(target {best.get('target_n_group','?')})")
    lines.append("- geometry:")
    for k in PARAM_ORDER:
        lines.append(f"    {k} = {best['geometry'][k]:.3f} um")

    # Convergence: best-so-far over time
    obj_series = []
    running = math.inf
    for r in successes:
        running = min(running, r["objective"])
        obj_series.append(running)
    lines.append("")
    lines.append("**Best-so-far trajectory:**")
    chunk = 4
    for i in range(0, len(obj_series), chunk):
        seg = obj_series[i:i + chunk]
        lines.append(f"  runs {i+1}..{i+len(seg)}: "
                     + " ".join(f"{v:.4f}" for v in seg))

    # Parameter correlations with objective (linear, only meaningful with N>=8)
    if len(successes) >= 6:
        lines.append("")
        lines.append("**Parameter-vs-objective correlations:**")
        ys = np.array([r["objective"] for r in successes])
        for k in PARAM_ORDER:
            xs = np.array([r["geometry"][k] for r in successes])
            if xs.std() < 1e-9 or ys.std() < 1e-9:
                continue
            corr = np.corrcoef(xs, ys)[0, 1]
            sign = "↓" if corr > 0 else "↑"
            lines.append(f"    {k}: r = {corr:+.2f}  ({sign} param "
                         f"-> {'higher' if corr>0 else 'lower'} objective)")

    # Boundary saturation: are best designs hitting the param bounds?
    lines.append("")
    lines.append("**Boundary saturation of best design:**")
    g = best["geometry"]
    for k in PARAM_ORDER:
        # Crude: check if value is close to either min or max of the
        # observed range.  This is a hint that bounds are limiting BO.
        all_vals = [r["geometry"][k] for r in successes]
        v = g[k]
        rel = (v - min(all_vals)) / max(1e-9, (max(all_vals) - min(all_vals)))
        flag = "  ← near LOW" if rel < 0.05 else (
            "  ← near HIGH" if rel > 0.95 else "")
        lines.append(f"    {k} = {v:.3f}  ({rel*100:.0f}% across observed range){flag}")

    # Failure summary
    if n_fail > 0:
        lines.append("")
        lines.append(f"**Failures ({n_fail}):**")
        for r in eval_rows:
            if r.get("failed"):
                reasons = r.get("failure_reasons", [])
                lines.append(f"  - {r.get('geometry_hash','??')[:8]}: "
                             f"{', '.join(reasons[:2]) or '(no reasons logged)'}")

    return "\n".join(lines)
