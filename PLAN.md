# Modulator Auto-Design — Plan & Status

This is the source-of-truth planning document. Update it as decisions evolve.

## Project Overview

LLM-driven design loop for silicon photonic PN-junction modulators on the
SiEPIC EBeam platform. Uses Tidy3D (charge + mode solver) and PhotonForge
(geometry, technology), wrapped in a `Design → Verify → Simulate → Keep`
agent loop driven by Claude Code.

## Environment

- Windows 10/11, PowerShell, VS Code with integrated terminal.
- Python 3.10 in `.venv\` inside the project folder.
- Tidy3D 2.11.1 (Flexcompute API key already configured by user).
- PhotonForge 1.4.0 from prebuilt wheel
  (`wheels\photonforge-1.4.0-cp310-cp310-win_amd64.whl[live_viewer]`).
- siepic_forge installed separately.
- vtk installed manually by user (required by Tidy3D charge solver
  visualization).
- Git initialized locally; remote deferred. Commits work offline.
- All large/regenerable artifacts (`.venv\`, `cache\`, `wheels\*.whl`,
  `field_plots\*.png`, simulation `.hdf5`) excluded via `.gitignore`.

## Project Layout

```
modulator-autodesign/
├── .venv/                      # local Python env, gitignored
├── cache/                      # pickled charge data, mode-solver n_eff
├── field_plots/                # PNG outputs from plot_tradeoffs.py
├── wheels/                     # local wheel cache (gitignored)
├── .gitignore
├── README.md                   # install instructions
├── requirements.txt            # human-readable top-level deps
├── requirements-lock.txt       # full pip freeze
├── PLAN.md                     # this file
├── laplace.py                  # 2D Laplace solver (user-authored, untouched)
├── pn_junction.py              # physics: evaluate_design(mult)
├── run_sweep.py                # orchestrator + bracket-and-fill strategy
├── plot_tradeoffs.py           # plots from journal.jsonl
└── journal.jsonl               # append-only experiment log
```

`program.md` and `CLAUDE.md` (agent-facing files) are deferred until the
manual loop is verified to work end-to-end.

## Step 1 — PN Junction Trade-off Sweep

### Goal
Characterize the PN junction across doping and bias to produce a (VπL, C)
trade-off plot that Step 2 (traveling-wave electrode design) will consume.

### Source notebook
`TWModulator_VpiL_Loss.ipynb` — refactored cell-for-cell into `pn_junction.py`.

### Geometry (FIXED for Step 1)
SOI rib waveguide, 500 nm wide, 220 nm tall, 90 nm slab, with p/p+/p++ on
the left and n/n+/n++ on the right. Aluminum contacts on side regions.
SiEPIC EBeam technology, top oxide 1.2 µm, BOX 2.0 µm.

Access dopings (FIXED):
- `P_P_DOPING = 1.5e19`,  `N_P_DOPING = 1.2e19`  (cm⁻³)
- `P_PP_DOPING = 1e20`,   `N_PP_DOPING = 1e20`   (cm⁻³)

### Control variable
A single scalar `mult` such that
- `p_doping = 5e17 * mult`
- `n_doping = 3e17 * mult`
Allowed range: `mult ∈ [0.2, 20]`. Default anchor set: {0.2, 1.0, 5.0, 20.0}.

### Voltage sweep (FIXED)
9 reverse-bias values: `np.linspace(-0.5, 1.5, 9)` =
`[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`.

**Interior 7 values** are the trade-off evaluation points (endpoints excluded
because `np.gradient` produces less-accurate one-sided VπL there):
`{-0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25}`.

### Per-evaluation outputs
For each `mult`, for each of 9 voltages:
- `C` [pF/cm]    — capacitance per unit length, from the charge sim
- `R` [Ω·cm]     — series resistance, **voltage-dependent** (recomputed at
                    each bias because depletion width grows with reverse bias)
- `f3dB` [GHz]   — `1 / (2π R C)`
- `VπL` [V·cm]   — `π / (dφ/dV per cm)` from mode-solver complex n_eff
- `loss` [dB/cm] — from `Im(n_eff)`

VπL endpoint values are kept as-is (less accurate but **not** masked to NaN).

### Cost model
- One charge simulation per `mult` (sweeps all 9 voltages internally).
- One mode-solver batch per `mult` (9 simulations, one per voltage).
- One Laplace solver call per `mult × voltage` for resistance — local CPU only,
  free.
- **Cache:** every heavy artifact is written to `cache/` keyed deterministically
  on `mult`. Filenames: `mult_XX_YYY_charge.pkl`, `mult_XX_YYY_modes.pkl`. The
  code checks existence before submitting; deleting the cache forces re-run.

### Adaptive strategy (Option A: bracket-and-fill)
1. Evaluate the four anchors {0.2, 1.0, 5.0, 20.0}.
2. Compute the (VπL, C) Pareto frontier from the journal (interior voltages
   only).
3. Insert next `mult` at the geometric midpoint of the largest gap on the
   frontier.
4. Repeat until the budget is spent.

### Budget
**10 full runs** total (= up to 10 mults × 7 interior voltages = 70 trade-off
points).

### Trade-off plots produced
Primary:
- `tradeoff_VpiL_C.png` — scatter, **color-coded by `mult`** (log-normalized
  viridis), **marker style by `target_v`** (one of 7 marker shapes), Pareto
  frontier overlaid.

Secondary:
- `tradeoff_VpiL_loss.png` — same color/marker scheme.
- `tradeoff_C_bandwidth.png` — same.
- `mult_sweep.png` — 2×2 grid of FoM vs `mult` at `target_v ≈ 0`.

### Journal schema (`journal.jsonl`)
One line per `(mult, voltage)`. Fields per row:
- `timestamp`, `run_id`, `mult`
- `p_doping_cm3`, `n_doping_cm3`, `target_v`
- `C_pF_per_cm`, `R_ohm_cm` (voltage-specific), `f3dB_GHz`
- `VpiL_V_cm`, `loss_dB_per_cm`
- `n_eff_re`, `n_eff_im` (baseline, for reference)
- `charge_cache`, `mode_batch_dir` (paths for traceability)
- `notes`

All 9 voltages are written (endpoints included for completeness; flag column
`is_interior` distinguishes the 7 trade-off points).

### Step 1 status
- Files created and parse-checked.
- User verified `mult=1.0` runs successfully end-to-end on Windows (v1).
- API key & vtk installed manually by user.
- **v2 changes (current):**
  1. R is now **voltage-dependent**: `_series_resistance_at_voltage(p, n, V)`
     computes a separate Laplace-based R for each of the 9 voltages, with the
     depletion edges recomputed from `V_total = V_bi + V` (clipped to
     `0.05*V_bi` on the forward-bias side).  Cached at `cache/<label>_R_sweep.pkl`.
  2. `DesignResult.R_ohm_cm` is now a (9,) array, and `f3dB_GHz[i]` uses
     `R[i]*C[i]` rather than scalar R.
  3. Mode-solver mesh-refinement boxes use the **bias=0** depletion edges as
     a single representative position (these are mesh hints, not physics).
  4. Journal writes **all 9 voltages** with an `is_interior` flag; endpoints
     are kept (VpiL is computed for all 9 with no NaN masking).
  5. Plot styling: scatter plots use **color = log10(mult)** with a colorbar,
     and **marker shape = target_v** with a marker legend.  Pareto frontier
     uses interior points only.
  6. `plot_tradeoffs.py` plots interior-only by default; pass
     `--include-endpoints` to show the endpoint points too.
  7. `journal.jsonl` rows now include: `is_interior`, `R_ohm_cm` (per voltage),
     `x_p_um`, `x_n_um`.

## Step 2 (preview, not started)
Take the (VπL, C) Pareto frontier from Step 1, pick an operating point,
and design the traveling-wave electrode (CPW dimensions, signal/ground
geometry, microwave loss, impedance match, velocity match) for that point.
Different control variables (`signal_width`, `gap_width`, `ground_width`,
electrode thickness) and different solver (RF S-parameter sim).

---
*Last updated: file regenerated alongside the v2 code with voltage-dependent R.*
