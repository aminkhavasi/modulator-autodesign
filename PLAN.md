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

## Step 2 — Segmented CPS Optimization with C-vs-Bandwidth Sweep

### Goal
For 10 linearly-spaced junction capacitance targets spanning Step-1's C range,
*independently* optimize the segmented-CPS T-rail geometry to drive the
**loaded** characteristic impedance toward 50 Ω and the **loaded** RF effective
index toward the optical group index `n_group_opt = 3.88`. Then, with the best
CPS at each C, compute the EO 3-dB bandwidth for an MZM whose length comes from
the 5 dB ER spec at quadrature with V_pp = 2 V (push-pull). Result is a
bandwidth-vs-C trade-off curve that identifies the junction operating point
giving the highest bandwidth.

### Key design decisions
- **Junction-loaded objective.** The objective evaluates `Z0` and `n_eff_rf` *after*
  loading the CPS by `y_junction(ω) = 1 / (R_eff + 1/(jωC_eff))` with
  `R_eff = 2 R_pn`, `C_eff = C_pn / 2` (series push-pull factor).
- **Objective:** `J = ((Re Z0_loaded(f0)−50)/50)^2 + ((n_eff_rf_loaded(f0)−n_group)/n_group)^2`
  evaluated at f0 = 25 GHz (band center).
- **C-target selection rule:** for each linearly-spaced target C, pick the
  Step-1 row with **lowest VπL** whose C is within ±10% of the target.
- **Inner loop per C:** 8 LHS samples → review → 4 BO samples → review → repeat.
  Soft cap = 20 runs/C; hard cap = 40 (requires `--allow-hard`).
- **Outer loop:** 10 fully-independent C-target optimizations.
- **L_parasitic = 0** (clean comparison, notebook's empirical 5 pH not used).
- **8 free CPS parameters:** g, ws, wg, s, r, h, t, c. Constraint
  `2(s+h) + 1 µm ≤ g` enforced with auto-repair (bumps g rather than rejecting).
- **Failure handling:** auto-retry once with ±2% perturbation; persistent
  failure logged with diagnostic, objective set to ∞.
- **Pacing:** "agent in the loop" — `run_batch.py` does ONE batch per
  invocation and stops, writing review notes into the journal. Agent (or user)
  decides when to issue the next batch.

### Step 2 file layout (under `step2/` subpackage)
```
step2/
├── __init__.py
├── fab_rules.py        # Min/max feature sizes, feasibility check, repair
├── geom.py             # CPSGeometry dataclass, build_structures()
├── simulate.py         # FDTD wrapper with caching + de-embed + sanity check
├── junction.py         # y_junction, loaded line, EO S21, bandwidth, MZM length
├── objective.py        # Loaded-line scalar objective at f0
├── propose.py          # propose_lhs(), propose_bo() (skopt + EI), perturb()
├── journal.py          # step2_journal.jsonl I/O incl. agent_notes
├── review.py           # Per-batch review report (text/Markdown)
├── select_C_targets.py # Picks 10 (C, R, VpiL) targets from Step-1 journal
├── run_batch.py        # CLI: lhs / bo / review / overview / bandwidth_sweep
└── plot_step2.py       # Plots: BW-vs-C, BO convergence, EO S21
```
Plus at project root:
```
mzm_length.py           # Fixed version (np.arcsin, push-pull factor explicit)
```

### Soft / hard budgets
- Per-C soft cap: 20 runs. Per-C hard cap: 40 (with `--allow-hard`).
- Across 10 C targets, soft total = 200, hard total = 400. **Track Tidy3D
  credit usage.**

### CLI workflow
```
# 0. Lock in C targets from Step-1 journal
python -m step2.select_C_targets

# 1. For each c_target_index in 0..9:
python -m step2.run_batch lhs --c-target 0 --n 8
python -m step2.run_batch bo  --c-target 0 --n 4
python -m step2.run_batch bo  --c-target 0 --n 4
python -m step2.run_batch bo  --c-target 0 --n 4

# 2. Status across all targets
python -m step2.run_batch overview

# 3. Final bandwidth sweep (no FDTD; uses cached results)
python -m step2.run_batch bandwidth_sweep

# 4. Plots
python -m step2.plot_step2 --all
```

### New dependencies
- `scikit-optimize >= 0.10`
- `scikit-learn >= 1.3` (skopt transitive)
- `scipy >= 1.7` (already present; for `qmc.LatinHypercube`)

Add to `requirements.txt`; refresh `requirements-lock.txt`.

### Step 2 status
- All 12 modules created and parse-checked.
- Strategy logic (LHS, BO, fab repair, MZM length) tested without Tidy3D —
  numerical sanity checks pass.
- Dry run on `c_target=0` (mult=1.0 from Step 1) succeeded:
  Z0_re(f0)=48.4 Ω, n_eff_rf(f0)=3.676, objective=0.00376.
- **n=2 batched LHS milestone:** seed=99, c_target=0. Both designs
  successful. Wall clock ~14 min (vs ~28 min serial). Best of three runs
  so far (1 dry + 2 batched): objective=0.00030, Z0=50.50 Ω, n_eff_rf=3.825.

## Step 2 — Pending Additions

### Live dashboard / progress viewer
Build a single-page web view (or local HTML+JS) that auto-refreshes every
~10 s and visualizes:
- Cross-target overview: bar chart of `len(history)` and `best_objective`
  per c_target_index.
- For the active c_target: best-so-far trajectory, parameter-vs-objective
  scatter, currently-running batch status.
- Loaded-line plots (Z0(f), n_eff_rf(f), alpha(f)) for the current best
  design at each C target.
- For the bandwidth-sweep step: BW-vs-C curve as it fills in.

Implementation options:
- **Dash/Plotly + Flask** — pure Python, easy local hosting.
- **Streamlit** — even simpler, file-watcher refresh.
- **Static HTML+JS that polls journal.jsonl** — no Python server needed.

The dashboard reads `step2_journal.jsonl` directly (same source as
`run_batch.py status`), so no synchronization issues with the agent's
batch process.

Defer until after Step 1 is complete and Step 2 has 1-2 c_targets fully
optimized. Simpler to design once we know what numbers actually need
live monitoring.

### End-of-project blog post
After the final bandwidth-vs-C result is in, the agent writes a Markdown
blog post documenting the design study. Required content:
1. **Problem framing.** SOI PN-junction MZM design at 1.55 µm; why
   bandwidth-vs-C trade-off is the operationally relevant trade.
2. **The two-step approach.** Step 1's PN-junction characterization
   (Charge sim → R, C, VπL, loss); Step 2's CPS optimization with
   junction-loaded objective.
3. **The integration story.** Tidy3D Charge solver + PhotonForge mode
   solver + Tidy3D RF FDTD + analytic post-processing, all coordinated
   by the LLM agent. **Emphasize that no single tool does the whole
   pipeline; the integration *is* the value.**
4. **Cost economy.** How LHS+BO+caching+batching kept FDTD count to
   ~30-60 instead of brute-force ~10000.
5. **Where the human did the real work.** This is the section that
   matters most. The LLM did not "design the modulator." Amin (Flexcompute
   engineer) did, with the LLM as a coding/orchestration partner. Items
   to credit explicitly:
   - Identified push-pull series-pn convention and how it enters the
     loaded-line model (the `c/2`, `r*2` factor).
   - Caught a benchmarked-code regression (the gamma-feed unit
     conversion) that the LLM had wrongly "simplified."
   - Decided on the C-target selection rule ("for given C, pick min
     VπL") that made Step 2 a tractable trade-off study.
   - Picked the inner-loop algorithm (LHS + BO with reviews) and the
     budget pacing (soft 20 / hard 40 per C, manual checkpoints).
   - Specified the loaded-line objective evaluation (after junction
     loading, not before — a non-obvious but important distinction).
   - Insisted on segmented-line length normalization to bound cost.
   - Reviewed the symmetry-trick attempt and accepted the deferral
     when the post-2.10 RF API schema couldn't be confirmed without a
     working example.
6. **Lessons learned and limitations.** What the methodology can't do
   (e.g., fab-rule-aware geometry constraints currently use placeholders;
   the 2D depletion-edge model is approximate; the BO surrogate is 8D
   GP which may not generalize to higher-D problems).
7. **Forward-looking.** What other devices this same workflow could
   tackle (ring modulators, MMI couplers, polarization rotators).

The agent must NOT claim it "designed" the modulator. It coded, ran
tools, and reported. Amin set the plan, made every methodological
choice, and caught the bugs. The LLM is a productivity multiplier on
top of an expert designer's plan, not a replacement for one.

Output: `BLOG_POST.md` at the project root. Include 2-3 plots
(`tradeoff_VpiL_C.png`, `step2_BW_vs_C.png`, optionally an EO S21
overlay). Length: 1500-2500 words. Tone: technical but accessible to
a competent EE/photonics engineer who isn't necessarily a Tidy3D user.

Defer until the full sweep + bandwidth comparison is complete.

### v2 fixes (current iteration of step2/simulate.py and step2/geom.py)
Three production-grade fixes applied after the dry run revealed inefficiencies:

1. **Constant-length segmented section.** `NUM_UNITS` is now a derived
   property of each `CPSGeometry`: `max(8, round(1000 / period))`. So
   `L_segmented ≈ 1000 µm` regardless of `r` and `c`. Simulation cost is
   now bounded.

2. **Length-scaled run_time.** `run_time = max(0.5 ns, 30 round-trips at
   n_eff=4)`. Replaces the notebook's fixed 0.5 ns. Default is unchanged
   for short periods; longer-period designs get up to ~1.1 ns.

3. **Half-cost via reciprocity-symmetry — DEFERRED.** I attempted to use
   `run_only=("WP1", 0)` and `element_mappings` to declare S22=S11, S12=S21
   so only one excitation runs. Hit two pydantic validation errors against
   the post-2.10 `tidy3d.rf.TerminalComponentModeler` API (which differs
   from the old `tidy3d.plugins.smatrix.ModalComponentModeler` example
   in the docs). **Reverted to running both ports.** The cost is 2× per
   FDTD evaluation but robustness wins for now. Worth revisiting when the
   2.10 RF API stabilizes.

4. **Batched submission.** `evaluate_cps_batch(geoms)` submits all uncached
   designs as one `web.Batch` (mode-solvers and FDTDs as two separate
   batches). Cloud-side, the FDTDs run in parallel; wall-clock for an
   8-LHS batch becomes max(individual times) instead of sum. **Saves
   ~5-8x on wall clock.**
5. **Batch-aware retry.** Failures within a batch are collected, perturbed,
   and re-submitted as a small follow-up batch (≤2 batches per LHS/BO
   command). Both the original failure and the retry are journaled.
- **Pending user actions before launching:**
  1. Establish per-FDTD cost: run the existing CPS notebook once with default
     geometry on this account. Multiply by ~30-60 for realistic Step-2 cost,
     by 200 for worst-case soft-cap, by 400 for worst-case hard-cap.
  2. Adjust `step2/fab_rules.py` to your actual fab PDK rules (current
     defaults are placeholders).
  3. After Step 1's full sweep is done, run `python -m step2.select_C_targets`
     and review the 10 chosen targets.
  4. Start the inner loop with `c_target_index=0` and a single LHS batch:
     `python -m step2.run_batch lhs --c-target 0 --n 8`. Review the output
     before continuing.

---
*Last updated: file regenerated alongside the v2 code with voltage-dependent R.*
