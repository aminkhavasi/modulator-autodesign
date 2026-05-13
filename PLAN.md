# Modulator Auto-Design — Plan

This is the source-of-truth planning document. It describes **what** the
agent should accomplish and **how**, but does NOT claim what has already
been done. The agent must check the actual state of `journal.jsonl` and
`step2_journal.jsonl` to find out where things stand.

## How to start (read this first, agent)

Before doing anything, in this order:

1. Read this whole file (`PLAN.md`).
2. Read `CLAUDE.md` (your operating rules).
3. Run these diagnostic commands to learn the current state. They are all
   safe — none submits a Tidy3D job:
   ```
   python run_sweep.py status
   python -m step2.run_batch overview
   ```
   The first prints what's in `journal.jsonl` (Step 1 progress). The second
   prints what's in `step2_journal.jsonl` (Step 2 progress) and reads
   `step2_targets.json` if it exists.

4. Then summarize to the user (in your terminal output, before starting):
   - What you understood from this PLAN and CLAUDE.md.
   - What the diagnostic commands revealed about prior progress.
   - Your plan for the autonomous run.
5. Begin the autonomous run. The user will not babysit; you proceed
   automatically through the workflow, stopping only at the conditions
   defined in `CLAUDE.md` (the 200-FDTD project gate, or the "stop and ask"
   safety brakes).

Important: **caches in `cache/` and `cache_step2/` may exist from prior
work.** If a command would re-submit an identical simulation, the cache
will hit and produce results at zero Tidy3D cost. Do not delete caches.

## Project Overview

LLM-driven design loop for silicon photonic PN-junction Mach-Zehnder
modulators on the SiEPIC EBeam platform at 1.55 µm. Uses Tidy3D
(charge solver + mode solver + RF FDTD) and PhotonForge (geometry,
technology), wrapped in a `Design → Verify → Simulate → Keep` agent loop
driven by Claude Code.

The full study is in two steps:

- **Step 1** characterizes the PN-junction itself across doping and bias
  (a charge sim and a mode-solver sweep per design point) to produce a
  family of (VπL, C, R, loss) trade-off points.
- **Step 2** optimizes a segmented coplanar-strip (CPS) microwave
  electrode for each of 10 junction operating points sampled from
  Step-1's lower envelope, then computes the EO 3-dB bandwidth for
  each. Result: a bandwidth-vs-junction-capacitance trade-off curve.

## Environment

- Windows 10/11, PowerShell, VS Code with integrated terminal.
- Python 3.10 in `.venv\` inside the project folder.
- Tidy3D 2.11.1 (Flexcompute API key already configured by user).
- PhotonForge 1.4.0 from prebuilt wheel
  (`wheels\photonforge-1.4.0-cp310-cp310-win_amd64.whl[live_viewer]`).
- siepic_forge installed separately.
- vtk installed manually by user (required by Tidy3D charge solver
  visualization).
- `scikit-optimize` installed for Step-2 BO.
- Git tracked locally; remote is `github.com/aminkhavasi/modulator-autodesign`.
- Large/regenerable artifacts (`.venv\`, `cache\`, `cache_step2\`,
  `wheels\*.whl`, `field_plots\*.png`, `*.hdf5`) excluded via `.gitignore`.

## Project Layout

```
modulator-autodesign/
├── .venv/                          # local Python env, gitignored
├── cache/                          # Step 1 charge / mode-solver caches
├── cache_step2/                    # Step 2 FDTD result caches
├── field_plots/                    # PNG outputs (gitignored)
├── wheels/                         # local wheel cache (gitignored)
├── notebooks/                      # reference notebooks (benchmarked)
│   ├── TWModulator_VpiL_Loss.ipynb
│   └── CPS_Modulator_tempversion.ipynb
├── .gitignore
├── README.md                       # install instructions
├── PLAN.md                         # this file
├── CLAUDE.md                       # agent operating rules
├── requirements.txt                # top-level deps
├── requirements-lock.txt           # full pip freeze
├── laplace.py                      # 2D Laplace solver (user-authored)
├── pn_junction.py                  # Step 1 physics: evaluate_design(mult)
├── run_sweep.py                    # Step 1 orchestrator + bracket-and-fill
├── plot_tradeoffs.py               # Step 1 plots from journal.jsonl
├── mzm_length.py                   # MZM length for target ER
├── step2/                          # Step 2 subpackage (12 modules)
│   ├── __init__.py
│   ├── fab_rules.py
│   ├── geom.py
│   ├── journal.py
│   ├── junction.py
│   ├── objective.py
│   ├── plot_step2.py
│   ├── propose.py
│   ├── review.py
│   ├── run_batch.py
│   ├── select_C_targets.py
│   └── simulate.py
├── journal.jsonl                   # Step 1 append-only log
├── step2_journal.jsonl             # Step 2 append-only log
└── step2_targets.json              # 10 (C, R, VpiL) targets for Step 2
```

## Step 1 — PN Junction Trade-off Sweep

### Goal
Characterize the PN junction across doping and bias to produce a (VπL, C)
trade-off plot that Step 2 will consume.

### Source notebook
`notebooks/TWModulator_VpiL_Loss.ipynb` — the benchmarked reference. The
physics in `pn_junction.py` is a faithful refactor; do not change the
mathematics.

### Geometry (FIXED)
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
9 reverse-bias values: `np.linspace(-0.5, 1.5, 9)`. The interior 7 are
the trade-off evaluation points; endpoints have less-accurate VπL gradients
but are still recorded with `is_interior=False` flag.

### Per-evaluation outputs (per mult, per voltage)
- `C` [pF/cm]    — capacitance per unit length (charge sim)
- `R` [Ω·cm]     — series resistance, **voltage-dependent** (depletion
                   width grows with reverse bias; recompute per V)
- `f3dB` [GHz]   — `1 / (2π R(V) C(V))`
- `VπL` [V·cm]   — `π / (dφ/dV per cm)` from mode-solver complex n_eff
- `loss` [dB/cm] — from `Im(n_eff)`

### Cost model
- One charge simulation per `mult` (sweeps all 9 voltages internally).
- One mode-solver batch per `mult` (9 sims, one per voltage).
- One Laplace solver call per `(mult, voltage)` for resistance — local CPU,
  free.
- All heavy artifacts cached at `cache/<label>_*.pkl` keyed on `mult`. The
  code checks existence before submitting.

### Adaptive strategy (bracket-and-fill)
1. Evaluate the four anchors {0.2, 1.0, 5.0, 20.0}.
2. Compute the (VπL, C) Pareto frontier from interior voltages only.
3. Insert next `mult` at the geometric midpoint of the largest gap on the
   frontier.
4. Repeat until budget is spent.

### Budget
**10 full runs** total = up to 10 mults × 7 interior voltages = 70
trade-off points. The CLI is `python run_sweep.py sweep --budget 10`.

### Trade-off plots produced
- `tradeoff_VpiL_C.png` — primary scatter, color = log10(mult), marker =
  target_v, Pareto frontier overlaid.
- `tradeoff_VpiL_loss.png`, `tradeoff_C_bandwidth.png`, `mult_sweep.png`.

Generate with `python plot_tradeoffs.py`.

### Journal schema (`journal.jsonl`)
One row per `(mult, voltage)`. Fields: `timestamp`, `run_id`, `mult`,
`p_doping_cm3`, `n_doping_cm3`, `target_v`, `is_interior`, `C_pF_per_cm`,
`R_ohm_cm` (per voltage), `f3dB_GHz`, `VpiL_V_cm`, `loss_dB_per_cm`,
`x_p_um`, `x_n_um`, `n_eff_re`, `n_eff_im`, `charge_cache`,
`mode_batch_dir`, `notes`.

## Step 2 — Segmented CPS Optimization with C-vs-Bandwidth Sweep

### Goal
For 10 linearly-spaced junction capacitance targets spanning Step-1's C
range, *independently* optimize the segmented CPS T-rail geometry to drive
the **loaded** characteristic impedance toward 50 Ω and the **loaded**
RF effective index toward `n_group_opt = 3.88` (matched to the optical
group index). Then, with the best CPS at each C, compute the EO 3-dB
bandwidth for an MZM whose length comes from the 5 dB ER spec at quadrature
with V_pp = 2 V (push-pull). Result: bandwidth-vs-C trade-off curve
identifying the operating point with highest bandwidth.

### Source notebook
`notebooks/CPS_Modulator_tempversion.ipynb` — the benchmarked reference
for the FDTD pipeline, de-embedding, and EO S21. The de-embedding logic
in `step2/simulate.py` is a faithful port; preserve the math.

### Key design decisions
- **Junction-loaded objective.** The objective evaluates `Z0` and `n_eff_rf`
  *after* loading by `y_junction(ω) = 1 / (R_eff + 1/(jωC_eff))` with
  series push-pull factor `R_eff = 2 R_pn`, `C_eff = C_pn / 2`.
- **Objective scalar:**
  ```
  J = ((Re Z0_loaded(f0) − 50) / 50)²
    + ((n_eff_rf_loaded(f0) − n_group) / n_group)²
  ```
  evaluated at `f0 = 25 GHz` (band center).
- **C-target selection rule:** for each linearly-spaced target C, pick
  the Step-1 row with **lowest VπL** whose C is within ±10% of the target.
  Tie-break: closest C, then min VπL. Captures the lower envelope of
  Step-1's (VπL, C) Pareto frontier.
- **Inner loop per C:** 8 LHS samples → agent_notes → 4 BO samples →
  agent_notes → repeat. Soft cap = 20 runs/C; hard cap = 40 (`--allow-hard`
  flag, only with explicit user authorization).
- **Outer loop:** 10 fully-independent C-target optimizations.
- **L_parasitic = 0** in EO model (clean comparison).
- **8 free CPS parameters:** g, ws, wg, s, r, h, t, c. Constraint
  `2(s+h) + 1 µm ≤ g` enforced with auto-repair (bumps g rather than
  rejecting).
- **Failure handling:** auto-retry once with ±2% perturbation; persistent
  failure logs the diagnostic and BO continues without that point.
- **Pacing:** autonomous. Each `run_batch.py` invocation does ONE batch
  and stops with review notes in the journal. The agent then proceeds to
  the next batch automatically (no user gate per batch). The agent stops
  only at the 200-FDTD project-wide gate or one of the safety conditions
  in `CLAUDE.md`.

### Step 2 file layout
```
step2/
├── fab_rules.py      # Min/max feature sizes, feasibility, repair
├── geom.py           # CPSGeometry dataclass, build_structures()
├── simulate.py       # FDTD wrapper with caching, batched submission,
│                     # de-embed, sanity check.  Constant-length segmented
│                     # section (~1000 µm), length-scaled run_time.
├── junction.py       # y_junction, loaded line, EO S21, BW, MZM length
├── objective.py      # Loaded-line scalar at f0
├── propose.py        # propose_lhs(), propose_bo() (skopt + EI), perturb()
├── journal.py        # step2_journal.jsonl I/O incl. agent_notes
├── review.py         # Per-batch review report (text/Markdown)
├── select_C_targets.py # Picks 10 (C, R, VpiL) targets from journal.jsonl
├── run_batch.py      # CLI: lhs / bo / review / overview / bandwidth_sweep
└── plot_step2.py     # Plots: BW-vs-C, BO convergence, EO S21
```

### Soft / hard budgets
- Per-C soft cap: 20 runs. Per-C hard cap: 40 (`--allow-hard`).
- Across 10 C targets: soft total = 200, hard total = 400. **Track Tidy3D
  credit usage at every checkpoint.**

### CLI workflow
```
# 0. Lock in C targets from Step-1 journal
python -m step2.select_C_targets

# 1. For each c_target_index in 0..9 (sequentially, autonomous):
python -m step2.run_batch lhs --c-target 0 --n 8
# Append agent_notes to step2_journal.jsonl analyzing the LHS results,
# then proceed automatically.
python -m step2.run_batch bo  --c-target 0 --n 4
# Same: agent_notes, proceed.
python -m step2.run_batch bo  --c-target 0 --n 4
# Same.
python -m step2.run_batch bo  --c-target 0 --n 4
# c_target=0 now at 20 runs (soft cap). Move to c_target=1
# automatically. Repeat.

# 2. Status across all targets at any time
python -m step2.run_batch overview

# 3. Final bandwidth sweep (no FDTD; uses cached results)
python -m step2.run_batch bandwidth_sweep

# 4. Plots
python -m step2.plot_step2 --all
```

### Performance notes (already in code, do NOT remove)
1. **Constant-length segmented section.** `NUM_UNITS` is derived per
   geometry as `max(8, round(1000 / period))`, so `L_segmented ≈ 1000 µm`.
   Bounds simulation cost across the parameter space.
2. **Length-scaled run_time.** `run_time = max(0.5 ns, 30 round-trips at
   n_eff=4)`.
3. **Batched submission.** `evaluate_cps_batch(geoms)` submits all
   uncached designs as one `web.Batch`. Cloud-side, FDTDs run in parallel.
   Wall-clock for an 8-LHS batch ≈ max(individual times), not sum.
4. **Batch-aware retry.** Failures collected after the main batch,
   perturbed ±2%, re-submitted as one retry batch. Both original failure
   and retry journaled.
5. **Symmetry trick deferred.** A previous attempt to halve cost via
   `run_only` + `element_mappings` hit pydantic validation errors against
   the post-2.10 RF API. Reverted to running both ports. Worth revisiting
   later with a working tidy3d.rf example to copy from.

### Journal schema (`step2_journal.jsonl`)
Two row kinds:
- **Evaluation rows:** `c_target_index, batch_id, geometry, geometry_hash,
  objective, Z0_re_f0, n_eff_f0, target_n_group, target_C_pF_per_cm,
  junction_C_pF_per_cm, junction_R_ohm_cm, junction_VpiL_V_cm, failed,
  failure_reasons, wall_time_s, timestamp`
- **Meta rows (`meta=true`):** `c_target_index, batch_id, agent_notes` —
  the agent's free-text analysis after a batch (Markdown).

## Step 3 — c_target=9 bandwidth refinement (added 2026-05-12)

### Why this phase exists

After Step 2 completed all 10 c_targets with the J objective
`J = ((Z0-50)/50)^2 + ((n_eff-3.88)/3.88)^2`, we asked whether relaxing
fab-rule minimums (s, r, h, c → 1 μm) could improve the heaviest-loading
design (c_target=9, C=16.47 pF/cm). The answer surfaced a much more
important problem: **for the short modulator length implied by the high
loading (L ≈ 282 μm), velocity walk-off is negligible. J's n_eff penalty
therefore pulled BO toward LOW Z0 (further from 50 Ω), which is the
wrong direction for BW.** Re-ranking all 32 c_target=9 designs by the
actual analytic 3-dB EO bandwidth (`analyze_bw.py`) showed:

- Correlation BW vs Z0: **+0.82** (higher Z0 → higher BW).
- Best-BW design (27.05 GHz) was J's *worst* design (J=2.05).
- J-incumbent (J=0.53) gives only 22.7–23.9 GHz.

So the actual c_target=9 work is BW maximization, and J was misaligned
with that goal at this point on the C trade-off curve.

### Objective

`step2/objective_bw.py` returns -BW (skopt minimizes) using the same
`bandwidth_3dB_GHz` pipeline as `bandwidth_sweep`. Length is fixed by
the same ER=5 dB, Vpp=2 V, push-pull recipe.

### Workflow

```
python -m step2.run_batch bo --c-target 9 --n 4 --rules relaxed \
       --budget-override --objective BW
```

- BO history is rebuilt: each prior c_target=9 row's BW is computed from
  its cached CPSResult (free) and fed to skopt as (-BW).
- Batches tagged `bw_bo_N` to distinguish from the J-phase batches.
- The journal stores BW_GHz and MZM_length_um alongside the existing
  Z0_re_f0, n_eff_f0 columns.

### Budget

- Up to 20 additional FDTDs at c_target=9 with the BW objective.
- Stop and ask the user before exceeding 20, regardless of best-so-far
  trajectory.

### What we DON'T do

- We do NOT retroactively change c_targets 0–8. Their J-phase result
  stands. The BW pivot only applies to c_target=9 where the misalignment
  is largest (heaviest loading, shortest modulator).
- We do NOT delete or rewrite the existing J-phase journal entries.
  They are preserved as historical evidence.

## Step 4 — BW reoptimization across all 10 c_targets (added 2026-05-12)

### Why this phase exists

Step 3 fixed c_target=9 with the new BW objective and got 23.9 → 28.16 GHz
(+18 %). Reviewing the other 9 c_targets showed the same J-vs-BW
misalignment, just with smaller absolute gaps because at lighter loading
J was less wrong.  User authorized a full reoptimization across all 10
c_targets with the BW objective, plus relaxing CPS-rail bounds to a 5 μm
minimum (RULES_V2).

### Key learnings to lock in (do not re-litigate these)

1. **Higher Z₀ is always better for BW at all 10 c_targets**, even at the
   long-modulator light-loading end where conventional CPS design says
   "match Z₀ to 50 Ω".  The reason is that the modulators are short
   enough that velocity walk-off contributes negligibly to the BW
   integral; only Z₀-mismatch reflection matters.  We empirically
   verified this by deliberately submitting low-Z₀ candidates with g=9,
   g=10, g=20 and finding they all gave BW = 17-29 GHz — strictly worse
   than the high-Z₀ corner.
2. **`g` should be at g_max (200 μm) for every c_target.** Sub-200 g
   candidates we tested were strictly worse.
3. **The optimal `ws=wg` shrinks with heavier loading.** c=0-2 prefer
   `ws ≈ 20`; c=3-6 prefer `ws ≈ 10`; c=7-9 prefer `ws ≈ 5`. RULES_V2's
   relaxation of ws_min to 5 was load-bearing for the heavy-loading
   c_targets.
4. **Cross-evaluation across c_targets is free.** A single CPS FDTD
   gives 10 data points (one per c_target's junction). Always
   cross-evaluate after every new batch.

### Workflow used

```
# Step 0: cross-evaluate all cached geometries against all c_target
# junctions (no FDTD cost):
python cross_eval_bw.py

# Round 1: 8 hand-picked candidates spanning the V2 corners (high-Z0
# AND low-Z0 to verify the hypothesis):
python bw_round1.py

# Round 2: refine the winning high-Z0 corner with finer ws scan:
python bw_round2.py

# (More rounds if the trajectory still shows uplift; otherwise stop.)
```

### Budget

- 200 FDTDs total authorized for the entire BW reoptimization phase.
- Used: 4 (Step 3 hand-picked) + 8 (R1) + 8 (R2) = **20 FDTDs**.
- Stopped: marginal BW gain per FDTD dropped below 0.2 GHz in R2.

### Outcome (see step4_bw_summary.json, field_plots/step4_BW_vs_C_pivot.png)

| c | C pF/cm | J-phase BW | BW-phase BW | uplift GHz |
|---|---|---|---|---|
| 0 | 2.92 | 38.25 | 39.89 | +1.6 |
| 1 | 4.01 | 35.96 | 39.42 | +3.5 |
| 2 | 6.27 | 33.84 | 39.86 | +6.0 |
| 3 | 7.62 | 26.11 | 36.82 | +10.7 |
| 4 | 9.02 | 25.77 | 35.35 | +9.6 |
| 5 | 10.35 | 24.21 | 34.55 | +10.3 |
| 6 | 12.11 | 23.88 | 32.57 | +8.7 |
| 7,8 | 14.07 | 21.94 | 31.04 | +9.1 |
| 9 | 16.47 | 21.79 | 28.99 | +7.2 |

### Geometry pattern that emerged

- `g = g_max = 200 μm` is optimal across all c_targets (max loop).
- Optimal `ws = wg` *shrinks* with junction loading: c=0-2 wants 20-192;
  c=4-5 wants 10; c=6-8 wants 5 (V2 rule load-bearing); c=9 also wants 5.
- T-bar density *bifurcates*: light/mid loading prefers sparse T
  (r=80, c=10); very-heavy c=9 prefers fine dense T (r=c≈1.5).

### What we DON'T do

- We do NOT run BO per-c_target separately.  Cross-evaluation means
  every FDTD informs every c_target; hand-picked design-of-experiments
  batches are more efficient.

## Pending Additions (after Step 2 is complete)

### Live dashboard / progress viewer
Single-page web view that auto-refreshes and visualizes:
- Cross-target overview: bar chart of `len(history)` and `best_objective`
  per c_target_index.
- For the active c_target: best-so-far trajectory, parameter-vs-objective
  scatter, currently-running batch status.
- Loaded-line plots (Z0(f), n_eff_rf(f), alpha(f)) for current best at
  each C target.
- For the bandwidth-sweep step: BW-vs-C curve as it fills in.

Implementation: probably **Streamlit** (single-file, file-watcher refresh).
Read `step2_journal.jsonl` directly; no synchronization issues with the
batch runner.


### End-of-project blog post
After the final bandwidth-vs-C result, write `BLOG_POST.md` at the project
root. ~1500-2500 words, technical but accessible to a competent EE/photonics
engineer.

Required sections:
1. **Problem framing.** SOI PN-junction MZM at 1.55 µm; why bandwidth-vs-C
   is the operationally relevant trade.
2. **The two-step approach.** Step 1's PN-junction characterization;
   Step 2's CPS optimization with junction-loaded objective.
3. **The integration story.** Tidy3D Charge + PhotonForge mode solver +
   Tidy3D RF FDTD + analytic post-processing, all coordinated by the LLM
   agent. **No single tool does the whole pipeline; the integration *is*
   the value.**
4. **Cost economy.** How LHS+BO+caching+batching kept FDTD count tractable.
5. **The human role.** 
  The engineer design the process, makes methodological choices and creates the workflow and reviews final output:
6. **Lessons learned and limitations.**
7. **Forward-looking.** Other devices this same workflow could tackle.

Defer until the full Step-2 sweep + bandwidth comparison is complete.
