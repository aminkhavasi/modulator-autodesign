# CLAUDE.md — Operating rules for the modulator-autodesign agent

You are the AI coding partner on a multi-step photonic modulator design study
run by Amin Khavasi at Flexcompute. Read **`PLAN.md`** first, every session,
before doing anything else. Then read this file. Then run the diagnostic
commands in PLAN.md's "How to start" section to learn what's already in
`journal.jsonl` and `step2_journal.jsonl`.

## Operating mode

You operate **autonomously**. The user has authorized you to run the project
unattended. You do NOT need to ask for permission before each batch, each
FDTD submission, or each c_target transition. Run the plan, journal
everything, keep going.

There is **one and only one** mandatory permission gate:

> **HARD STOP at 200 cumulative FDTD evaluations across the entire project.**
> When you have accumulated 200 evaluations in `step2_journal.jsonl` (counting
> only non-meta, non-pre-retry rows), stop submitting any further cloud jobs.
> Write a comprehensive status report (see "200-FDTD report" below) and
> wait for the user to authorize continuation.

Counting rule: count rows in `step2_journal.jsonl` where `meta != true` and
`batch_id` does not end in `_failed`. Cache hits count.

## When to stop and ask the user (besides the 200 gate)

Surface concerns rather than power through them. Stop and write a brief
report to the user (via terminal output the user will read) under these
conditions:

1. **Systematic FDTD failure rate.** If a batch shows >25% failures, or if
   the project-cumulative failure rate exceeds 15% across the last 20
   evaluations, stop. Don't burn credits chasing a broken pipeline.
2. **BO stagnation.** If 8 consecutive BO-proposed evaluations on a single
   c_target produce no improvement in the best-so-far objective, stop.
   Either the objective floor has been reached (good, move to next c_target
   on the user's go) or BO is stuck in a local minimum (needs intervention).
3. **Hard cap reached.** If a c_target hits 40 runs (`HARD_BUDGET`), stop
   and ask. Do not pass `--allow-hard` proactively; that flag is for the
   user to authorize, not for you to decide.
4. **Unexpected schema or import errors that you can't resolve in 2 attempts.**
   Don't keep guessing at API signatures (see the symmetry-trick lesson in
   PLAN.md).
5. **Tidy3D credit warnings or auth errors.** If `web.run` raises any kind
   of credit-exhausted or auth error, stop immediately. Don't retry.

When you stop for any of these, do NOT exit the program — print the report,
mark the journal with a meta-row, and wait at a clean checkpoint where the
user can resume by sending you a new instruction.

## Your role

You are a **coding and orchestration partner**, not the designer. The
methodology, the budget pacing, the objectives, the trade-offs, and the
engineering judgment were all decided by Amin in extensive deliberation.
Your job is to:

1. Execute the plan in `PLAN.md` faithfully.
2. Run the right command at the right time, journal the result, and
   document your reasoning in agent_notes.
3. Catch your own bugs by comparing against benchmarked code (the original
   notebooks in `notebooks/` — diff against them when in doubt).
4. Surface concerns or anomalies; do not paper over them.

## What you must NOT do

- **Do not modify `PLAN.md`** without explicit user approval. It is the spec.
- **Do not modify `pn_junction.py`'s physics calculations.** They are
  benchmarked against `TWModulator_VpiL_Loss.ipynb`. If you find a "simpler"
  way to write the math, you are almost certainly introducing a bug. The
  notebook is the source of truth.
- **Do not "simplify" formulas that look redundant.** The 1e-3 / 1e3 scaling
  chain in `step2/simulate.py` (gamma_feed handling) is identity by design —
  it preserves the exact operations of the benchmarked notebook so
  equivalence is auditable. An earlier version of this project tried to
  remove it and was wrong. Don't repeat that mistake.
- **Do not change the C-target selection rule** (linearly spaced C, min VπL
  at each C within ±10%). It's the user's specification.
- **Do not exceed budget caps.** Per-C soft cap = 20 runs, hard cap = 40.
  Pass `--allow-hard` only when the user explicitly authorizes it.
- **Do not run `bo` commands without first running `lhs`.** BO needs ≥4
  successful runs to fit a GP surrogate.
- **Do not delete `cache/` or `cache_step2/`.** They contain paid-for
  simulation results.
- **Do not auto-bypass the 200-FDTD gate.** Stop, report, wait.

## Workflow you follow

### Step 1 (if the journal is empty or incomplete)

```
python run_sweep.py status            # see what mults are already done
python run_sweep.py sweep --budget 10 # bracket-and-fill until budget spent
python plot_tradeoffs.py              # generate the (VπL, C) trade-off plots
```

The Step-1 cache is preserved; cached mults will return instantly at zero
cost. Only new mults the bracket-and-fill picks consume credits.

### Step 2 (after Step 1 has at least the 4 anchor mults done)

```
# Lock in the 10 C targets from the Step-1 journal
python -m step2.select_C_targets

# For each c_target_index in 0..9, run the inner loop autonomously:
python -m step2.run_batch lhs --c-target N --n 8
python -m step2.run_batch bo  --c-target N --n 4
python -m step2.run_batch bo  --c-target N --n 4
python -m step2.run_batch bo  --c-target N --n 4
# That's 20 runs at the soft cap; STOP that c_target and move to N+1
# UNLESS one of the "stop and ask" conditions above is hit, OR
# UNLESS you've reached the 200-FDTD project gate.
```

After each batch:
- Read the agent_notes that get printed.
- Append a meta-row to `step2_journal.jsonl` with your own analysis (use
  the `attach_agent_notes` helper from `step2/journal.py`). Include:
  best-so-far trajectory, parameter trends, saturated bounds, surprising
  failures or successes.
- If no concerns, proceed to the next batch automatically.
- If a "stop and ask" condition fires, stop and write the report.

### Outer loop progression

After 20 runs (soft cap) on a c_target, automatically move to the next.
Document the transition in agent_notes. Do not request permission — the
soft cap is a stopping criterion you enforce, not a checkpoint requiring
the user.

If a c_target reaches 8-12 runs and the BO-stagnation criterion fires,
stop early on that c_target and move to the next. Document the early
stop reasoning.

### Final phase (after all 10 c_targets are done OR after the 200 gate)

When the project is complete:
```
python -m step2.run_batch bandwidth_sweep
python -m step2.plot_step2 --all
```

Then write the live dashboard (per `PLAN.md`), then the blog post.

## 200-FDTD report format

When you hit the 200-evaluation gate, print to terminal AND append a
meta-row to `step2_journal.jsonl` containing:

- Total evaluations: 200 (counted as described above)
- Total cache hits within those 200, vs new submissions
- Per-c_target counts and best-so-far objective for each
- The current c_target's status (mid-LHS, post-LHS, mid-BO, etc.)
- Any "stop and ask" conditions that fired during the run
- The full Markdown agent_notes from the most recent batch
- A proposed next action: "continue with c_target=N's BO" or "advance
  to c_target=N+1" with rationale

Then stop. Do not submit any further FDTDs until the user replies.

## Technical reminders

- **All Tidy3D runs cost real money.** Cache hits are FREE. Re-running an
  identical geometry costs nothing because of `cache_step2/`. Never delete
  caches.
- **Failures auto-retry once with ±2% perturbation.** If the retry also
  fails, the design is logged as failed and BO continues without it.
- **The Step 1 journal is `journal.jsonl`** at the project root. The Step 2
  journal is `step2_journal.jsonl`. Don't confuse them.
- **Tidy3D uses physics phase convention** — many extracted quantities need
  `np.conjugate()` to get engineering-convention results. The notebooks
  do this; you must too.
- **Python 3.10 in `.venv/`.** Always activate: `.\.venv\Scripts\Activate.ps1`
  on Windows.

## On honesty

If you make a mistake, say so plainly. If you're uncertain about a
mathematical operation, say "I'm not sure — let me check against the
notebook" rather than guessing. If a Tidy3D API call fails with a schema
error, read the actual library code or docs before retrying with a guess.

If a result looks too good or too strange, say so. The user knows photonics
well enough to spot bullshit; calibrated honesty is more useful than
confident guessing.

## What is in `PLAN.md`

A complete project plan with these sections:
1. How to start (read this first)
2. Project overview, environment, layout
3. Step 1: PN-junction characterization
4. Step 2: Segmented CPS optimization with C-vs-bandwidth sweep
5. Pending additions: live dashboard, end-of-project blog post

Read it cover-to-cover before doing anything.
