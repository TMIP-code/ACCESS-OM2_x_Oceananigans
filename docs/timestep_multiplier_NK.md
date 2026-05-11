# Timestep multiplier — Newton-Krylov test (`TIMESTEP_MULT`)

The stability sweep in [docs/timestep_multiplier.md](timestep_multiplier.md)
established that the 1-year forward map is stable at OM2-1 `M ∈ {1, 2, 4}`
and that the resulting age fields agree to ≤ 23 ms RMS whole-domain
(≤ 88 ms in the surface layer). The 1-year forward map is the inner
operation of the periodic Newton-Krylov solve, so a stable forward map
is a **necessary** but not sufficient condition for the NK solve to
behave well at larger `M`.

This doc covers the **sufficient** test: run the full NK pipeline at
`M = 1`, `M = 2`, `M = 4` (OM2-1) and compare:

- **Convergence**: number of Newton iterations and cumulative GMRES
  iterations to reach a fixed residual tolerance. May differ across `M`
  even with the TM warm start because the surface-eigenvalue spectrum
  of the JVP changes with `Δt`.
- **Periodic age field**: volume-weighted RMS difference vs `M = 1`
  on the steady periodic solution (whole-domain + surface-layer), and
  divergence concentration.
- **Wall time**: TMbuild + TMsolve + NK end-to-end. This is the
  speedup that motivates the whole exercise — the NK inner loop has
  no per-step output I/O, so a ~`M`× speedup on the JVP should pass
  through cleanly.

## Pipeline

The NK steady-state solve depends on a transport matrix `M.jld2` and a
warm-start `steady_age_full_*.jld2`. The driver DAG resolves these:

```
TMbuild → TMsolve → NK → run1yrNK → plotNK
                  ↘   plotNKtrace
```

- **TMbuild** (CPU, `scripts/preprocessing/build_TMconst.sh`): builds
  the Jacobian `M.jld2` from the yearly time-averaged velocity. Δt
  affects only the surface diagonal (`−1/(3·Δt)`); sparsity and
  coloring are bitwise identical across `M`, so build cost is
  independent of `M`.
- **TMsolve** (CPU, `scripts/solvers/solve_TM_age_CPU.sh`): direct-
  solves `M · age = source` to produce `steady_age_full_*.jld2`, the
  warm start the NK solver loads when `INITIAL_AGE=TMage`.
- **NK** (GPU, `scripts/solvers/solve_periodic_NK.sh`): Newton-GMRES
  on the periodic problem `Φ(age) − age = 0` where `Φ` is the 1-year
  forward map. Writes
  `outputs/{PM}/{EXP}/{TW}/periodic/{MC}_DTx{M}/NK/age_{LINEAR_SOLVER}_{precond_tag}.jld2`.
- **run1yrNK** (GPU): forward-integrates one year from the periodic
  solution to produce `age_periodic_1year.jld2` for plotting.
- **plotNK** / **plotNKtrace**: diagnostic plots (see [Helper scripts](#helper-scripts)).

## Running it

One `driver.sh` invocation per `M`. The pipeline runs on `TM_SOURCE=const`
(yearly-averaged matrix, no `run1yr` dependency) — the path that the
stability doc already validated.

```bash
for M in 1 2 4; do
  PARENT_MODEL=ACCESS-OM2-1 \
  TIMESTEP_MULT=$M \
  TM_SOURCE=const \
  INITIAL_AGE=TMage \
  JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK-plotNK-plotNKtrace \
    bash scripts/driver.sh
done
```

Notes:

- `INITIAL_AGE=TMage` is the default; spelled out here so the warm-
  start path is explicit. Setting `INITIAL_AGE=0` would force a cold
  start and probably blow up the Newton iteration count.
- `TM_SOURCE=const` selects the yearly-averaged matrix branch. The
  `avg` branch would chain after a `run1yr` and a `TMsnapshot` step,
  which we don't need here.
- The driver propagates `TIMESTEP_MULT` to every PBS job through
  [scripts/driver.sh:245](../scripts/driver.sh#L245) (`COMMON_VARS`).
- If you want the per-Newton-iteration residual trace plotted by
  `plotNKtrace`, also set `TRACE_SOLVER_HISTORY=yes` — without it the
  NK solver still converges but doesn't write the iter-level JLD2 files
  that `plot_trace_history.jl` reads.
- Optional: `LUMP_AND_SPRAY=yes` enables the coarse-grid preconditioner
  (filename tag changes to `_LSprec` from `_prec`). The existing M=1
  precedent in
  `outputs/.../periodic/{MC}_mkappaV/NK/age_Pardiso_LSprec.jld2` was
  run with this on; keep it for cross-comparison.

## Verifying the right `M` was used

Three independent checks, in order from cheapest to most thorough:

### 1. Output path

The driver appends `_DTx{M}` to `MODEL_CONFIG` whenever `TIMESTEP_MULT > 1`
(see [scripts/env_defaults.sh:77-79](../scripts/env_defaults.sh#L77-L79)
and `build_model_config` in [src/shared_utils/config.jl:34-35](../src/shared_utils/config.jl#L34-L35)).
So the NK output for `M = 4` must land at:

```
outputs/{PM}/{EXP}/{TW}/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/NK/age_*.jld2
```

If your `M = 4` invocation wrote to a directory **without** `_DTx4`, the
flag didn't propagate. Check `qstat -xf <jobid> | grep TIMESTEP_MULT`
to see what reached the job.

### 2. Julia log line

Every script that calls `load_project_config()` prints:

```
[ Info: TIMESTEP_MULT    = 4  (Δt_base = 5400.0 s → Δt = 21600.0 s)
```

near the top of its log. For `M = 4` on OM2-1 you should see exactly
`Δt = 21600.0 s` (= 6 h). Search for `TIMESTEP_MULT` in:

```
logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/periodic/NK/*.log
logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/TM/*.log
```

### 3. Manifest TOML

`driver.sh` writes a manifest per submission that captures the full env
state, including `TIMESTEP_MULT`:

```bash
grep TIMESTEP_MULT outputs/{PM}/{EXP}/{TW}/manifests/*.toml | tail -3
```

Each line is one submission and reads `TIMESTEP_MULT = "<M>"`. The
manifest also carries the git commit (`commit = "..."`) so you can
prove exactly which code produced that run.

### Catch-all: validation in `load_project_config`

If `TIMESTEP_MULT` is not a divisor of `year/Δt_base`, the Julia code
errors before any compute starts. The error message lists the practical
divisors:

```
ERROR: TIMESTEP_MULT=5 is not a divisor of N_base=5844 (= year/Δt_base
for ACCESS-OM2-1). Valid multipliers ≤ 12 (Δt ≤ 18 h):
{1, 2, 3, 4, 6, 12}. Next valid value is 487 (= 1 month per step).
```

So a typo (e.g. `TIMESTEP_MULT=5`) never silently produces wrong
output — it aborts in [src/shared_utils/config.jl:117-135](../src/shared_utils/config.jl#L117-L135).

## Helper scripts

### Periodic age plots — `src/plot_periodic_1year_age.jl`

Submitted as `plotNK` step in the driver chain
([scripts/plotting/plot_1year_from_periodic_sol.sh](../scripts/plotting/plot_1year_from_periodic_sol.sh)).
Loads `outputs/.../periodic/{MC}_DTx{M}/1year/{LINEAR_SOLVER}_{precond}/age_periodic_1year.jld2`
and emits, per basin (global / Atlantic / Pacific / Indian) and at
depths 100 / 200 / 500 / 1000 / 2000 / 3000 m:

- Zonal-mean PNGs of the year-averaged age
- Zonal-mean MP4s animating the 1-year periodic cycle
- Horizontal-slice PNGs (year-averaged) and MP4s

Output lands in
`outputs/.../periodic/{MC}_DTx{M}/1year/{LINEAR_SOLVER}_{precond}/plots/`.

### Newton/GMRES residual history — `src/plot_trace_history.jl`

Submitted as `plotNKtrace`
([scripts/plotting/plot_trace_history_job.sh](../scripts/plotting/plot_trace_history_job.sh)).
Reads the per-iteration JLD2 trace files written by the NK solver when
`TRACE_SOLVER_HISTORY=yes` and produces residual-vs-iteration plots
(Newton outer-loop, GMRES inner-loop). Without the trace env flag the
NK solver still converges but writes no iter-level files, so this step
has nothing to plot.

### Sweep comparison — `src/plot_timestep_multiplier_sweep.jl`

Currently coded against the standard-run age files in
`{MC}_DTx{M}/standardrun/age_1year.jld2`. To reuse it for the NK
**steady periodic** comparison you would point it at
`{MC}_DTx{M}/periodic/.../age_periodic_1year.jld2` (year-mean) or the
final NK fixed-point `age_{LINEAR_SOLVER}_{precond}.jld2`. That's a
small follow-up — not wired up yet.

### Simulation wall time — `scripts/plotting/plot_simtime_vs_walltime.py`

For the run1yrNK step (which writes outputs every Δt), extract the
Julia-internal sim wall time the same way as the stability sweep:

```bash
python3 scripts/plotting/plot_simtime_vs_walltime.py --no-plot \
  logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/standardrun/*_DTx{M}*.log
```

(Parses both `run_1year_benchmark.jl` and regular `run_1year.jl` logs.)

The NK solver itself doesn't print a single end-of-run wall line; for
total NK time use the PBS-side `resources_used.walltime`
(reconciled into `scripts/runs/submissions.tsv` via
`scripts/runs/reconcile_submissions.sh`).

## Metrics to record

| Metric | Source | Notes |
|---|---|---|
| TMbuild wall (s) | PBS `walltime_used` | Should be ≈ constant across `M` (sparsity unchanged) |
| TMsolve wall (s) | PBS `walltime_used` | Linear-solve cost; should be ≈ constant across `M` |
| NK wall (s) | PBS `walltime_used` | Where the speedup lives (no per-step I/O) |
| run1yrNK wall (s) | `plot_simtime_vs_walltime.py` | Same script as stability sweep |
| End-to-end wall (s) | sum of the above | The speedup that motivates the exercise |
| Newton iterations | NK log | Count of `Newton iter` markers |
| Cumulative GMRES iters | NK log | Sum of per-Newton inner-loop counts |
| Final ‖Φ(age) − age‖ | NK log | At Newton convergence |
| Periodic age: mean / max / min (yr) | `plot_timestep_multiplier_sweep.jl`-style | After pointing at periodic outputs |
| RMS Δ vs M=1, whole / surface (yr) | same | Periodic-field comparison |
| Job IDs | `scripts/runs/submissions.tsv` | TMbuild + TMsolve + NK + run1yrNK |

## Results

### OM2-1 NK at `M ∈ {1, 2, 4}`

Stage labels match the stability sweep — only the M values that passed
Stage 2a there are run here.

| `M` | Δt | Status | TMbuild (s) | TMsolve (s) | NK wall (s) | Newton iters | GMRES iters | Final ‖res‖ | Job IDs (TMbuild / TMsolve / NK / run1yrNK) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.5 h | — | — | — | — | — | — | — | — |
| 2 | 3 h   | — | — | — | — | — | — | — | — |
| 4 | 6 h   | — | — | — | — | — | — | — | — |

#### Periodic age comparison (after run1yrNK + post-hoc)

| `M` | Mean age (yr) | Max age (yr) | Min age (yr) | RMS Δ whole (yr) | RMS Δ surf (yr) | max\|Δ\| (yr) |
|---|---|---|---|---|---|---|
| 1 | — | — | — | 0 | 0 | — |
| 2 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |

#### Diff plots vs M=1

Same pattern as the stability sweep — once the periodic comparison
script is wired up, embed the PNGs from
`outputs/.../periodic/{MC}_DTx{M}/diff_vs_DTx1_periodic/` in 2×2 (zonal
× 4 basins) and 2×3 (slices × 6 depths) tables here.

### Conclusions

TBD after the three NK runs complete and the comparison metrics are
filled in. The headline questions:

1. Does the Newton iteration count grow with `M`? If it grows
   sub-linearly, the cumulative GMRES iters are likely to stay flat
   enough that the NK wall time speeds up close to `M×`.
2. Does the periodic age field at `M = 4` agree with `M = 1` to
   within the stability-sweep tolerances (~25 ms RMS whole-domain,
   ~90 ms surface)? If yes, `M = 4` is the new default for the NK
   workflow.
3. End-to-end TMbuild → NK → run1yrNK wall time at `M = 4` vs
   `M = 1`. If the speedup is ≥ 3× the exercise paid for itself.
