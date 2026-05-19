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
- **Diff plots vs M=1**: same `age_M − age_1` zonal-average × 4 basins
  and horizontal-slice × 6 depths plots as the stability sweep
  produced — embedded in this doc as 2×2 / 2×3 image tables so the
  spatial structure of the divergence is directly visible. This is a
  required deliverable, not a follow-up.
- **Wall time**: TMbuild + TMsolve + NK end-to-end. This is the
  speedup that motivates the whole exercise — the NK inner loop has
  no per-step output I/O, so a ~`M`× speedup on the JVP should pass
  through cleanly.

## Pipeline

The NK steady-state solve depends on a transport matrix `M.jld2` and a
warm-start `steady_age_full_*.jld2`. The driver DAG resolves these:

```
TMbuild → TMsolve → NK → run1yrNK → plotNK
```

- **TMbuild** (CPU, `scripts/preprocessing/build_TMconst.sh`): builds
  the Jacobian `M.jld2` from the yearly time-averaged velocity. Δt
  affects only the surface diagonal (`−1/(3·Δt)`); sparsity and
  coloring are bitwise identical across `M`, so build cost is
  independent of `M`.
- **TMsolve** (CPU, `scripts/solvers/solve_TM_age_CPU.sh`): direct-
  solves `M · age = source` to produce `steady_age_full_*.jld2`, the
  warm start the NK solver loads when `INITIAL_AGE=TMage`. Per-`M`
  because the M=1 steady age is *not* equivalent to the M=4 steady
  age in the surface layer — the relaxation BC scales with Δt — and
  the design choice in
  [docs/timestep_multiplier.md](timestep_multiplier.md#design-choice-surface-relaxation-scales-with-δt)
  means we're solving a slightly different continuum operator at
  `M = M_max`. The TMsolve at each `M` produces the warm start that
  is consistent with that `M`'s NK problem.
- **NK** (GPU, `scripts/solvers/solve_periodic_NK.sh`): Newton-GMRES
  on the periodic problem `Φ(age) − age = 0` where `Φ` is the 1-year
  forward map. Writes
  `outputs/{PM}/{EXP}/{TW}/periodic/{MC}_DTx{M}/NK/age_{LINEAR_SOLVER}_{precond_tag}.jld2`.
- **run1yrNK** (GPU): forward-integrates one year from the periodic
  solution to produce `age_periodic_1year.jld2` for plotting.
- **plotNK**: diagnostic plots of the periodic 1-year run (see
  [Helper scripts](#helper-scripts)).

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
  JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK-plotNK \
    bash scripts/driver.sh
done
```

Notes:

- `INITIAL_AGE=TMage` is the default; spelled out here so the warm-
  start path is explicit. The NK solver looks under
  `outputs/.../TM/{MC}_DTx{M}/{TM_SOURCE}/steady_age_full_*.jld2`, so
  TMsolve must run at each `M` to produce a warm start consistent
  with that `M`'s relaxation rescaling. Setting `INITIAL_AGE=0` would
  force a cold start and probably blow up the Newton iteration count.
- `TM_SOURCE=const` selects the yearly-averaged matrix branch. The
  `avg` branch would chain after a `run1yr` and a `TMsnapshot` step,
  which we don't need here.
- The driver propagates `TIMESTEP_MULT` to every PBS job through
  [scripts/driver.sh:245](../scripts/driver.sh#L245) (`COMMON_VARS`).
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

### Sweep comparison — `src/plot_timestep_multiplier_sweep.jl`

Currently coded against the standard-run age files in
`{MC}_DTx{M}/standardrun/age_1year.jld2`. **Extend it (or fork a sibling
script)** to also read the NK outputs — point at either
`{MC}_DTx{M}/periodic/{precond_tag}/NK/age_{LINEAR_SOLVER}_{precond}.jld2`
(the final NK fixed-point) or the year-mean of
`{MC}_DTx{M}/periodic/.../1year/.../age_periodic_1year.jld2`, and route
the diff plots into a parallel `diff_vs_DTx1_periodic/` subdir so they
don't collide with the standardrun ones. The plot routine
([plot_age_diagnostics](../src/shared_utils/analysis_and_plotting.jl#L174))
is already generic over the input field — only the loader needs to
change.

A natural switch: `COMPARE_TARGET=standardrun|nk_steady|nk_periodic`
in the script, defaulting to `standardrun` for back-compat. The
ColorSchemes / OceanBasins imports (added in commit 4d1cd63) already
match the production pattern; no new imports needed.

The same 2×2 (zonal × 4 basins) + 2×3 (slices × 6 depths) Markdown
table layout used in
[docs/timestep_multiplier.md](timestep_multiplier.md#diff-plots-vs-m1)
should be reproduced under [Diff plots vs M=1](#diff-plots-vs-m1)
below once the PNGs land.

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
| TMbuild wall | PBS `walltime_used` | Should be ≈ constant across `M` (sparsity unchanged) |
| TMsolve wall | PBS `walltime_used` (CPU job) | Linear-solve cost; should be ≈ constant across `M` |
| NK wall | PBS `walltime_used` | Where the speedup lives (no per-step I/O) |
| run1yrNK wall | PBS `walltime_used` | Same step as stability sweep |
| End-to-end wall | sum of the above | The speedup that motivates the exercise |
| Φ! calls (total) | NK log | Count of `Φ! call # … starting` markers. Mirrors PROGRESS.md's `(N Φ)` convention — one number covering Newton G-evaluations + GMRES JVP-via-`linΦ` calls. |
| Final ‖G‖_∞ | NK log (`Final` line of NewtonRaphson trace) | Inf-norm of `G(x) = Φ(x) − x` at convergence (years). Solver `abstol = 0.0001 yr` is on the volume-weighted RMS norm, not on this inf-norm — so a large inf-norm with a Success retcode is expected. |
| Periodic age: mean / max / min (yr) | `plot_timestep_multiplier_sweep.jl` with `COMPARE_TARGET=nk_periodic` | Year-mean of the 25 half-monthly snapshots from run1yrNK. |
| RMS Δ vs M=1, whole / surface (yr) | same | Periodic-field comparison. |
| Job IDs | `scripts/runs/submissions.tsv` | TMbuild + TMsolve_c + NK + run1yrNK + plotNK |

## Results

### OM2-1 NK at `M ∈ {1, 2, 4}`

Stage labels match the stability sweep. The first three rows (`M ∈
{1, 2, 4}`) are the converged runs; the last two (`M ∈ {6, 12}`) are
follow-up sweep extrema, **both of which failed** — see the per-`M`
notes below. All runs used `LINEAR_SOLVER=Pardiso`,
`LUMP_AND_SPRAY=yes` (filename tag `Pardiso_LSprec`),
`JVP_METHOD=exact` (linear-tracer JVP), `INITIAL_AGE=TMage` (warm start
from the per-`M` TMsolve), and `TM_SOURCE=const`.

Wall times below are PBS `resources_used.walltime`. **Φ! calls** is the
total count of `Φ! call # … starting` markers in the NK log — the same
convention PROGRESS.md uses (`(N Φ)`). **Final ‖G‖_∞** is the
NewtonRaphson trace's `Final` line (inf-norm of `G = Φ(x) − x` in years).
**Status**: ✓ = `retcode = Success`, ✗ = solver did not converge.

| `M` | Δt | Status | TMbuild | TMsolve_c | NK wall | run1yrNK | plotNK | End-to-end | Φ! calls | Final ‖G‖_∞ (yr) | Job IDs (TMbuild / TMsolve_c / NK / run1yrNK / plotNK) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.5 h | ✓ | 08:25 | 03:29 | **51:59** | 06:16 | 08:32 | 78:41 | 62 | 4.06e+02 | 168091718 / 168091719 / 168091721 / 168091722 / 168091723 |
| 2 | 3 h   | ✓ | 06:38 | 02:52 | **30:12** | 05:43 | 07:54 | 53:19 | 62 | 4.18e+02 | 168093148 / 168093149 / 168093151 / 168093152 / 168093153 |
| 4 | 6 h   | ✓ | 06:16 | 03:15 | **19:52** | 05:51 | 09:16 | 44:30 | 62 | 1.10e+03 | 168093162 / 168093163 / 168093165 / 168093166 / 168093167 |
| 6 | 9 h   | ✗ (crash) | — | — | **18:30** | — | — | — | 16 (≤) | n/a | — / — / 168257380 / — / — |
| 12 | 18 h | ✗ (stalled) | — | — | **24:16** | — | — | — | 104 | 5.15e+78 | — / — / 168251280 / — / — |

**NK speedup** (between converged runs): M=2 → 1.72× vs M=1; M=4 → 2.61× vs M=1.
**Per-Φ-call speedup** (NK inner Julia time `2714 / 1465 / 894` s): 1.85× and 3.04× — close to the ideal `M×` at M=2, sub-linear at M=4 because fixed per-call overhead (init + first-step warmup + I/O) doesn't shrink with `M`.
**End-to-end speedup** (TMbuild → plotNK): M=4 / M=1 = 1.77× — diluted because TMbuild / TMsolve / run1yrNK / plotNK are roughly constant across `M`.
**Φ! calls flat at 62 across `M ∈ {1, 2, 4}`** — the JVP spectrum of the surface relaxation operator is not sensitive enough to Δt to change the Krylov inner-iteration count in this regime. Headline question 1 (does Newton count grow with `M`?) answers cleanly: **no, within the convergent regime.**

##### Failure modes

- **M=6 (Δt = 9 h)** — numerical instability in the forward map. Inside
  Φ! call #16, the simulation went unstable at `sim iter 574` of the
  1-year integration: `max(age)` jumped from 8.8 yr to **1.0×10⁴ yr** at
  `(i, j, k) = (331, 160, 39)` (surface tropical Pacific, one cell off
  from the converged `M ∈ {1, 2, 4}` max-`|Δ|` location `(224, 40, 50)`
  — same surface-relaxation regime). The next step crashed CUDA
  (`copyto!` failure inside `progress_message`, Julia exit code 271).
  This is the 1-year forward-map's stability wall, *not* an NK-solver
  failure — the linear stability sweep should be re-run with `M=6` on
  the *NK warm start* to confirm the start state is the trigger.
- **M=12 (Δt = 18 h)** — solver stalled. `total_G_calls = 104`,
  `retcode = ReturnCode.Stalled`. Final inf-norm of `G` blew up to
  `5.15×10⁷⁸` yr and the saved `age_steady_3D` field has volume-mean
  **−9.1×10⁻¹¹ yr** (effectively zero — the warm start was destroyed
  by the first GMRES Newton step). At this Δt the forward map is
  evidently far from the linearisation around the TMage warm start, so
  the Krylov direction is dominated by spurious modes.

Verdict for the OM2-1 base Δt: the practical convergent regime is
`M ∈ {1, 2, 4}`. **M=4 (6 h)** is the highest stable multiplier; the
9 h / 18 h forward maps are unstable / under-conditioned for this warm
start. Future work: try `INITIAL_AGE=0` cold start, `LUMP_AND_SPRAY=no`,
or a tighter GMRES inner tolerance to push the wall higher.

#### Periodic age comparison (after run1yrNK + post-hoc)

Two comparison targets are reported, both produced by
[plot_timestep_multiplier_sweep.jl](../src/plot_timestep_multiplier_sweep.jl)
via the new `COMPARE_TARGET` switch:

- **nk_steady** — the NK fixed point, `Φ(x*) = x*`, loaded directly
  from `periodic/{MC}_DTx{M}/NK/age_Pardiso_LSprec.jld2`. This is the
  state *at the start of the periodic year*.
- **nk_periodic** — the volume-weighted year-mean of the 25 half-
  monthly snapshots of `run1yrNK`, from
  `periodic/{MC}_DTx{M}/1year/Pardiso_LSprec/age_periodic_1year.jld2`.
  Same convention as
  [plot_periodic_1year_age.jl:159-165](../src/plot_periodic_1year_age.jl#L159-L165).

The two targets should be physically very close — and they are: RMS-Δ
values agree to ≤ 1% across both `M` values, and `max|Δ|` is at the same
`(i, j, k) = (224, 40, 50)` (surface tropical Pacific, model level 50 =
ocean surface) in both. The differences in `min` age come from the
surface-relaxation boundary cells, which sit very close to zero and
fluctuate in sign during the year — so the *minimum* is a snapshot
artefact, not a property of the fixed point.

**nk_steady** (final NK fixed point) — TSV: [timestep_multiplier_summary_nk_steady.tsv](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/timestep_multiplier_summary_nk_steady.tsv)

| `M` | Mean age (yr) | Max age (yr) | Min age (yr) | RMS Δ whole (yr) | RMS Δ surf (yr) | max\|Δ\| (yr) | max\|Δ\| (i,j,k) |
|---|---|---|---|---|---|---|---|
| 1 | 911.423 | 2443.003 | −116.443 | 0 | 0 | — | — |
| 2 | 915.299 | 2443.250 | −116.594 | 4.699 | 2.519 | 55.665 | (224, 40, 50) |
| 4 | 922.535 | 2443.785 | −116.758 | 13.385 | 6.918 | 118.109 | (224, 40, 50) |

**nk_periodic** (year-mean of run1yrNK) — TSV: [timestep_multiplier_summary_nk_periodic.tsv](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/timestep_multiplier_summary_nk_periodic.tsv)

| `M` | Mean age (yr) | Max age (yr) | Min age (yr) | RMS Δ whole (yr) | RMS Δ surf (yr) | max\|Δ\| (yr) | max\|Δ\| (i,j,k) |
|---|---|---|---|---|---|---|---|
| 1 | 911.429 | 2443.007 | −127.982 | 0 | 0 | — | — |
| 2 | 915.306 | 2443.254 | −128.169 | 4.698 | 2.494 | 54.373 | (224, 40, 50) |
| 4 | 922.541 | 2443.788 | −128.387 | 13.383 | 6.851 | 116.379 | (224, 40, 50) |

The stability-sweep tolerances (on the *1-year forward map*) were
~23 ms RMS whole-domain and ~88 ms surface at `M=4`. The periodic NK
state is ~580× worse at `M=4` (13.4 yr whole-domain, 6.9 yr surface) —
**expected**, because the equilibrium amplifies per-step truncation
error over the ventilation timescale, not over a single year. Whether
~13 yr (≈ 1.5% of the mean ventilation age of ~900 yr) is acceptable
depends on the downstream use of the NK output. See *Conclusions*.

To regenerate either table (e.g. after a re-run):

```bash
# both modes run cleanly in parallel — outputs land in separate subdirs
qsub -v COMPARE_TARGET=nk_steady,LUMP_AND_SPRAY=yes   scripts/plotting/plot_timestep_multiplier_sweep.sh
qsub -v COMPARE_TARGET=nk_periodic,LUMP_AND_SPRAY=yes scripts/plotting/plot_timestep_multiplier_sweep.sh
```

#### Diff plots vs M=1

Same pattern as the
[stability sweep diff plots](timestep_multiplier.md#diff-plots-vs-m1):
symmetric `:RdBu` (reversed) colormap, colour range capped at the 99th
percentile of `|age_M − age_1|` (= per-M `Δmax`, listed below), same
Δmax across all plots of a given M. PNGs land under
`periodic/{MC}_DTx{M}/diff_vs_DTx1_periodic_{nk_steady,nk_periodic}/`
— per-target subdirs are required because both NK modes share
`periodic/` as their discovery root, so a shared subdir would race when
the two sweep jobs run in parallel.

| Target | M=2 Δmax (yr) | M=4 Δmax (yr) |
|---|---|---|
| nk_steady | 17.05 | 48.66 |
| nk_periodic | 16.96 | 48.37 |

The two targets give *visually* indistinguishable diff fields (Δmax
agrees to ~1%), so the patterns described below apply equally to both;
they're shown side-by-side primarily as a sanity check that the NK
fixed point and the integrated periodic state really do agree.

##### nk_periodic (year-mean of run1yrNK — primary comparison)

###### M=2 vs M=1 — zonal averages (Δmax ≈ 16.96 yr)

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_zonal_avg_atlantic.png) |
| **Pacific** | **Indian** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_zonal_avg_indian.png) |

###### M=2 vs M=1 — horizontal slices (Δmax ≈ 16.96 yr)

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_500m.png) |
| **1000 m** | **2000 m** | **3000 m** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_periodic/DTx2_vs_DTx1_slice_3000m.png) |

###### M=4 vs M=1 — zonal averages (Δmax ≈ 48.37 yr)

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_zonal_avg_atlantic.png) |
| **Pacific** | **Indian** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_zonal_avg_indian.png) |

###### M=4 vs M=1 — horizontal slices (Δmax ≈ 48.37 yr)

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_500m.png) |
| **1000 m** | **2000 m** | **3000 m** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_periodic/DTx4_vs_DTx1_slice_3000m.png) |

##### nk_steady (NK fixed point — cross-check)

###### M=2 vs M=1 — zonal averages (Δmax ≈ 17.05 yr)

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_zonal_avg_atlantic.png) |
| **Pacific** | **Indian** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_zonal_avg_indian.png) |

###### M=2 vs M=1 — horizontal slices (Δmax ≈ 17.05 yr)

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_500m.png) |
| **1000 m** | **2000 m** | **3000 m** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1_periodic_nk_steady/DTx2_vs_DTx1_slice_3000m.png) |

###### M=4 vs M=1 — zonal averages (Δmax ≈ 48.66 yr)

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_zonal_avg_atlantic.png) |
| **Pacific** | **Indian** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_zonal_avg_indian.png) |

###### M=4 vs M=1 — horizontal slices (Δmax ≈ 48.66 yr)

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_500m.png) |
| **1000 m** | **2000 m** | **3000 m** |
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1_periodic_nk_steady/DTx4_vs_DTx1_slice_3000m.png) |

### Conclusions

1. **Does the Newton iteration count grow with `M`?** No. The total
   `Φ!` call count was identical (`62`) at all three `M ∈ {1, 2, 4}`,
   so the NK wall time scales purely with the per-Φ-call cost. The
   surface-relaxation eigenvalues that change with Δt don't shift the
   Krylov spectrum enough to alter the inner iteration count in this
   regime.
2. **Does `M=4` agree with `M=1` to within the stability-sweep
   tolerances?** No. The 1-year-forward map tolerances were ~23 ms RMS
   whole-domain and ~88 ms surface; at `M=4` the periodic state shows
   **13.4 yr** RMS whole-domain and **6.9 yr** surface — three orders
   of magnitude larger. This is the expected behaviour of an
   equilibrium amplifying per-step truncation error over the
   ventilation timescale (~900 yr mean age), not a single year. As a
   *fraction* of the mean age, the discrepancy is ~1.5 %, with `max|Δ|`
   = 118 yr concentrated at the surface tropical Pacific at `(i, j, k)
   = (224, 40, 50)` — i.e., the surface-relaxation cell where the BC
   itself scales with Δt by design.
3. **End-to-end speedup at `M=4`?** NK alone: **2.61×** (52 min → 20
   min wall). Whole pipeline TMbuild → plotNK: **1.77×** (79 min → 45
   min). The NK speedup is short of the `4×` ideal because per-Φ-call
   fixed cost (init + warmup + I/O) doesn't shrink with `M` — per-Φ
   timing alone is **3.04×** at `M=4`. The end-to-end ratio is further
   diluted by `M`-independent stages (TMbuild, TMsolve, run1yrNK,
   plotNK).

**Verdict.** `M=4` cuts NK wall by a factor of 2.6 without changing
convergence, at the cost of a ~1.5 % shift in the periodic age field
(localised at the surface). Whether to adopt it as the OM2-1 NK default
is a downstream-use call: acceptable for parameter studies and
benchmarking where relative comparisons dominate; **probably not**
acceptable for products comparing against observations at single-cell
precision. A middle ground — `M=2` for **1.72×** NK speedup at **4.70 yr**
RMS — may be the right operating point in practice.

For further reduction, the remaining levers are not Δt-side:
preconditioner quality (e.g. LSprec coarsening factor), better warm
start than TMage, or moving GMRES inner iters off-GPU.

## OM2-025: NK steady-state instability at SRK3-M=12 (2026-05-18)

The OM2-025 stability table in
[docs/timestep_multiplier.md](timestep_multiplier.md) marks SRK3-M=12
as ✓ — but that ✓ is from the standalone `run1yr` test only. When the
full NK pipeline at this config was used to build the inputs for the
cross-resolution age comparison
([docs/IAF_NK_age_comparison_plan.md](IAF_NK_age_comparison_plan.md))
we found:

| TW | NK exit | run1yrNK exit | Periodic FTS max\|age\| |
|---|---|---|---|
| 1968-1977 | 0 | 0 | a few thousand years (sane) |
| 1999-2008 | 0 | 0 | **≈ 1.74 × 10⁶¹ yr** (unphysical) |

The 1999-2008 pipeline's NK iterate carries Inf-scale values in
scattered cells. NK's residual stopping criterion didn't catch it
(the huge cells are sparse enough that the volume-weighted RMS norm
still passed the tolerance test, while per-cell maxima escaped). The
1-year forward map from that solution propagates the pathology into
the FTS we tried to plot — hence the `check_age_field` error in
[src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl)
and the abort of `compare_NK_ages.jl` on first contact with this
pipeline.

Why does 1999-2008 fail while 1968-1977 passes at the same SRK3-M=12
config? The 1999-2008 OMIP-IAF forcing carries more mid-latitude eddy
energy and tropical Pacific variability than 1968-1977; with Δt = 6 h
SRK3 is close enough to its absolute-stability boundary that the more
energetic regime tips it over. The 1-year `run1yr` test doesn't see
this — there's no Newton feedback amplifying per-step truncation
error over many outer iterates, and 1 year isn't long enough to
accumulate a blow-up.

**Action.** Re-run OM2-025 / 1999-2008 at two smaller-Δt configs in
parallel:

```bash
# AB2, Δt = 1.5 h (most conservative)
PARENT_MODEL=ACCESS-OM2-025 TIME_WINDOW=1999-2008 \
TIMESTEPPER=AB2 TIMESTEP_MULT=3 \
VELOCITY_SOURCE=totaltransport W_FORMULATION=wdiagnosed \
ADVECTION_SCHEME=centered2 MONTHLY_KAPPAV=yes \
LINEAR_SOLVER=Pardiso LUMP_AND_SPRAY=yes \
JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK \
  bash scripts/driver.sh

# SRK3, Δt = 4.5 h (one divisor step below the failing M=12)
PARENT_MODEL=ACCESS-OM2-025 TIME_WINDOW=1999-2008 \
TIMESTEPPER=SRK3 TIMESTEP_MULT=9 \
VELOCITY_SOURCE=totaltransport W_FORMULATION=wdiagnosed \
ADVECTION_SCHEME=centered2 MONTHLY_KAPPAV=yes \
LINEAR_SOLVER=Pardiso LUMP_AND_SPRAY=yes \
JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK \
  bash scripts/driver.sh
```

Each lands in its own `outputs/.../periodic/totaltransport_wdiagnosed_centered2_{AB2,SRK3}_mkappaV_DTx{3,9}/`
tree, so they don't collide. Whichever converges to a sane periodic
state first becomes the new MC for `compareNK`.

**Implication for the recommendation.** SRK3-M=12 as the OM2-025 NK
default needs an asterisk: it is the best *tested* speedup but is not
robust across IAF forcing decades. A defensible default is SRK3-M=9
(Δt = 4.5 h, 3× speedup) if it converges cleanly on both windows;
otherwise fall back to AB2-M=3 / SRK3-M=6 — slower but with more
margin to the stability boundary. The reruns above will resolve this.

### Update (2026-05-18) — AB2-M=4 also fails; AB2-M=3 / SRK3-M=9 both succeed

Discovered while running the TRAF (Time-Reversed Adjoint Flow)
campaign — same NK code path, just with the sign-flipped, time-reversed
velocity FTS — that this stability wall is **not specific to SRK3 at
Δt=6h.** AB2 at Δt=6h also fails for 1999-2008:

| Integrator × M | Δt | NK retcode | mean periodic age | location of blow-up |
|---|---|---|---|---|
| SRK3 M=12 | 6 h | Stalled (TRAF) | 2.4e-18 yr (garbage) | (1288, 1047, 36), `max(age)` reaches 6.8e+25 yr inside Φ! call #1 |
| AB2  M=4  | 6 h | Stalled (TRAF) | 1048.87 yr (untrusted, residual 1e+107) | same cells, max(age) 3.7e+40 yr by 0.083 yr, 2.5e+106 yr by end of yr |
| **SRK3 M=9** | 4.5 h | ✓ **Success** | **933.39 yr** | — |
| **AB2  M=3** | 1.5 h | ✓ **Success** | **917.35 yr** | — |

The same Δt=4.5h / 1.5h reruns also work for OM2-025/1968-1977
(885.88 / 868.80 yr respectively), so this isn't a 1999-2008-only
remedy — both timesteppers at Δt ≤ 4.5 h are robust across the IAF
forcing decades tested. See [TRAF_simulations.md § Attempt 3f](TRAF_simulations.md#attempt-3f--om2-025-ab2-m3--srk3-m9-2026-05-18-commits-5e69399--earlier)
for the full job-ID table and convergence trajectories.

**Cross-Δt mean-age summary for OM2-025 (TRAF NK, both TWs):**

| Δt config | effective Δt (s) | 1968-1977 (yr) | 1999-2008 (yr) |
|---|---:|---:|---:|
| SRK3 M=12 | 21600 | 892.89 | garbage (stalled) |
| AB2 M=4   | 21600 | 872.01 | 1048.87 (stalled) |
| **SRK3 M=9** | 16200 | **885.88** | **933.39** |
| **AB2 M=3**  | 16200 | **868.80** | **917.35** |

For 1968-1977 the four converged values span 868-893 yr (~3% spread)
with no monotone trend across Δt — i.e. just timestepper noise. For
1999-2008 the SRK3 / AB2 values at Δt = 16200 s differ by ~15 yr
(~2%), also within the expected timestepper-spread band.

**Revised recommendation for the OM2-025 NK production default.** Use
**SRK3-M=9** (Δt=4.5h, 3× speedup over Δt_base of 1800 s). This is the
largest Δt that converges in both TWs and under both forward (IAF) and
adjoint (TRAF) NK. The 1-year `run1yr` SRK3-M=12 ✓ in
[timestep_multiplier.md § OM2-025](timestep_multiplier.md#om2-025-δt_base--1800-s)
remains correct for *standalone* use; it just doesn't transfer cleanly
to NK at the more energetic 1999-2008 forcing.

The TRAF rollout also surfaced and fixed two related issues that had
been silently corrupting the warm-start path for both IAF and TRAF
runs across the entire campaign:
- `load_initial_age` was hard-coded to look for `steady_age_full_*.jld2`
  but `solve_matrix_age.jl` writes `steady_age_coarse_*.jld2` whenever
  `LUMP_AND_SPRAY=yes` (the production setting). No NK run had ever
  loaded its TMage warm-start. Fix: commit `ed4c140`.
- `NK_c` only `afterok`-depended on `TMbuild`, not `TMslv_c`, so even
  after fixing the filename NK could still start before the warm-start
  file was on disk. Fix: commit `d40e13a`.

With both fixes in place, the SRK3-M=9 / AB2-M=3 NK runs above were
the first in the campaign to actually start from the TM steady age
(max ~3000 yr, mean ~430 yr) rather than from zeros.
