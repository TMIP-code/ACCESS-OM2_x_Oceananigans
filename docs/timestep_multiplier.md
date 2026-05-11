# Timestep multiplier (`TIMESTEP_MULT`)

## Intent

The per-model `dt_seconds` values in [src/shared_utils/config.jl](../src/shared_utils/config.jl)
(5400 / 1800 / 400 s for OM2-1 / OM2-025 / OM2-01) inherit from
ACCESS-OM2's dynamical-core stability constraints (CFL on fast
barotropic / gravity-wave modes). The offline simulations in this
project advect a passive tracer (age) through a prescribed velocity
field with no dynamics — so the dynamical CFL is not the binding
constraint. The binding constraint is the tracer-advection /
diffusion CFL on the prescribed (u, v, w), which is much looser.

Goal: introduce an integer env flag `TIMESTEP_MULT=M` that scales
`Δt ← M·Δt_base`, and find the largest `M` for which the 1-year
simulation remains stable and the resulting age field is acceptably
close to the `M=1` reference.

## Valid multipliers

`year = 365.25 days = 31,557,600 s` (matches [src/setup_model.jl:39](../src/setup_model.jl#L39)).
Requiring `M·Δt_base` to divide one year exactly gives
`M ∈ Divisors(N_base)` where `N_base = year / Δt_base`.

| Model | `Δt_base` | `N_base` | Factorization |
|---|---|---|---|
| ACCESS-OM2-1 | 5400 s | 5844 | 2² · 3 · 487 |
| ACCESS-OM2-025 | 1800 s | 17,532 | 2² · 3² · 487 |
| ACCESS-OM2-01 | 400 s | 78,894 | 2 · 3⁴ · 487 |

487 is prime, so the small-`M` and large-`M` divisor families are
cleanly separated. Once `M·Δt_base ≥ 1 month` (M ≥ 487 / 1461 / 6574
respectively, with the first valid jump landing at exactly 1 month per
step) the simulation is uselessly coarse. The practical sweep range is
`Δt ≤ 18 h`.

Practical multipliers per model (rows aligned on `M`; cell shows the
resulting Δt where `M` is a valid divisor for that model):

| `M` | OM2-1 Δt | OM2-025 Δt | OM2-01 Δt |
|---|---|---|---|
| 1   | **1.5 h** | **30 min** | **6.67 min** |
| 2   | 3 h     | 1 h        | 13.3 min |
| 3   | 4.5 h   | 1.5 h      | 20 min |
| 4   | 6 h     | 2 h        | — |
| 6   | 9 h     | 3 h        | 40 min |
| 9   | —       | 4.5 h      | 1 h |
| 12  | 18 h    | 6 h        | — |
| 18  | —       | 9 h        | 2 h |
| 27  | —       | —          | 3 h |
| 36  | —       | 18 h       | — |
| 54  | —       | —          | 6 h |
| 81  | —       | —          | 9 h |
| 162 | —       | —          | 18 h |

Invalid `M` (those not dividing `N_base`) must error early — see
[Workflow → Validation](#validation).

## Plan

### Phase 1 — wire up `TIMESTEP_MULT`

1. Add `TIMESTEP_MULT=1` default in [scripts/env_defaults.sh](../scripts/env_defaults.sh)
   and export.
2. In [src/shared_utils/config.jl](../src/shared_utils/config.jl):
   parse `M = parse(Int, get(ENV, "TIMESTEP_MULT", "1"))`, validate
   `M ≥ 1`, then validate `mod(N_base, M) == 0` against the parent
   model's `Δt_base`, error with the full divisor list if invalid.
   Apply `Δt ← M·Δt`.
3. In `build_model_config` (same file) append `_DTx{M}` to
   `MODEL_CONFIG` **only when `M > 1`** so existing M=1 paths are
   unchanged. Mirror this in [scripts/env_defaults.sh](../scripts/env_defaults.sh)
   (the shell builds `MODEL_CONFIG` for PBS scripts).

### Phase 2 — 1-year stability sweep

Run [src/run_1year.jl](../src/run_1year.jl) at progressively larger
`M`, staged:

- **Stage 2a (initial sweep)**: `M ∈ {1, 2, 4}` (substituting `M=6`
  for OM2-01 where `M=4` is invalid). Cheap go/no-go check.
- **Stage 2b (full sweep)**: only if Stage 2a is clean — extend each
  model to its full practical-M range from the multiplier table
  (OM2-1 up to `M=12`, OM2-025 up to `M=36`, OM2-01 up to `M=162`).

Concrete invocation in [Workflow → Running the sweep](#running-the-sweep).
Single-run driver invocation looks like:

```bash
PARENT_MODEL=ACCESS-OM2-1 TIMESTEP_MULT=4 JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
```

Metrics to record per `M`:

| Metric | Source | Pass condition |
|---|---|---|
| Run completes | PBS log exit status | no NaN / no abort |
| Wall time (s) | Julia log (`run_1year`) | — (record) |
| Max age (yr) at t=1 yr | `validate_age_field` | finite, > 0, < ~3000 yr |
| Mean age (yr) at t=1 yr | `validate_age_field` | finite |
| RMS diff vs. M=1 (yr) | post-hoc analysis | < tolerance TBD |

`validate_age_field` is already invoked at the end of `run_1year.jl`
(see [src/run_1year.jl:51-53](../src/run_1year.jl#L51-L53)) — it prints
max / mean / min and detects NaN. The cross-`M` comparison is
post-hoc and reads the saved age field from
`outputs/{PM}/{EXP}/{TW}/standardrun/{MC}_DTx{M}/`.

### Phase 3 — Newton-Krylov periodic solve

If Phase 2 identifies a "safe" `M_max`, go straight to the
Newton-Krylov periodic solver ([solve_periodic_NK.jl](../src/solve_periodic_NK.jl))
at `M = 1` (baseline) and `M = M_max`. The 1-year Φ map is the inner
operation of the periodic solve, so a stable 1-year run is the only
prerequisite — no need to run intermediate `run_10years` /
`run_100years` checks.

Note on Δt-dependence of the JVP. The exact JVP is
`Φ(v; source_rate=0) - v` ([periodic_solver_common.jl:239-243](../src/periodic_solver_common.jl#L239-L243)),
which runs the simulation for 1 year. It is **not** Δt-invariant,
for the same reason the transport matrix is not (Phase 4): the surface
relaxation `relaxation_timescale = 3·Δt` in
[setup_model.jl:275-280](../src/setup_model.jl#L275-L280) is tied to
the timestep, so scaling `Δt` by `M` weakens the surface age=0 BC by
the same factor. So at `M = M_max` we're solving a slightly different
fixed-point problem (weaker surface restoring) — not a numerical
discretization error but a different continuum operator. Two
implications:

- The solved periodic age field at `M = M_max` should be compared
  against `M = 1` with the surface layer treated as the most
  sensitive region.
- GMRES convergence rate may differ between `M` (different
  preconditioning effectiveness, different eigenvalue distribution
  near the surface).

If we choose option (b) in Phase 4 (decouple `relaxation_timescale`
from `Δt`), this Δt-dependence disappears from the JVP too, and the
`M = M_max` solve becomes a pure speedup test.

### Phase 4 — transport matrix (mostly Δt-independent)

The matrix build in [src/create_matrix.jl](../src/create_matrix.jl)
(via [src/matrix_setup.jl](../src/matrix_setup.jl)) takes the Jacobian
of the **instantaneous** tendency `∂c/∂t` — no simulation, no
timestepper invoked. The only place `Δt` enters is the surface-cell
relaxation forcing:

```julia
age_parameters = (; relaxation_timescale = 3Δt, source_rate = 1.0)
@inline linear_source_sink(i, j, k, grid, clock, fields, params) =
    ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)
```
([matrix_setup.jl:250-257](../src/matrix_setup.jl#L250-L257))

Consequences:

- **Sparsity pattern, coloring, autodiff prep, build cost**: bitwise
  identical across `M`. No re-validation needed.
- **Off-surface rows of M**: bitwise identical across `M`.
- **Surface rows of M**: diagonal entry is `-1/(3·Δt)`, so scales as
  `1/M`. The surface "Dirichlet-like" age=0 BC weakens at larger `M`.
  At `M=1` (OM2-1) the timescale is 4.5 h; at `M=12` it's 54 h —
  still much shorter than any ocean ventilation timescale, so the
  surface age should remain near zero in the steady solve, but
  worth verifying.

Two ways forward:

(a) **Keep `relaxation_timescale = 3·Δt`** and verify that the
steady-state age solve agrees across `M` (small differences expected
in the top layer only). Cheap: matrix builds are unchanged in cost,
only the solve needs re-running.

(b) **Decouple `relaxation_timescale` from `Δt`** in *both*
[matrix_setup.jl:250-257](../src/matrix_setup.jl#L250-L257) and
[setup_model.jl:275-280](../src/setup_model.jl#L275-L280) — use a
fixed physical value (e.g. `3 × Δt_base` for that parent model, or a
stated absolute scale like `1 hour`). Then:
- the transport matrix is literally Δt-independent and Phase 4
  collapses to "no change needed; reuse the existing M.jld2 at any
  `M`",
- the 1-year Φ map (and therefore the JVP used in Phase 3) is also
  Δt-independent up to discretization error, so the NK solve at
  `M = M_max` becomes a pure speedup test rather than a "weaker
  surface BC" experiment.

This is also the cleaner stance for the matrix-based steady solver,
where `Δt` is conceptually meaningless.

**Decision pending** (do this before kicking off Stage 2b): commit to
(a) or (b). If (b), apply the decoupling change *before* the
follow-up sweep so all `M > 1` runs share the same surface BC and the
diff plots isolate the discretization effect.

## Workflow

### Implementation checklist

- [ ] [scripts/env_defaults.sh](../scripts/env_defaults.sh): set
      `TIMESTEP_MULT=${TIMESTEP_MULT:-1}`, export, log, and append
      `_DTx${TIMESTEP_MULT}` to `MODEL_CONFIG` when `TIMESTEP_MULT != 1`.
- [ ] [src/shared_utils/config.jl](../src/shared_utils/config.jl):
      parse + validate `M` against the parent model's `Δt_base`,
      multiply `Δt`, append `_DTx{M}` to `MODEL_CONFIG` in
      `build_model_config`.
- [ ] Smoke test: `TIMESTEP_MULT=1` produces bitwise-identical results
      to the unmodified pipeline (same output path, same wall time).
- [ ] Smoke test: `TIMESTEP_MULT=0` and `TIMESTEP_MULT=5` (for OM2-1)
      both error early with a clear message listing valid divisors.
- [ ] After Stage 2a results land: decide Phase 4 option (a) keep
      `relaxation_timescale = 3·Δt` everywhere, or (b) decouple from
      `Δt` in [src/setup_model.jl:275-280](../src/setup_model.jl#L275-L280)
      *and* [src/matrix_setup.jl:250-257](../src/matrix_setup.jl#L250-L257).
      If (b), apply the change before Stage 2b.

### Validation

The `M`-validity check (in `load_project_config`) errors with:

```
ERROR: TIMESTEP_MULT=5 is not a divisor of N_base=5844 (= year/Δt_base
for ACCESS-OM2-1). Valid multipliers ≤ 12 (Δt ≤ 18 h):
{1, 2, 3, 4, 6, 12}. Next valid value is 487 (= 1 month per step).
```

We surface the full picture rather than silently coercing —
consistent with the project's "error rather than silently coercing
assumed values" stance.

### Running the sweep

**Initial sweep: `M ∈ {1, 2, 4}`** — 1× baseline, a small bump, and a
4× bump. Cheap to run, fast to interpret, and tells us whether the
"large Δt is fine for a passive tracer" hypothesis even holds before
we commit to the fuller sweep. Note `M=4` is not a valid divisor for
OM2-01 (`N_base = 2 · 3⁴ · 487` has only one factor of 2); the
closest analog there is `M=6` (Δt = 40 min).

Each invocation runs the simulation *and* the per-run diagnostic
plots (`run1yr-plot1yr`) so each `{MC}_DTx{M}/` directory ends up with
both the saved age field and the standard zonal-mean / horizontal-slice
PNGs — useful for eyeballing before the diff plots are produced.

```bash
# OM2-1 (Δt = 1.5 h, 3 h, 6 h)
for M in 1 2 4; do
  PARENT_MODEL=ACCESS-OM2-1 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
done

# OM2-025 (Δt = 30 min, 1 h, 2 h)
for M in 1 2 4; do
  PARENT_MODEL=ACCESS-OM2-025 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
done

# OM2-01 (Δt = 6.67 min, 13.3 min, 40 min — substituting M=6 for M=4)
for M in 1 2 6; do
  PARENT_MODEL=ACCESS-OM2-01 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
done
```

**Follow-up sweep (only if initial sweep passes)**: extend each model
to its full practical range from the multiplier table — OM2-1 up to
`M=12`, OM2-025 up to `M=36`, OM2-01 up to `M=162`.

### Comparison script (TBD)

A small post-hoc Julia script that loads `age_*.jld2` from each
`{MC}_DTx{M}/` directory and produces, for each `M > 1`:

1. **Scalar diagnostics**: volume-weighted RMS difference vs. `M = 1`,
   max abs difference, mean difference, and where they live in
   (lat, depth). These feed the "RMS Δ vs M=1 (yr)" column of the
   Results tables.
2. **Difference plots**: the same set of diagnostics that
   `plot_age_diagnostics` ([analysis_and_plotting.jl:174](../src/shared_utils/analysis_and_plotting.jl#L174))
   already produces for a single run, but applied to the diff field
   `age_M − age_1`:
   - Zonal averages × 4 basins (global / Atlantic / Pacific / Indian)
     — lat × depth contourf with a symmetric diverging colormap
     (e.g. `:RdBu_r`) centred at 0.
   - Horizontal slices at 100 / 200 / 500 / 1000 / 2000 / 3000 m —
     heatmap with the same diverging colormap.
   - Colour range: probably auto-scaled per `M` to the 99th-percentile
     of `|age_M − age_1|`; record the range in the figure title or
     filename so figures across `M` are comparable.

Likely lives in `src/plot_timestep_multiplier_sweep.jl`. Strategy: load
the `M=1` `age_3D` once, then loop over `M > 1` values present on disk,
call `plot_age_diagnostics(age_M − age_1, grid, wet3D, vol_3D,
diff_dir, "DTx$(M)_vs_DTx1"; colormap = cgrad(:RdBu_r, ...),
colorrange = (-Δmax, Δmax), levels = …, colorbar_label = "Age diff
(years)")`. The existing function takes generic `colormap` / `levels` /
`colorrange` / `colorbar_label` keywords so the same machinery handles
diff plots with no code change to `plot_age_diagnostics`.

Output directory: `outputs/{PM}/{EXP}/{TW}/standardrun/{MC}_DTx{M}/diff_vs_DTx1/`
— each `M > 1` run's directory gains a `diff_vs_DTx1/` subdir
alongside its own `plots/`, so the comparison artifacts live with the
run they describe.

## Results

Rows marked **(2a)** are part of the initial sweep; rows marked
**(2b)** are added in the follow-up sweep only if Stage 2a passes.

### OM2-1 (Δt = 5400 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 1.5 h | 5844 | 2a | — | — | — | — | 0 | — |
| 2  | 3 h   | 2922 | 2a | — | — | — | — | — | — |
| 3  | 4.5 h | 1948 | 2b | — | — | — | — | — | — |
| 4  | 6 h   | 1461 | 2a | — | — | — | — | — | — |
| 6  | 9 h   | 974  | 2b | — | — | — | — | — | — |
| 12 | 18 h  | 487  | 2b | — | — | — | — | — | — |

### OM2-025 (Δt = 1800 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 30 min | 17532 | 2a | — | — | — | — | 0 | — |
| 2  | 1 h    | 8766  | 2a | — | — | — | — | — | — |
| 3  | 1.5 h  | 5844  | 2b | — | — | — | — | — | — |
| 4  | 2 h    | 4383  | 2a | — | — | — | — | — | — |
| 6  | 3 h    | 2922  | 2b | — | — | — | — | — | — |
| 9  | 4.5 h  | 1948  | 2b | — | — | — | — | — | — |
| 12 | 6 h    | 1461  | 2b | — | — | — | — | — | — |
| 18 | 9 h    | 974   | 2b | — | — | — | — | — | — |
| 36 | 18 h   | 487   | 2b | — | — | — | — | — | — |

### OM2-01 (Δt = 400 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1   | 6.67 min  | 78894 | 2a | — | — | — | — | 0 | — |
| 2   | 13.3 min  | 39447 | 2a | — | — | — | — | — | — |
| 3   | 20 min    | 26298 | 2b | — | — | — | — | — | — |
| 6   | 40 min    | 13149 | 2a | — | — | — | — | — | — |
| 9   | 1 h       | 8766  | 2b | — | — | — | — | — | — |
| 18  | 2 h       | 4383  | 2b | — | — | — | — | — | — |
| 27  | 3 h       | 2922  | 2b | — | — | — | — | — | — |
| 54  | 6 h       | 1461  | 2b | — | — | — | — | — | — |
| 81  | 9 h       | 974   | 2b | — | — | — | — | — | — |
| 162 | 18 h      | 487   | 2b | — | — | — | — | — | — |

### Conclusions

TBD after sweep completes.
