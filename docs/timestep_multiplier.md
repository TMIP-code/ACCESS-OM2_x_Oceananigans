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

For each parent model, run [src/run_1year.jl](../src/run_1year.jl) at
all practical `M` from the table above. Driver invocation:

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

Sanity-check the exact-JVP machinery on the `M = M_max` solve: the
JVP is defined as `Φ(v; source_rate=0) - v` ([periodic_solver_common.jl:239-243](../src/periodic_solver_common.jl#L239-L243))
and is independent of `Δt` mathematically, but a too-large Δt could
amplify roundoff and slow GMRES convergence.

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

(b) **Decouple `relaxation_timescale` from `Δt`** in the matrix build
— use a fixed physical value (e.g. `3 × Δt_base` for that parent
model, or a stated absolute scale like `1 hour`). Then M is literally
Δt-independent and Phase 4 collapses to "no change needed; reuse the
existing M.jld2 at any `M`". This is also the cleaner stance for the
matrix-based steady solver, where `Δt` is conceptually meaningless.

Decision pending: skip Phase 4 entirely if (b) is chosen; otherwise
run the steady age solve with `M = 1` and `M = M_max` matrices and
compare.

## Workflow

### Implementation checklist

- [ ] [scripts/env_defaults.sh](../scripts/env_defaults.sh): set
      `TIMESTEP_MULT=${TIMESTEP_MULT:-1}`, export, log, and append
      `_DTx${TIMESTEP_MULT}` to `MODEL_CONFIG` when `TIMESTEP_MULT != 1`.
- [ ] [src/shared_utils/config.jl](../src/shared_utils/config.jl):
      parse + validate `M` against the parent model's `Δt_base`,
      multiply `Δt`, append `_DTx{M}` to `MODEL_CONFIG` in
      `build_model_config`.
- [ ] Confirm `relaxation_timescale = 3·Δt` ([src/setup_model.jl:276](../src/setup_model.jl#L276))
      behaves as expected when Δt scales — the relaxation should scale
      with Δt by construction (it's "3 steps"); flag if a fixed
      physical timescale is preferred instead.
- [ ] Smoke test: `TIMESTEP_MULT=1` produces bitwise-identical results
      to the unmodified pipeline (same output path, same wall time).
- [ ] Smoke test: `TIMESTEP_MULT=0` and `TIMESTEP_MULT=5` (for OM2-1)
      both error early with a clear message listing valid divisors.

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

For OM2-1:

```bash
for M in 1 2 3 4 6 12; do
  PARENT_MODEL=ACCESS-OM2-1 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr bash scripts/driver.sh
done
```

For OM2-025:

```bash
for M in 1 2 3 4 6 9 12 18 36; do
  PARENT_MODEL=ACCESS-OM2-025 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr bash scripts/driver.sh
done
```

For OM2-01:

```bash
for M in 1 2 3 6 9 18 27 54 81 162; do
  PARENT_MODEL=ACCESS-OM2-01 TIMESTEP_MULT=$M \
    JOB_CHAIN=run1yr bash scripts/driver.sh
done
```

### Comparison script (TBD)

A small post-hoc Julia script that loads `age_*.jld2` from each
`{MC}_DTx{M}/` directory and computes volume-weighted RMS difference
vs. the `M=1` reference. Likely lives in `src/plot_timestep_multiplier_sweep.jl`
and is invoked once the sweep is complete.

## Results

### OM2-1 (Δt = 5400 s baseline)

| `M` | Δt | Steps/yr | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.5 h | 5844 | — | — | — | — | 0 | — |
| 2 | 3 h | 2922 | — | — | — | — | — | — |
| 3 | 4.5 h | 1948 | — | — | — | — | — | — |
| 4 | 6 h | 1461 | — | — | — | — | — | — |
| 6 | 9 h | 974 | — | — | — | — | — | — |
| 12 | 18 h | 487 | — | — | — | — | — | — |

### OM2-025 (Δt = 1800 s baseline)

| `M` | Δt | Steps/yr | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|
| 1 | 30 min | 17532 | — | — | — | — | 0 | — |
| 2 | 1 h | 8766 | — | — | — | — | — | — |
| 3 | 1.5 h | 5844 | — | — | — | — | — | — |
| 4 | 2 h | 4383 | — | — | — | — | — | — |
| 6 | 3 h | 2922 | — | — | — | — | — | — |
| 9 | 4.5 h | 1948 | — | — | — | — | — | — |
| 12 | 6 h | 1461 | — | — | — | — | — | — |
| 18 | 9 h | 974 | — | — | — | — | — | — |
| 36 | 18 h | 487 | — | — | — | — | — | — |

### OM2-01 (Δt = 400 s baseline)

| `M` | Δt | Steps/yr | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.67 min | 78894 | — | — | — | — | 0 | — |
| 2 | 13.3 min | 39447 | — | — | — | — | — | — |
| 3 | 20 min | 26298 | — | — | — | — | — | — |
| 6 | 40 min | 13149 | — | — | — | — | — | — |
| 9 | 1 h | 8766 | — | — | — | — | — | — |
| 18 | 2 h | 4383 | — | — | — | — | — | — |
| 27 | 3 h | 2922 | — | — | — | — | — | — |
| 54 | 6 h | 1461 | — | — | — | — | — | — |
| 81 | 9 h | 974 | — | — | — | — | — | — |
| 162 | 18 h | 487 | — | — | — | — | — | — |

### Conclusions

TBD after sweep completes.
