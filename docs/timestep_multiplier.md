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

Practical multipliers per model:

| New Δt | OM2-1 `M` | OM2-025 `M` | OM2-01 `M` |
|---|---|---|---|
| 6.67 min | — | — | 1 |
| 13.3 min | — | — | 2 |
| 20 min   | — | — | 3 |
| 30 min   | — | 1 | — |
| 40 min   | — | — | 6 |
| 1 h      | — | 2 | 9 |
| 1.5 h    | **1** | 3 | — |
| 2 h      | — | 4 | 18 |
| 3 h      | 2 | 6 | 27 |
| 4.5 h    | 3 | 9 | — |
| 6 h      | 4 | 12 | 54 |
| 9 h      | 6 | 18 | 81 |
| 18 h     | 12 | 36 | 162 |

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

### Phase 3 — long-run / periodic stability

If Phase 2 identifies a "safe" `M_max`, repeat with the 10-year and
100-year scripts ([run_10years.jl](../src/run_10years.jl),
[run_100years.jl](../src/run_100years.jl)) at `M = 1` and `M = M_max`
(and maybe an intermediate point) to confirm errors don't accumulate
over many years.

If we plan to use `M > 1` in the Newton-Krylov periodic solver
([solve_periodic_NK.jl](../src/solve_periodic_NK.jl)), verify that the
exact-JVP machinery still gives a clean linear residual — the JVP is
defined as `Φ(v; source_rate=0) - v` ([periodic_solver_common.jl:239-243](../src/periodic_solver_common.jl#L239-L243))
and is independent of `Δt` mathematically, but a too-large Δt could
amplify roundoff. Worth one sanity check.

### Phase 4 — transport matrix (optional)

The matrix build in [src/create_matrix.jl](../src/create_matrix.jl)
also uses `Δt` (via `matrix_setup.jl`). Two questions:
(a) does the Jacobian sparsity coloring still work at large `Δt`?
(b) does the resulting steady-state age solve agree with the M=1
matrix solve? Defer until Phases 1–2 are clean.

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
