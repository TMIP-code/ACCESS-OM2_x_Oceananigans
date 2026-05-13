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

## Design choice: surface relaxation scales with Δt

Both the simulation ([setup_model.jl:275-280](../src/setup_model.jl#L275-L280))
and the transport matrix build ([matrix_setup.jl:250-257](../src/matrix_setup.jl#L250-L257))
use `relaxation_timescale = 3·Δt` for the surface-layer age=0
restoring forcing. We deliberately keep this Δt-coupled when
`TIMESTEP_MULT > 1`, rather than pinning it to a fixed physical
timescale, because:

- The "3 timesteps" rule keeps the surface forcing safely resolvable
  by the integrator at every `M`, regardless of resolution or scheme.
- The relaxation only affects the top layer (`k ≥ Nz`); off-surface
  dynamics are Δt-coupled only through truncation error.
- At `M = 12` (OM2-1) the relaxation timescale grows from 4.5 h to
  54 h — still far below ocean ventilation timescales — so the
  surface age should remain near zero in the steady solve.

Consequences to keep in mind when reading results:

- **Transport matrix M**: off-surface rows are bitwise identical
  across `M`; only the surface diagonal entry (`-1/(3·Δt)`) scales as
  `1/M`. Sparsity, coloring, and build cost are unaffected, so
  rebuilding M at the new Δt is cheap.
- **1-year Φ map** (used by `run_1year.jl` and as the inner
  operation of the NK exact JVP): not Δt-invariant — scaling Δt by
  `M` weakens the surface age=0 BC, so we are solving a slightly
  different continuum operator at `M = M_max`, not just a different
  discretization.
- **Diff plots and RMS metrics**: treat the surface layer as the
  most sensitive region; report whole-domain and surface-layer
  metrics separately.

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

### Phase 3 — Full periodic pipeline at `M_max`

If Stage 2b identifies a "safe" `M_max`, run the full periodic
pipeline at `M = 1` (baseline) and `M = M_max`. The NK solver
consumes the matrix-based steady age as its warm start
(`INITIAL_AGE=TMage` in
[periodic_solver_common.jl:82-110](../src/periodic_solver_common.jl#L82-L110)),
so the transport matrix is a hard prerequisite — not a parallel
exercise. Three sequential steps per `M`:

1. **Transport matrix** ([create_matrix.jl](../src/create_matrix.jl)):
   rebuild M at the new Δt. Off-surface rows and sparsity/coloring
   are bitwise identical to `M = 1` (see [Design choice](#design-choice-surface-relaxation-scales-with-δt));
   only the surface diagonal scales as `1/M`. Build cost is
   unchanged.
2. **Steady-state age solve** ([solve_matrix_age.jl](../src/solve_matrix_age.jl)):
   produces `steady_age_full_*.jld2`, which the NK solver loads as
   `INITIAL_AGE=TMage`.
3. **Newton-Krylov solve** ([solve_periodic_NK.jl](../src/solve_periodic_NK.jl)):
   run with the `M = M_max` warm start. The 1-year Φ map is the
   inner operation; Stage 2's stability check is the prerequisite.

Driver invocation (one chain per `M`):

```bash
for M in 1 M_max; do
  PARENT_MODEL=ACCESS-OM2-1 TIMESTEP_MULT=$M \
    JOB_CHAIN=TMbuild..NK bash scripts/driver.sh
done
```

Comparison metrics at `M = M_max` vs `M = 1`:

- **Periodic age field**: volume-weighted RMS difference both
  whole-domain and surface-only (the surface layer is the sensitive
  region per the design note above). Reuse the diff-plot machinery
  from [Comparison script](#comparison-script-tbd) on the periodic
  age field.
- **NK convergence rate**: number of Newton iterations and total
  GMRES iterations to a fixed residual tolerance. May differ across
  `M` even with the warm start, because the surface eigenvalue
  distribution of the JVP changes.
- **End-to-end wall time**: TMbuild + TMsolve + NK across `M`. This
  is the speedup that motivates the whole exercise.

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

### Comparison script

[src/plot_timestep_multiplier_sweep.jl](../src/plot_timestep_multiplier_sweep.jl)
is a CPU-only post-hoc script that discovers every `{MC}` and
`{MC}_DTx{M}/` directory under `outputs/{PM}/{EXP}/{TW}/standardrun/`
containing an `age_1year.jld2`, loads the final age snapshot from each,
and for every `M` reports:

- volume-weighted mean / max / min age (years)
- RMS Δ vs M=1 (whole-domain + surface-layer k=Nz)
- max|Δ| and its `(i, j, k)` location

Output: a fixed-width table printed to stdout *plus* a TSV at
`outputs/{PM}/{EXP}/{TW}/standardrun/timestep_multiplier_summary.tsv`
that can be copied into the Results tables.

Submit it with:

```bash
PARENT_MODEL=ACCESS-OM2-1 qsub scripts/plotting/plot_timestep_multiplier_sweep.sh
```

(or via `driver.sh` once the step is wired into the DAG).

It also produces **diff plots** for each `M > 1` (skip with
`DIFF_PLOTS=no`): zonal averages × 4 basins (global / Atlantic /
Pacific / Indian) and horizontal slices at 100 / 200 / 500 / 1000 /
2000 / 3000 m, on a symmetric `:RdBu_r` colormap auto-scaled to the
99th percentile of `|age_M − age_1|` (same Δmax across all plots of a
given M so they're directly comparable). Plots land in
`outputs/{PM}/{EXP}/{TW}/standardrun/{MC}_DTx{M}/diff_vs_DTx1/`, one
subdir per `M > 1` so the comparison artifacts live with the run they
describe.

## Results

Rows marked **(2a)** are part of the initial sweep; rows marked
**(2b)** are added in the follow-up sweep only if Stage 2a passes.

Wall time below is the Julia-internal simulation wall time
(`Simulation is stopping after running for X` from the run log, extracted
via [scripts/plotting/plot_simtime_vs_walltime.py](../scripts/plotting/plot_simtime_vs_walltime.py)).
It excludes Julia startup, package loading, and model setup; the PBS-side
`walltime_used` is larger by ~5–8 min of startup overhead.

**Caveat: these are not benchmark runs.** `run_1year.jl` writes the full
age field (and `u`, `v`, `w`, `η`, `dt_sigma`, `eta_n`, `sigma_cc` —
~1.7 GB per run) at every output interval, so the reported wall time is
(simulation step time) + (output writing). At `M = 4` the output cost
becomes a larger fraction of the total because the simulation step time
shrinks ~linearly with `M` while the per-snapshot I/O cost stays roughly
constant. For a pure step-time speedup measurement use
`run_1year_benchmark.jl` (no output writers).

Mean age (yr) and RMS Δ vs M=1 (yr) come from
[src/plot_timestep_multiplier_sweep.jl](../src/plot_timestep_multiplier_sweep.jl)
post-hoc (a CPU job — `qsub scripts/plotting/plot_timestep_multiplier_sweep.sh`);
"—" means "not yet run".

### OM2-1 (Δt = 5400 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 1.5 h | 5844 | 2a | ✅ pass | 108.1 | 2.083 | 0.9730 | 0 | 168060698 |
| 2  | 3 h   | 2922 | 2a | ✅ pass |  88.0 | 1.978 | 0.9743 | 0.0096 | 168060700 |
| 3  | 4.5 h | 1948 | 2b | — | — | — | — | — | — |
| 4  | 6 h   | 1461 | 2a | ✅ pass |  78.1 | 1.855 | 0.9765 | 0.0229 | 168060703 |
| 6  | 9 h   | 974  | 2b | — | — | — | — | — | — |
| 12 | 18 h  | 487  | 2b | — | — | — | — | — | — |

Julia-internal speedup so far: M=2 → 1.23×, M=4 → 1.38×.

The age-field hotspot at (i=65, j=209, k=36) — interior, mid-depth —
is the *raw* maximum and is the same at all three M values, so the
dynamics are consistent and the divergence is in the size of that
overshoot rather than its location.

The divergence introduced by larger Δt lives almost entirely in the
surface layer (k=50): the maximum |Δ| for both M=2 and M=4 is at
k=50, and the surface-layer RMS Δ vs M=1 is 3–4× the whole-domain
RMS (M=2: 0.033 surface vs 0.010 whole; M=4: 0.088 surface vs 0.023
whole). This is the expected behaviour of the design choice — the
relaxation timescale `3·Δt` scales with M, weakening the surface
age=0 BC; the off-surface dynamics see only truncation error from
larger Δt. Whole-domain mean age drifts by ≤ 4 ms (0.973 → 0.976 yr)
across the M=1→4 range, far below ocean-ventilation timescales.

Comparison job: 168081165 — raw output at
[outputs/.../1968-1977/standardrun/timestep_multiplier_summary.tsv](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/timestep_multiplier_summary.tsv).

##### Diff plots vs M=1

[src/plot_timestep_multiplier_sweep.jl](../src/plot_timestep_multiplier_sweep.jl)
emits `age_M − age_1` zonal averages (4 basins) and horizontal slices
(6 depths) on a symmetric diverging colormap auto-scaled to the 99th
percentile of `|Δ|` (recorded in each figure title as Δmax). Same Δmax
across all plots of a given M so they're directly comparable. PNGs land
in `outputs/.../{MC}_DTx{M}/diff_vs_DTx1/`.

Note: `outputs/` is on scratch and gitignored, so the embeds below
render in a local Markdown previewer (VS Code, etc.) but not on GitHub.
Open the doc locally to view.

###### M = 2 vs M = 1

Zonal averages (4 basins):

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_zonal_avg_atlantic.png) |

| Pacific | Indian |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_zonal_avg_indian.png) |

Horizontal slices (6 depths):

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_500m.png) |

| 1000 m | 2000 m | 3000 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/DTx2_vs_DTx1_slice_3000m.png) |

Directory: [outputs/.../DTx2/diff_vs_DTx1/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx2/diff_vs_DTx1/)

###### M = 4 vs M = 1

Zonal averages (4 basins):

| Global | Atlantic |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_zonal_avg_atlantic.png) |

| Pacific | Indian |
|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_zonal_avg_indian.png) |

Horizontal slices (6 depths):

| 100 m | 200 m | 500 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_500m.png) |

| 1000 m | 2000 m | 3000 m |
|---|---|---|
| ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/DTx4_vs_DTx1_slice_3000m.png) |

Directory: [outputs/.../DTx4/diff_vs_DTx1/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_DTx4/diff_vs_DTx1/)

#### OM2-1 benchmark wall times (no output writers)

`run_1year_benchmark.jl` runs the same step loop with output writers
disabled, isolating step time from I/O cost. Submitted via
`JOB_CHAIN=run1yrfast`.

| `M` | Δt | Steps/yr | Benchmark wall (s) | Speedup vs M=1 | Job ID |
|---|---|---|---|---|---|
| 1 | 1.5 h | 5844 | 38.6 | 1.00× | 168081163 |
| 4 | 6 h   | 1461 |  9.5 | 4.06× | 168081164 |

The pure step-time speedup is **4.06×** at M=4 — essentially perfectly
linear with M (a 4× larger Δt gives a 4× shorter simulation). The fact
that `run_1year` (which writes outputs) showed only 1.38× speedup is
explained by the I/O cost being constant:

|       | run_1year wall (s) | benchmark wall (s) | I/O overhead (s) |
|-------|-------------------:|-------------------:|-----------------:|
| M = 1 | 108.1 | 38.6 | 69.5 |
| M = 4 |  78.1 |  9.5 | 68.6 |

I/O is ~69 s regardless of M (same number of output snapshots, same
per-snapshot cost). At M=4 the I/O is already 88% of total `run_1year`
wall time. For longer runs, the periodic-solver inner loop (no per-step
I/O) gets the full 4× speedup — which is the speedup that matters for
the Newton-Krylov use case.

#### Timestepper comparison: SRK3 stability sweep

The NK pipeline at M=6 / M=12 with the default `AB2` timestepper failed
([NK doc](timestep_multiplier_NK.md#failure-modes)) — M=6 crashed
inside the 1-year forward map (max age jumped to 1e4 yr within the first
year), M=12 stalled. Stability of the 1-year forward map is a
*necessary* precondition for the NK pipeline at those M's. Hypothesis:
a richer absolute-stability region (multi-stage RK) lets the forward
map survive at larger Δt. SRK3 (3-stage Split Runge-Kutta) is the natural
first try.

Per-step *new* tendency-evaluation count is **3 for SRK3 vs 1 for AB2**
(AB2 caches `G(t_{n-1})` from the previous step, so it does only 1 new
RHS eval per step). So SRK3 step cost is roughly 3× AB2. The net win
needs the SRK3 wall to push to an M that drops the per-year step count
by more than 3× — i.e. M ≥ 12 on OM2-1 — to beat AB2 at M=4.

This sweep tests stability only — RMS Δ vs AB2-M=1 left blank since the
goal is "does it complete cleanly".

| `M` | Δt | TIMESTEPPER | Status | Sim wall (s) | Max age (yr) | Mean age (yr) | Job ID |
|---|---|---|---|---|---|---|---|
| 1 | 1.5 h | AB2 (ref) | ✓ | 108.1 | 2.083 | 0.973 | 168060698 |
| 2 | 3 h | AB2 | ✓ | 88.0 | 1.978 | 0.974 | 168060700 |
| 4 | 6 h | AB2 | ✓ | 78.1 | 1.855 | 0.977 | 168060703 |
| 6 | 9 h | AB2 | (not run as 1yr; NK inner sim crashed at this Δt) | — | — | — | — |
| 12 | 18 h | AB2 | (not run as 1yr; NK stalled at this Δt) | — | — | — | — |
| 4 | 6 h | SRK3 | ✓ | 100.4 | 1.854 | 0.977 | 168282254 |
| 6 | 9 h | SRK3 | ✓ | 90.1 | 1.767 | 0.977 | 168282256 |
| 12 | 18 h | SRK3 | ✓ | 79.4 | 1.592 | 0.977 | 168282258 |

Submission:
```bash
for M in 4 6 12; do
  PARENT_MODEL=ACCESS-OM2-1 TIMESTEP_MULT=$M TIMESTEPPER=SRK3 \
    JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
done
```
Output paths: `standardrun/cgridtransports_wdiagnosed_centered2_SRK3_DTx{4,6,12}/`
— separate from the existing `_AB2` paths.

##### Per-step cost (AB2 vs SRK3 at matched M=4)

| | Sim wall (s) | Minus I/O (~69 s) | Step work / total steps | Per-step ratio |
|---|---|---|---|---|
| AB2 M=4  | 78.1  | 9.1 s  | 9.1 s / 1461 = 6.2 ms/step  | 1× |
| SRK3 M=4 | 100.4 | 31.4 s | 31.4 s / 1461 = 21.5 ms/step | **3.45×** |

So a SRK3 step costs **~3.45× an AB2 step** at matched M (consistent with the
textbook 3:1 ratio of new RHS evaluations per step: AB2 reuses Gⁿ⁻¹ from the
previous step, SRK3 does 3 fresh stages).

##### Verdict — SRK3 stability sweep on OM2-1

**Stability**: SRK3 completed cleanly at all three M's, including the
M=6 and M=12 cases where AB2 inside the NK pipeline blew up
([see NK doc](timestep_multiplier_NK.md#failure-modes)). Max age at
t=1yr stays in the ~1.6–1.9 yr range across `M ∈ {4, 6, 12}` — no sign
of the explosive divergence seen in AB2-OM2-025 M=4.

**Wall time**: SRK3 sim wall **drops monotonically** with M (100.4 →
90.1 → 79.4 s) — at M=12 it's essentially **tied with AB2 M=4** (79.4 s
vs 78.1 s). So SRK3-M=12 is a "free upgrade": same wall as the
fastest-stable AB2 config, but with the larger Δt the *NK inner loop*
can use (where the I/O cost vanishes and the M-speedup is felt fully).

**Next step for the NK pipeline**: re-run the NK sweep with
`TIMESTEPPER=SRK3 TIMESTEP_MULT=12 INITIAL_AGE=TMage` and compare to
the AB2-M=4 baseline. The win is twofold:

1. The 1-year forward map is *known stable* at SRK3-M=12 (this sweep).
2. The NK inner loop has no per-step I/O, so the per-step speedup of
   M=12 vs M=4 (3× fewer steps) directly translates to wall reduction
   — and the SRK3 step cost penalty (3.45× per step) is the cost.
   Net: SRK3-M=12 NK wall ≈ AB2-M=4 NK wall = (52/2.61) ≈ 20 min,
   but with the *known* convergence in the SRK3-M=12 regime.

### OM2-025 (Δt = 1800 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 30 min | 17532 | 2a | — | — | — | — | 0 | — |
| 2  | 1 h    | 8766  | 2a | — | — | — | — | — | — |
| 3  | 1.5 h  | 5844  | 2b | — | — | — | — | — | — |
| 4  | 2 h    | 4383  | 2a | ⚠ unstable (see below) | 402 | 8.89e+02 | 0.24 | — | 168276371 |
| 6  | 3 h    | 2922  | 2b | — | — | — | — | — | — |
| 9  | 4.5 h  | 1948  | 2b | — | — | — | — | — | — |
| 12 | 6 h    | 1461  | 2b | — | — | — | — | — | — |
| 18 | 9 h    | 974   | 2b | — | — | — | — | — | — |
| 36 | 18 h   | 487   | 2b | — | — | — | — | — | — |

Submission: `PARENT_MODEL=ACCESS-OM2-025 TIMESTEP_MULT=4 JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh`
on `gpuhopper` (1×H200, 256 GB) — single-GPU; monthly velocity preprocessing
already complete (no `vel` step needed). M=1 baseline still pending, so RMS Δ
will be left blank until a baseline run lands.

**Stability — M=4 (Δt=2h) is not safe on OM2-025**: the run returned exit 0
but `max(age)` blew up to **3.9×10⁸ yr** at sim iter 3294 (t=0.75 yr) at
`(i, j, k) = (143, 464, 45)` (subsurface western Pacific), then partially
"recovered" to a final `max(age) = 889 yr` — still wildly non-physical for
a passive tracer initialised at zero over a single year. The `mean` stays
near 0.24 yr because the explosion is spatially local. `validate_age_field`
catches NaN/Inf but not absurd magnitudes, so this run *was not caught* —
future work should add a magnitude sanity check (e.g. fail if `max(age)
> 100 yr` at t=1 yr). Julia-internal sim wall: 6m 42s.

The first plot job (168276372) hit the 30-min `WALLTIME_PLOT` after writing
10 PNGs + 4 of 10 MP4s; slice-depth animations were dropped. Initial
resubmit (168280065) was cancelled because `model_configs/ACCESS-OM2-025.sh:46`
hardcoded `WALLTIME_PLOT=00:30:00` and ignored the env override; the
config has since been fixed to use `${WALLTIME_PLOT:-00:30:00}`. Plot
resubmitted as **168280430** with `WALLTIME_PLOT=01:00:00`.

#### Timestepper comparison at M=4

The AB2 result above is unstable at `M=4` on OM2-025. Hypothesis: a
larger-stability-region timestepper might let us push Δt further before
the explosion. AB2 is 2nd-order multistep with a small absolute-stability
region tangent to the imaginary axis — barely stable for the
mildly-diffusive offline-tracer operator at large Δt. Split Runge-Kutta
methods (`SRK{N}`, N=2..5) have richer stability regions; SRK3 is the
natural first try.

Trade-off: each SRK3 step does ~3× the per-step work of AB2 (3 stages
per step). The win is only real if SRK3 stays stable at an `M` that
makes the per-year step count drop by > 3× — i.e., at `M ≥ 12` on
OM2-025. Otherwise SRK3 is slower than AB2 at `M=1`.

| `M` | TIMESTEPPER | Status | Sim wall (s) | Max age (yr) | Job ID |
|---|---|---|---|---|---|
| 4 | AB2  | ⚠ unstable (max=8.9e+02 yr at t=1yr; peak 3.9e+08 mid-run) | 402 | 8.89e+02 | 168276371 |
| 4 | SRK3 | ✓ stable | 484 | 1.97 | 168280609 |

**SRK3 fixes the OM2-025 M=4 instability**: max age drops from 889 yr →
1.97 yr — about a **450× reduction**, and now in line with what the
tracer should produce (max ~2 yr after 1 year of integration from
zero). Sim wall is 8m 4s (SRK3) vs 6m 42s (AB2) — only ~20% slower
end-to-end because I/O dominates at this resolution too.

This validates the AB2 → SRK3 hypothesis for OM2-025 as well. Natural
follow-ups: extend the SRK3 sweep to OM2-025 M ∈ {6, 9, 12, 18, 36}
to find the SRK3 wall, since AB2 already fails at M=4.

Submission: `TIMESTEPPER=SRK3 PARENT_MODEL=ACCESS-OM2-025 TIMESTEP_MULT=4 WALLTIME_PLOT=01:00:00 JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh`
on `gpuhopper` (1×H200). Output lands at the separate
`{MC}_DTx4` path where `{MC} = cgridtransports_wdiagnosed_centered2_SRK3`,
so it doesn't collide with the AB2 result above. Headline question: does
SRK3 *complete cleanly* (max age ≲ 5 yr at t=1yr) at `M=4`? If yes, try
`M=6 / M=12` to find the SRK3 wall and see if the speedup overcomes the
3× per-step cost.

### OM2-01 (Δt = 400 s baseline)

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1   | 6.67 min  | 78894 | 2a | — | — | — | — | 0 | — |
| 2   | 13.3 min  | 39447 | 2a | — | — | — | — | — | — |
| 3   | 20 min    | 26298 | 2b | — | — | — | — | — | — |
| 6   | 40 min    | 13149 | 2a | ⏳ queued (1×2) | — | — | — | — | 168280162 |
| 9   | 1 h       | 8766  | 2b | — | — | — | — | — | — |
| 18  | 2 h       | 4383  | 2b | — | — | — | — | — | — |
| 27  | 3 h       | 2922  | 2b | — | — | — | — | — | — |
| 54  | 6 h       | 1461  | 2b | — | — | — | — | — | — |
| 81  | 9 h       | 974   | 2b | — | — | — | — | — | — |
| 162 | 18 h      | 487   | 2b | — | — | — | — | — | — |

Submission: `PARTITION=1x2 PARENT_MODEL=ACCESS-OM2-01 TIMESTEP_MULT=6 JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh`
on `gpuhopper` (2×H200, 512 GB total). plot1yr = 168280163 chains afterok
run1yr. `W_FORMULATION=wdiagnosed` (default) means `w` is computed online
via continuity in [setup_model.jl:173-220](../src/setup_model.jl#L173-L220)
— no precomputed `w_diagnosed_monthly.jld2` is read, so
`u/v_from_mass_transport_monthly.jld2` + `eta_monthly.jld2` (all present)
are the only velocity inputs needed.

> **Submission history for this row** —
> 1. `diagnose_w-run1yr-plot1yr` (jobs 168276434/168276435/168276437):
>    cancelled mid-flight after clarification — `diagnose_w` is only
>    needed for `W_FORMULATION=wprescribed` + `PRESCRIBED_W_SOURCE=diagnosed`,
>    a different code path.
> 2. `run1yr-plot1yr` at `PARTITION=1x1` (jobs 168277463/168277464):
>    failed with **GPU OOM** — tried to allocate 79.3 GiB on top of
>    84 GiB already used on a single 140 GiB H200. OM2-01's 3600×2700×75
>    grid (729M cells) needs more than a single device. The driver default
>    of `PARTITION=1x1` is wrong for OM2-01 — it should default to at
>    least 1x2 for this model. Filed as a follow-up.
> 3. `run1yr-plot1yr` at `PARTITION=1x2` (jobs 168280162/168280163):
>    current attempt.

Stability sanity check is the headline question — RMS Δ vs M=1 stays
blank until a baseline lands.

### Conclusions

TBD after sweep completes.

### Known issues

- **`plot_standardrun_age.jl` unconditionally loads κV** (`kappa_v_monthly.jld2`),
  which currently fails when the on-disk halo size doesn't match the
  active `GRID_HZ`. The OM2-1 κV file dates from 2026-05-04 with `Hz=2`
  (parent z size 54), the current default is `GRID_HZ=7` (parent z size
  64) → `DimensionMismatch` at
  [plot_standardrun_age.jl:230](../src/plot_standardrun_age.jl#L230).
  This blocked every plot1yr submitted after the sweep (AB2 OM2-025
  resubmit + all 4 SRK3 plots). Possible fixes: (a) gate κV loading on
  `MONTHLY_KAPPAV=yes`, (b) repreprocess κV with current halo, (c)
  rebuild the κV `Field` with the active halo before pushing it into
  `field_specs`. Stability conclusions above are unaffected — they come
  from the run logs, not from the plot output.
