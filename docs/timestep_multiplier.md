# Timestep multiplier (`TIMESTEP_MULT`)

## Stability summary

What's been tested so far for the 1-year offline tracer simulation,
per (parent model, timestepper, multiplier). `✓` = ran cleanly with
max(age) ≲ a few years at t = 1 yr; `✗` = numerically unstable (NaN,
overflow, or wildly nonphysical max(age)); blank = not tested.
Untested cells are *not* claims about stability — many are valid
divisors of `N_base` that simply haven't been exercised yet.

After each `✓` the table shows the **theoretical speedup vs AB2 M=1
baseline** = `M / stage_count`, where `stage_count` is the number of
new RHS evaluations per step: AB2 = 1 (one new + one cached multistep
tendency), SRK2 = 2, SRK3 = 3, SRK4 = 4, SRK5 = 5. This is the ideal
the NK *inner* loop can reach (no per-step I/O). Empirically the
SRK-N cost is slightly worse than the textbook ratio
(`run1yr` measured SRK3 at 3.45× AB2 vs 3× theoretical) because of
kernel-launch and halo-exchange overhead per stage, but the integer
ratio is the right way to compare regimes.

### OM2-1 (Δt_base = 5400 s)

| M | Δt | AB2 | SRK2 | SRK3 | SRK4 | SRK5 |
|---|---|---|---|---|---|---|
| 1  | 1.5 h | ✓ (1×) |   |   |   |   |
| 2  | 3 h   | ✓ (2×) |   |   |   |   |
| 3  | 4.5 h |   |   |   |   |   |
| 4  | 6 h   | ✓ (4×) |   | ✓ (1.3×) |   |   |
| 6  | 9 h   | ✗ |   | ✓ (2×) |   |   |
| 12 | 18 h  | ✗ | ✗ | ✓ (4×) |   |   |

AB2-M=6 and AB2-M=12 were exercised as the 1-year inner sim of the
NK pipeline and both failed (M=6 crashed mid-Φ-call; M=12 stalled
with the residual blowing up). They weren't run as standalone
`run1yr`, but the integrator path is the same.

SRK2-M=12 (job 168365882) returned exit 0 but the simulation went
silently unstable: `max(age)` reached **3.4×10²³ yr** at t=1yr at
`(332, 175, 42)`, with `mean(age) ≈ −1.4×10¹⁵ yr` — signed garbage
(`validate_age_field`'s NaN/Inf-only check missed it). At Δt=18h on
OM2-1, CFL is ~0.65, so this is **not** CFL-limited — it's SRK2's
absolute-stability region being too small to reach Δt=18h for our
centered-2 + surface-relaxation operator. SRK3 reaches it; SRK2
doesn't.

**Recommendation — best tested speedup: 4×** (AB2-M=4 or SRK3-M=12,
tied). M=12 is the largest valid divisor of `N_base=5844` below the
practical 18-h cap, so the divisor structure caps any speedup at
`M=12 / stage_count`. SRK4-M=12 (3×) and SRK5-M=12 (2.4×) can't beat
that and aren't worth testing.

Use **SRK3-M=12** for the NK pipeline (only one stable at Δt=18h,
which the NK *inner* loop exploits with no per-step I/O); use
**AB2-M=4** for standalone runs.

To push past 4×: a different operator (`ADVECTION_SCHEME=weno5`) or
an implicit / IMEX integrator (not currently in tree).

### OM2-025 (Δt_base = 1800 s)

| M | Δt | AB2 | SRK2 | SRK3 | SRK4 | SRK5 |
|---|---|---|---|---|---|---|
| 1  | 30 min |   |   |   |   |   |
| 2  | 1 h    |   |   |   |   |   |
| 3  | 1.5 h  |   |   |   |   |   |
| 4  | 2 h    | ✗ |   | ✓ (1.3×) |   |   |
| 6  | 3 h    |   |   |   |   |   |
| 9  | 4.5 h  |   |   |   |   |   |
| 12 | 6 h    |   | ✗ | ✓ (4×) |   | ✓ (2.4×) |
| 18 | 9 h    |   | ✗ | ✗ |   | ✗ |
| 36 | 18 h   |   |   | ✗ |   | ✗ |

> [!WARNING]
> **OM2-025 Δt ≥ 6 h fails NK steady state for 1999-2008 with both SRK3 and AB2.**
> The ✓ for SRK3-M=12 in the table above reflects the *1-year `run1yr`* test only
> (no NK warm start, no Newton feedback). The full NK pipeline at Δt = 6 h for
> `TIME_WINDOW=1999-2008` fails in two different ways depending on the integrator:
>
> - **SRK3-M=12 (Δt=6h)** — IAF: NK exits 0 but the periodic age has scattered cells
>   at `|age| ≈ 1.7 × 10⁶¹ yr` (discovered when `compare_NK_ages.jl` tried to plot it,
>   2026-05-18; NK's RMS norm passed but `check_age_field` rejected the per-cell max).
>   TRAF (Attempt 2 in [TRAF_simulations.md](TRAF_simulations.md)): forward Φ map blows up
>   at `(i=1288, j=1047, k≈36)` (high-latitude near the tripolar fold) reaching
>   `max(age) ≈ 10⁶⁸ yr` inside Φ! call #1; NK retcode `Stalled`, mean age `2.4e-18 yr`.
> - **AB2-M=4 (Δt=6h)** — TRAF (Attempt 3e): same fold-region blow-up. Smaller Δt
>   (2 vs 6 h) does NOT save it. NK retcode `Stalled` at residual `~10¹⁰⁷`.
>
> Both the 1968-1977 TW converges fine at Δt = 6 h for either integrator; the
> 1999-2008 forcing is energetic enough at this resolution to push Δt = 6 h
> past its stability margin at the polar grid singularity.
>
> **Resolution.** TRAF Attempt 3f (2026-05-18) ran the full NK pipeline at smaller Δt
> for both timesteppers and both TWs — all converged:
>
> | Δt config | OM2-025/1968-1977 mean (yr) | OM2-025/1999-2008 mean (yr) |
> |---|---:|---:|
> | SRK3-M=12 (Δt=6 h)  | 892.89 ✓     | garbage (stalled) |
> | AB2-M=4  (Δt=6 h)   | 872.01 ✓     | 1048.87 (stalled) |
> | **SRK3-M=9** (Δt=4.5 h) | **885.88 ✓** | **933.39 ✓** |
> | **AB2-M=3**  (Δt=1.5 h) | **868.80 ✓** | **917.35 ✓** |
>
> See [TRAF_simulations.md § Attempt 3f](TRAF_simulations.md#attempt-3f--om2-025-ab2-m3--srk3-m9-2026-05-18-commits-5e69399--earlier),
> [IAF_NK_age_comparison_plan.md § Known issue](IAF_NK_age_comparison_plan.md#known-issue--om2-025--1999-2008-instability),
> and [timestep_multiplier_NK.md § OM2-025](timestep_multiplier_NK.md#om2-025-nk-steady-state-instability-at-srk3-m12-2026-05-18).
>
> **Moral**: 1-year `run1yr` stability is **necessary but not sufficient** for NK
> validity. The NK fixed-point amplifies per-step truncation error over the
> ventilation timescale (~900 yr), so a marginally-stable Δt for a single year
> can be catastrophically unstable inside a 60+ Newton-iteration solve. The
> stability table above measures the *forward map*, not the NK pipeline.

**Recommendation — best tested speedup: 3×** (SRK3-M=9, Δt=4.5h). The
SRK3-M=12 4× claim above held for the 1-year `run1yr` test only; in
the full NK pipeline Δt=6 h fails for the 1999-2008 forcing window
(see warning callout above and [TRAF_simulations.md § Attempt 3f](TRAF_simulations.md#attempt-3f--om2-025-ab2-m3--srk3-m9-2026-05-18-commits-5e69399--earlier)
for the NK-converged numbers at Δt=4.5h). For NK production runs
use **SRK3-M=9** (3× speedup, the 1968-1977 mean age is 885.88 yr and
the 1999-2008 mean is 933.39 yr). AB2-M=3 (1× per-step speedup, Δt=1.5h)
is the conservative fallback if SRK3-M=9 ever flakes — both TWs also
converged there (868.80 / 917.35 yr).

Key findings from the SRK{2,3,5} × M ∈ {12, 18} 1-year sweep:

- **SRK2 fails everywhere** even at M=12 (CFL ~0.86); SRK2's
  absolute-stability region is too narrow for centered-2 + surface
  relaxation at any Δt > Δt_base.
- **No SRK-N reaches Δt=9h** — SRK3-M=18 and SRK5-M=18 both blow up
  at the same grid point (1288, 1047, ~35), an equatorial-jet region
  where local CFL exceeds the global average. More stages don't
  help past this point. (Same grid point that fails the *NK* at
  Δt=6 h for 1999-2008 — see callout — so the NK steady state hits
  its stability wall at a smaller Δt than the standalone forward map.)
- **M=12 is the 1-year-stability wall.** `N_base=17532` has no divisor
  between 12 and 18, so the integer-divisor structure plus the
  integrator-independent Δt=9h wall caps explicit standalone speedup
  at `M_max_stable / stage_count_min = 12 / 3 = 4×`. SRK4-M=12 (3×)
  isn't worth testing. **But** the NK pipeline's effective wall is
  one divisor step tighter — M=9 (Δt=4.5h) for SRK3 — see callout.

To push past 3×-4×: same as OM2-1 — different operator (WENO5) or
implicit / IMEX integrator (not currently in tree).

### OM2-01 (Δt_base = 400 s)

| M | Δt | AB2 | SRK2 | SRK3 | SRK4 | SRK5 |
|---|---|---|---|---|---|---|
| 1   | 6.67 min |   |   |   |   |   |
| 2   | 13.3 min | ✓ (2×) |   |   |   |   |
| 3   | 20 min   | ✗ |   | ✓ (1×) |   |   |
| 6   | 40 min   | ✗ |   | ✗ |   | ✓ (1.2×) |
| 9   | 1 h      |   |   | ✗ |   | ✗ |
| 18  | 2 h      |   |   | ✗ |   |   |
| 27  | 3 h      |   |   |   |   |   |
| 54  | 6 h      |   |   |   |   |   |
| 81  | 9 h      |   |   |   |   |   |
| 162 | 18 h     |   |   |   |   |   |

**Recommendation — best tested speedup: 2×** (AB2-M=2, Δt = 13.3 min).
Each integrator has exactly one stable M on OM2-01 (AB2: M=2; SRK3:
M=3; SRK5: M=6) — moving up one valid divisor blows up every time —
and AB2-M=2 wins on per-step efficiency:

| Integrator | Stable M | Speedup = M / stage_count |
|---|---|---|
| AB2  | 2 | **2×** |
| SRK3 | 3 | 1× |
| SRK5 | 6 | 1.2× |

Use AB2-M=2 for production runs and as the NK 1-year forward map.

CFL is not the bottleneck — SRK3-M=9 (Δt=1h, CFL_x ≈ 0.36) and SRK5-M=9
both fail. The ceiling is the surface / implicit-κV absolute-stability
footprint tightening with finer Δx; stage-count increases buy roughly
one factor of Δt each (M=2 → 3 → 6) but not more.

The pattern across resolutions:

| Model | Best stable point (1-year `run1yr`) | Best stable point (NK pipeline) | Effective speedup |
|---|---|---|---|
| OM2-1   | AB2-M=4 or SRK3-M=12 | same | 4× |
| OM2-025 | SRK3-M=12 (1968-1977 NK works; 1999-2008 NK fails) | **SRK3-M=9** (both TWs converge) | 3× |
| OM2-01  | **AB2-M=2**          | not yet tested | **2×** |

The OM2-025 row collapses two different stability walls into one table.
Standalone `run1yr` at SRK3-M=12 (Δt=6 h) is fine for both 1968-1977
and 1999-2008, but the NK fixed-point amplifies per-step truncation
error over ~900-yr ventilation timescales and the 1999-2008 forcing
pushes Δt=6 h past its NK margin at the tripolar fold. SRK3-M=9
(Δt=4.5 h) converges cleanly in both TWs under both IAF and TRAF
(see [TRAF_simulations.md § Attempt 3f](TRAF_simulations.md#attempt-3f--om2-025-ab2-m3--srk3-m9-2026-05-18-commits-5e69399--earlier)).

For a step-change speedup, an operator change is needed:
`ADVECTION_SCHEME=weno5` adds numerical diffusion (enlarging the
stable region), or an implicit / IMEX integrator (not currently in
tree) would push past the explicit absolute-stability bound.

M=4 and M=8 are not valid divisors of `N_base = 2 · 3⁴ · 487`.

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

The NK pipeline at M=6 / M=12 with the default AB2 timestepper failed
([NK doc](timestep_multiplier_NK.md#failure-modes)). SRK3's richer
absolute-stability region is the natural retry. Per-step cost is 3× AB2
(3 stages vs 1 cached tendency), so the net win needs M ≥ 12 to beat
AB2-M=4.

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

SRK3 completed cleanly at M ∈ {4, 6, 12}, including the M=6 and M=12
cases where AB2 inside the NK pipeline blew up. Sim wall drops
monotonically with M (100.4 → 90.1 → 79.4 s); SRK3-M=12 is tied with
AB2-M=4 on wall (79.4 vs 78.1 s) but Δt=18h, so it's a free upgrade
for the NK inner loop where per-step I/O vanishes.

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

Submission: `PARENT_MODEL=ACCESS-OM2-025 TIMESTEP_MULT=<M> [TIMESTEPPER=<TS>] JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh`
on `gpuhopper` (1×H200, 256 GB). M=1 baseline still pending, so RMS Δ
stays blank until that lands.

**AB2-M=4 is not safe on OM2-025**: returned exit 0 but `max(age)`
blew up to 3.9×10⁸ yr at sim iter 3294 then partially "recovered" to a
final 889 yr — still wildly non-physical. `validate_age_field` catches
NaN/Inf but not absurd magnitudes, so this slipped through — future
work should add a magnitude sanity check (e.g. fail if `max(age) > 100
yr` at t=1 yr).

#### Timestepper comparison

| `M` | Δt | TIMESTEPPER | Status | Sim wall (s) | Max age (yr) | Job ID |
|---|---|---|---|---|---|---|
| 4  | 2 h  | AB2  | ✗ unstable (max=8.9e+02 yr) | 402 | 8.89e+02 | 168276371 |
| 4  | 2 h  | SRK3 | ✓                            | 484 | 1.97 | 168280609 |
| 12 | 6 h  | SRK2 | ✗                            | —   | —    | 168363327 |
| 12 | 6 h  | SRK3 | ✓                            | 253 | 1.73 | 168283651 |
| 12 | 6 h  | SRK5 | ✓                            | —   | —    | 168363332 |
| 18 | 9 h  | SRK2 | ✗                            | —   | —    | 168363330 |
| 18 | 9 h  | SRK3 | ✗                            | —   | —    | 168363323 |
| 18 | 9 h  | SRK5 | ✗                            | —   | —    | 168363335 |
| 36 | 18 h | SRK3 | ✗                            | 130 (DNF) | NaN | 168283653 |
| 36 | 18 h | SRK5 | ✗                            | 137 (DNF) | NaN | 168286118 |

SRK3 fixes the M=4 AB2 instability and extends cleanly to M=12 — max
age stays in the ~2 yr range across M ∈ {4, 12}, sim wall drops from
484 s (M=4) to 253 s (M=12) — almost the full 3× per-step speedup, I/O
overhead ~70 s.

The SRK3 wall sits between M=12 and M=36. SRK5-M=36 also fails (same
grid point as SRK3-M=36), ruling out *timestepper stability* as the
bottleneck at Δt=18h. CFL at Δt=18h is ~2.6 — centered-2 is
unconditionally unstable for CFL > 1, so the ceiling is the velocity
field, not the integrator. SRK3-M=18 and SRK5-M=18 also fail (CFL
~1.3) at the same equatorial-jet grid point.

**M=12 (SRK3, Δt=6h) is the OM2-025 production sweet spot.** To push
past 4× the operator must change: `ADVECTION_SCHEME=weno5` (WENO's
numerical diffusion stabilizes CFL > 1) or an IMEX integrator (not in
tree).

### OM2-01 (Δt = 400 s baseline)

AB2 table:

| `M` | Δt | Steps/yr | Stage | Status | Wall time (s) | Max age (yr) | Mean age (yr) | RMS Δ vs M=1 (yr) | Job ID |
|---|---|---|---|---|---|---|---|---|---|
| 1   | 6.67 min  | 78894 | 2a | — | — | — | — | 0 | — |
| 2   | 13.3 min  | 39447 | 2a | ✓ exit 0               | 6124 (1.701 h) | 1.319 | — | — | 168574394 |
| 3   | 20 min    | 26298 | 2b | ✗ NaN at iter 1600     | — | NaN | — | — | 168482446 |
| 6   | 40 min    | 13149 | 2a | ✗ NaN at iter 600      | — | NaN | — | — | 168280162 / 168482509 |
| 9   | 1 h       | 8766  | 2b | — | — | — | — | — | — |
| 18  | 2 h       | 4383  | 2b | — | — | — | — | — | — |
| 27  | 3 h       | 2922  | 2b | — | — | — | — | — | — |
| 54  | 6 h       | 1461  | 2b | — | — | — | — | — | — |
| 81  | 9 h       | 974   | 2b | — | — | — | — | — | — |
| 162 | 18 h      | 487   | 2b | — | — | — | — | — | — |

OM2-01 requires `PARTITION=1x2` (won't fit on a single H200).
Submission template:

```bash
PARTITION=1x2 PARENT_MODEL=ACCESS-OM2-01 TIMESTEPPER=<TS> TIMESTEP_MULT=<M> \
  JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
```

RMS Δ vs M=1 stays blank until a baseline lands.

#### Timestepper comparison

AB2 fails on OM2-01 at every tested M > 1, so the focus is the SRK
sweep across `M ∈ {3, 6, 9, 18}` plus the in-flight AB2-M=2 / SRK5
probes.

| `M` | Δt | TIMESTEPPER | Status | Sim wall (s) | Max age (yr) | Job ID |
|---|---|---|---|---|---|---|
| 2  | 13.3 min | AB2  | ✓ exit 0               | 6124 (1.701 h)  | 1.319 | 168574394 |
| 3  | 20 min   | AB2  | ✗ NaN at iter 1600     | — | NaN | 168482446 |
| 3  | 20 min   | SRK3 | ✓ exit 0               | 11106 (3.085 h) | 1.305 | 168482506 |
| 6  | 40 min   | AB2  | ✗ NaN at iter 600      | — | NaN | 168482509 |
| 6  | 40 min   | SRK3 | ✗ NaN at iter 12900    | — | NaN | 168482512 |
| 6  | 40 min   | SRK5 | ✓ exit 0               | 9320 (2.589 h)  | 1.297 | 168574437 |
| 9  | 1 h      | SRK3 | ✗ NaN at iter 500      | — | NaN | 168482517 |
| 9  | 1 h      | SRK5 | ✗ NaN at iter 800      | — | NaN | 168574832 |
| 18 | 2 h      | SRK3 | ✗ NaN at iter 200      | — | NaN | 168482520 |

##### Verdict

Each integrator has one stable M and fails at the next valid divisor:
AB2-M=2 ✓ / AB2-M=3 ✗; SRK3-M=3 ✓ / SRK3-M=6 ✗; SRK5-M=6 ✓ / SRK5-M=9 ✗.
AB2-M=2 (2× theoretical speedup) is the production sweet spot and the
recommended NK 1-year forward map.

**Note on the NaN-hang**: NaN-stopped distributed runs deadlock and
eat the full walltime. Filed as a follow-up; set a shorter walltime
for stability probes in the meantime.

### Conclusions

TBD after sweep completes.

### Known issues

- *(resolved)* `kappa_v_monthly.jld2` was retired when `MONTHLY_KAPPAV=yes`
  switched to deriving κV from a 2D monthly MLD on the fly (see commit
  history around the MLD κV refactor). `plot_standardrun_age.jl` already
  uses `_try_load_diag_fts`, so it silently skips the 200 m κV diagnostic
  now that the file no longer exists. Reviving that diagnostic from MLD
  is a separate follow-up.
