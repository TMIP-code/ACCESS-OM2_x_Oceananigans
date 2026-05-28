# NK at OM2-01 — handoff (v2)

A focused continuation of [NK_OM2-01.md](NK_OM2-01.md). That earlier doc captures the full
history (TMbuild OOMs/walltime fights, run1yr JLD2 collision, etc.) and the
`LUMP_AND_SPRAY=AxB` refactor that landed. This file lists only **what's
running, what's left, and what's needed to submit** so a fresh session can
pick up cleanly.

## In flight

`169283378` — OM2-01 NK_5x5 (gpuhopper 1×4, defaults via env_defaults.sh
after the refactor: wprescribed/parent/cgridtransports/Q5x5/M=2/LBS).
**Running**, but the G! residual reports astronomically large drift
norms — see "Open issue" below.

## Open issue — OM2-01 G! residual is ~10⁵⁹ years at iteration 1

Active OM2-01 NK (`169283378`) shows:

```
G! n=1: vol_rms_drift_years = 1.5e+59   max_drift_years = 4.7e+61
G! n=2: vol_rms_drift_years = 9.6e+49   max_drift_years = 2.8e+52
```

Compare the same NK code on OM2-1 / 1×4 / LBS (`169413550`,
`totaltransport`/M=4/Q2x2, also on Distributed{GPU} so MCBC bug is
exercised in the same way):

```
G! n=1: vol_rms_drift_years = 0.98      max_drift_years = 1.61
G! n=2: vol_rms_drift_years = 0.033     max_drift_years = 1.83
G! n=3: vol_rms_drift_years = 3.9e-6    max_drift_years = 0.001
```

So OM2-1 looks textbook, OM2-01 is ~60 orders of magnitude wrong.

NonlinearSolve.jl's own residual norm matches the order of magnitude
(`1.47e+69` at iter 0), and the iterate `x₁` recovered from Pardiso's
preconditioned step has *bounded* values (`norm ≈ 19,449 years`,
`max ≈ 477 years`), suggesting either:

1. **Uninitialized memory in the `dage` buffer** that `gather!`
   doesn't fully overwrite — small number of indices retain
   garbage ~10⁶¹ from the initial `similar(u0)` alloc. The bulk of the
   field is sensible; Pardiso filters the bad indices when computing
   `Δx`, so the iterate stays bounded.

2. **Genuine numerical blowup** concentrated in some region (tripolar
   fold corner, shallow cells near steep bathymetry…) under the
   aggressive `Δt = 800 s` (M=2 at 0.1°). Pardiso could still filter
   it the same way, masking the issue at the iterate level but not
   in the raw residual norm.

Both hypotheses produce the same `max ≫ mean` signature, so the
diagnostic numbers alone can't distinguish them.

### Decisive result (2026-05-28) — genuine instability, not a gather bug

OM2-01 1-year forward run with same env as NK (`169563958`,
exit 0, 54 min on gpuhopper 1x4) reports per-rank drift stats that
isolate the blowup geographically:

| Rank | Nidx_local | max_drift (yr) | mean_drift (yr) | Verdict |
|---|---|---|---|---|
| 0 | 95.0 M | 1.35 | 0.53 | **healthy** |
| 1 | 97.4 M | 9.8 × 10³⁰ | 4.8 × 10²⁴ | blown up |
| 2 | 96.1 M | 4.7 × 10⁶¹ | 2.0 × 10⁵⁷ | blown up (most severe) |
| 3 | 63.1 M | 1.4 × 10⁵ | 6.0 | partial — max blown, mean ~OK |

OM2-1 1x4 sanity check (`169563926`) for comparison: all 4 ranks
show `max ∈ [1.2, 1.6] yr`, `mean ∈ [0.66, 0.76] yr` — clean
distributed forward map.

So the gather/perm hypothesis is dead. The blowup is **in the field
values themselves** in 2-3 ranks' subdomains. Rank 0 (bottom y-band)
is fine, rank 3 (top y-band incl. tripolar fold) has only large `max`
with bounded mean, ranks 1-2 (mid latitudes / tropics) have full mean
contamination.

**Best guess: CFL instability at Δt=800 s** (M=2 at 0.1°). Tropics and
mid-latitudes have the smallest dy + largest velocities at this
resolution; tripolar fold area is more sensitive to the fold halo
itself (large max but bulk fine).

The NK run (`169283378`) hit walltime (24:01:06, exit -29). Pardiso
absorbed the garbage well enough to keep iterates bounded but the
solver never converged.

### Next steps

1. **Drop `TIMESTEP_MULT` to 1** and rerun the 1-year forward to
   confirm CFL:
   ```bash
   PARENT_MODEL=ACCESS-OM2-01 TIMESTEP_MULT=1 JOB_CHAIN=run1yr bash scripts/driver.sh
   ```
   - If max_drift ~1 yr on all 4 ranks → CFL is the cause; bake
     `TIMESTEP_MULT=1` into [model_configs/ACCESS-OM2-01.sh](../model_configs/ACCESS-OM2-01.sh).
   - If still blown → not CFL. Look at velocity field magnitudes,
     check cgridtransports interpolation near rank 1/2 boundaries.
2. If M=1 works, rebuild the OM2-01 transport matrix at DTx1
   (`JOB_CHAIN=TMbuild` with the new default) before resubmitting NK.
3. Resubmit NK at M=1 and watch G! n=1 residual — should now be ~1 yr
   instead of 10⁵⁹.

### Pre-existing scaffolding for future sessions

- Drift print already in [src/run_1year.jl](../src/run_1year.jl)
  (mirrors G!'s vol_norm / max(abs) / mean(abs) definitions, gated on
  `INITIAL_AGE=0`).
- `vol_norm` defined in [src/shared_utils/grid.jl:668](../src/shared_utils/grid.jl#L668).
- `G!` itself at [src/periodic_solver_common.jl:391](../src/periodic_solver_common.jl#L391).

### Original (pre-result) debugging plan — keep for reference

Cheapest discriminating test: **a single 1-year forward run with the
exact same env as the NK job**, with per-rank drift stats printed at
the end (mirroring G!). The run is ~30 min instead of 24 h.

The drift print landed in [src/run_1year.jl](../src/run_1year.jl) — for
`INITIAL_AGE=0` it computes
`drift = age_final - 0 = age_final` and reports
`vol_rms_drift_years`, `max_drift_years`, `mean_drift_years`
(and `Nidx_local`) per rank, using the same `vol_norm` /
`maximum(abs, ⋅)` / `mean(abs, ⋅)` definitions as `G!` in
[periodic_solver_common.jl:396](../src/periodic_solver_common.jl#L396).

Sequence:

1. ✅ Add the drift print to `run_1year.jl`.
2. ▢ Submit an OM2-1 1-year run (cheap sanity check; should reproduce
   the ~1-year drift seen at n=1 of the OM2-1 NK).
3. ▢ Submit an OM2-01 1-year run with the **same env as the running
   NK job** (`169283378`):
   ```bash
   PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=run1yr bash scripts/driver.sh
   ```
   (PARTITION=1x4 LOAD_BALANCE=surface VELOCITY_SOURCE=cgridtransports
   TIMESTEP_MULT=2 W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=parent
   come from defaults.)
4. ▢ Inspect the per-rank `max_drift_years` from step 3:
   - If **finite/sensible** (~1 yr on each rank) → the forward map is
     fine; bug is in `gather!`/`perm` at OM2-01 scale.
   - If **astronomical** on at least one rank → genuine numerical
     blowup in `Φ!`. Look at which rank(s); inspect that region's
     velocities, bathymetry, tripolar fold proximity.
5. ▢ Based on (4), pick the surgical fix:
   - Gather bug → assert `sort(perm) == 1:Nidx_global` in
     `build_global_permutation`; if it fires, fix `build_global_permutation`.
   - Instability → reduce Δt (drop `TIMESTEP_MULT` from 2 → 1) and/or
     investigate seam handling for cgridtransports at 0.1°.

### Related observations

- OM2-1 1×4 forward map under the *same* NK code is healthy
  (`169413550` exit 0, NK converged). So the Distributed{GPU} 1×4 +
  LBS code path is fine in general — the failure is OM2-01-specific.
- The active OM2-01 NK is likely to continue iterating without
  crashing (Pardiso absorbs the garbage), but if convergence test
  uses absolute residual it'll never trigger; if relative, it may
  appear to converge to a wrong solution.

`M.jld2` (39 GB) lives at
`outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/const/M.jld2`.

## Known bug — `compute_wet_mask` crashes on `Distributed{GPU}` grids at OM2-01

`169183483` (NK_5x5 at OM2-01, `PARTITION=1x4`) exited 1 after ~18 min, 0%
GPU utilisation. It crashes in `compute_wet_mask` at
[src/shared_utils/grid.jl:630](../src/shared_utils/grid.jl#L630):

```julia
wet3D = .!isnan.(interior(on_architecture(CPU(), fNaN)))
```

called from [src/periodic_solver_common.jl:48](../src/periodic_solver_common.jl#L48)
via [src/solve_periodic_NK.jl:126](../src/solve_periodic_NK.jl#L126).

`on_architecture(CPU(), fNaN)` rebuilds the field+grid on CPU. The
rebuilt CPU grid loses the per-rank connectivity of the
`Distributed{GPU{CUDABackend}, …, Partition{…NTuple{4,Int64}…}}`
grid, so the `MultiRegionCommunication` south BC ends up with `Nothing`
where a connectivity object with a `.rank` field is expected. Then
`materialize_immersed_boundary` calls `fill_halo_regions!` on it:

```
FieldError: type Nothing has no field `rank`
@ Oceananigans.MultiRegion .../multi_region_boundary_conditions.jl:138
```

**Why we hadn't seen it.** A prior partitioned NK *did* succeed:
`169107705` (NK_restart_A1, OM2-1, `1x2`, gpuvolta, exit 0, 21 m, 18 GB —
2026-05-23). The `compute_wet_mask` function dates from commit `e445a9f`
(2026-03-19) and was unchanged between then and the OM2-01 crash. So the
trigger is either **OM2-01 grid size** or the jump to **1x4 partition**
(possibly both). Nothing in `grid.jl` / `periodic_solver_common.jl` /
`setup_model.jl` changed between the working OM2-1 1x2 run and the
failing OM2-01 1x4 run.

**Proposed fix** — skip the grid rebuild; copy just the array data:

```julia
wet3D = .!isnan.(Array(interior(fNaN)))
```

`Array(::SubArray{T,N,<:CuArray})` is the canonical CUDA.jl host copy,
so `wet3D` stays on CPU exactly like before — we just avoid materialising
a CPU twin of the distributed GPU grid.

**Validation plan before re-running OM2-01.** Iterate on OM2-1 / `1x2` /
gpuvolta first (cheap, fast, and known-good baseline: `169107705`).
Submission:

```bash
PARENT_MODEL=ACCESS-OM2-1 \
  PARTITION=1x2 LOAD_BALANCE=no \
  LUMP_AND_SPRAY=no \
  JOB_CHAIN=NK bash scripts/driver.sh
```

If that succeeds with the fix in place we move back to OM2-01 / `1x4` /
gpuhopper. If it still fails, the bug is more general than just the OM2-01
grid and we can diagnose against the much cheaper OM2-1 setup.

## Remaining work

1. **Wait for `169183483` to land** and record: walltime, peak rank-0 RSS,
   number of Newton iterations, GMRES rate. Update [NK_OM2-01.md](NK_OM2-01.md)
   outcome log.
2. **LB sweep at OM2-01 (wparent) via nsys profiles** (not full 1-year runs).
   See [docs/profiling_workflow.md](profiling_workflow.md) for the canonical
   workflow — `PROFILE=yes` + `BENCHMARK_STEPS=200` on a `run1yrfast`
   driver invocation produces a 200-step nsys-traced run per LB variant.
   The driver already plumbs everything through.

   ```bash
   PARENT_MODEL=ACCESS-OM2-01 \
     PARTITION=1x4 LOAD_BALANCE=<variant> \
     LUMP_AND_SPRAY=5x5 \
     PROFILE=yes BENCHMARK_STEPS=200 \
     JOB_CHAIN=run1yrfast bash scripts/driver.sh
   ```

   (`W_FORMULATION=wprescribed`, `PRESCRIBED_W_SOURCE=parent`,
   `TIMESTEP_MULT=2`, `MONTHLY_KAPPAV=yes`, `TM_SOURCE=const`,
   `TRACE_SOLVER_HISTORY=yes` are all defaults after the recent
   env/model-config changes — no need to set them.)

   LB variants to run (in priority order):
   - `surface` (LBS) — partition `1x4_LBS` ✓ correct halo
   - `no` — partition `1x4` ✓ rebuilt (`169132252`)
   - `cell` (LB) — partition `1x4_LB` ✓ rebuilt (`169132264`)

   Skip `mix` and `minmax` for now (partitions not built; not worth the
   extra megamem build at this stage).

   **Why re-sweep**: the existing OM2-01 LB results in
   [docs/profiling_results_v2.md § Phase 3](profiling_results_v2.md#phase-3-om2-01--h200-gpuhopper)
   were measured under `W_FORMULATION=wdiagnosed`. The
   `compute_w_from_continuity!` kernel is about **25%** of step cost
   under wdiagnosed — not the dominant term, but not negligible either.
   Switching to `wparent` drops it entirely, so the LB ranking may
   shift; the wparent sweep is what decides the production LB choice.
3. **Sweep finer/coarser if 5×5 stalls** (per [NK_OM2-01.md](NK_OM2-01.md)
   § Coarsening sweep): walk to `4x4`, `3x3` if preconditioner too weak;
   `6x6`, `7x7` if memory/time pressure (hard floor at `3x3`; `2x2` is
   infeasible).

## How to submit (full env-var set)

After the defaults landed in `scripts/env_defaults.sh` and
`model_configs/*.sh` (this session, see "Defaults that landed" below),
the OM2-01 NK submission collapses to:

```bash
PARENT_MODEL=ACCESS-OM2-01 \
  PARTITION=1x4 LOAD_BALANCE=surface \
  LUMP_AND_SPRAY=5x5 \
  JOB_CHAIN=NK \
  bash scripts/driver.sh
```

Every other relevant variable comes from defaults:

| Var | Default source | Value |
|---|---|---|
| `VELOCITY_SOURCE` | env_defaults | `cgridtransports` |
| `W_FORMULATION` | env_defaults | `wprescribed` (was `wdiagnosed`) |
| `PRESCRIBED_W_SOURCE` | env_defaults | `parent` |
| `ADVECTION_SCHEME` | env_defaults | `centered2` |
| `TIMESTEPPER` | env_defaults | `AB2` |
| `TIMESTEP_MULT` | model_config | `2` for OM2-01 (4 for OM2-1, 3 for OM2-025) |
| `MONTHLY_KAPPAV` | env_defaults | `yes` |
| `MATRIX_PROCESSING` | env_defaults | `symdrop` |
| `LINEAR_SOLVER` | env_defaults | `Pardiso` |
| `TM_SOURCE` | env_defaults | `const` (was `avg`) |
| `TRACE_SOLVER_HISTORY` | env_defaults | `yes` (was `no`) |
| `INITIAL_AGE` | env_defaults | `0` (use `latest` on restart) |
| `GPU_QUEUE` | OM2-01 model_config | `gpuhopper` |

## Defaults that landed this session

`scripts/env_defaults.sh`:

- `W_FORMULATION`: `wdiagnosed` → **`wprescribed`** (with `PRESCRIBED_W_SOURCE=parent` unchanged).
- `TM_SOURCE`: `avg` → **`const`**.
- `TRACE_SOLVER_HISTORY`: `no` → **`yes`** (needed for `INITIAL_AGE=latest` restarts).
- Model config is now sourced **before** the variable defaults so per-model
  defaults (e.g. `TIMESTEP_MULT`) win over the cross-model fallbacks. User
  env-var overrides still win over everything.

`model_configs/ACCESS-OM2-1.sh`:

- `TIMESTEP_MULT=${TIMESTEP_MULT:-4}` (was implicit 1).

`model_configs/ACCESS-OM2-025.sh`:

- `TIMESTEP_MULT=${TIMESTEP_MULT:-3}` (was implicit 1).

`model_configs/ACCESS-OM2-01.sh` (in addition to the TMbuild bumps from
earlier in the session):

- `TIMESTEP_MULT=${TIMESTEP_MULT:-2}` (was implicit 1).

`src/create_matrix.jl`:

- Dropped the `M_spy.png` post-save plot (was crashing at OM2-01 scale
  and producing exit-1, breaking `JOB_CHAIN=TMbuild-NK` chains).

## Still left to land (TBD)

- `LOAD_BALANCE` default — leave at `no` until the wparent LB sweep
  (item 2 above) tells us which variant wins. Then bake that into the
  model configs (probably per-model: OM2-1/025/01 may not agree).
- `PARTITION` default — currently `1x1`. OM2-01 effectively needs `1x4`
  on H200; could bake per-model. Less urgent — the driver already
  errors clearly if NGPUS=1 is used at OM2-01.

## Pointers

- Full history + outcome log: [docs/NK_OM2-01.md](NK_OM2-01.md).
- Profiling workflow + nsys mechanics: [docs/profiling_workflow.md](profiling_workflow.md).
- Existing LB-sweep results under `wdiagnosed` (the comparison baseline
  for the upcoming wparent sweep): [docs/profiling_results_v2.md](profiling_results_v2.md).
- TMbuild scaling note (OM2-025 → OM2-01) is in [NK_OM2-01.md](NK_OM2-01.md)
  § Context. Sparsity detection scales ~18× (vs ~10× for everything else)
  due to GC pressure — keep an eye on this if matrix size grows further.
- Submissions index: `scripts/runs/submissions.tsv` (use
  `scripts/runs/reconcile_submissions.sh` to fill PBS-side columns after
  jobs finish).
