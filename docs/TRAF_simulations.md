# TRAF_simulations.md — periodic NK adjoint age across TW × resolution

## Context

The companion to [IAF_simulations.md](IAF_simulations.md). Same 2×2 (PM × TW) grid, same model physics, **but with the Time-Reversed Adjoint Flow flag `TRAF=yes`** so we compute the adjoint age (a.k.a. "time to re-emergence", Holzer & Hall 2000 / Khatiwala 2003) under each circulation. Together with the forward NK ages from the IAF runs, this gives us surface ventilation fractions via `V↓ = A⁻¹ V (age_traf / sink_timescale)` (Pasquier *et al.* 2024, *JGR-Oceans*, doi:10.1029/2024JC021043).

|              | TW=1968-1977 | TW=1999-2008 |
|--------------|--------------|--------------|
| OM2-1        | run 1        | run 2        |
| OM2-025      | run 3        | run 4        |

**Hard dependency**: each TRAF run reads the corresponding **forward** `M.jld2` (built in the IAF pipeline) and synthesizes `invVMtV.jld2 = V⁻¹ Mᵀ V` in the `_traf` directory — so the four IAF runs (or at least their `TMbuild` step) must complete before the TRAF jobs start. No new preprocessing or velocity-field generation is needed: the existing `preprocessed_inputs/{PM}/{EXP}/{TW}/` tree is reused as-is, with the FTS reversal/sign-flip applied in memory at load time.

## Agreed configuration (per-run ENV)

The unified MODEL_CONFIG tag is `totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12_traf` — i.e. the IAF tag with a `_traf` suffix automatically appended by `env_defaults.sh` + `build_model_config(...)` when `TRAF=yes`.

### Per-run varying
| ENV var | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| `PARENT_MODEL` | ACCESS-OM2-1 | ACCESS-OM2-1 | ACCESS-OM2-025 | ACCESS-OM2-025 |
| `TIME_WINDOW` | 1968-1977 | 1999-2008 | 1968-1977 | 1999-2008 |
| `GPU_QUEUE` (model default) | gpuvolta | gpuvolta | gpuhopper | gpuhopper |
| `EXPERIMENT` (auto from PM) | 1deg_jra55_iaf_omip2_cycle6 | (same) | 025deg_jra55_iaf_omip2_cycle6 | (same) |

### Shared across all four runs
| ENV var | Value | Why |
|---|---|---|
| `TRAF` | `yes` | Time-Reversed Adjoint Flow: reverse every monthly FTS in time + sign-flip u, v |
| `TRAF_TM_SOURCE` | `invVMtV` (default) | Synthesize `M_traf = V⁻¹ Mᵀ V` from the forward M instead of rebuilding via autodiff |
| `VELOCITY_SOURCE` | `totaltransport` | (same as IAF) |
| `W_FORMULATION` | `wdiagnosed` | **Required** under `TRAF=yes` in first cut — w is recomputed from continuity using reversed u, v, η, which is automatically the adjoint w |
| `ADVECTION_SCHEME` | `centered2` | (same as IAF) |
| `TIMESTEPPER` | `SRK3` | (same as IAF) |
| `TIMESTEP_MULT` | `12` | (same as IAF) |
| `MONTHLY_KAPPAV` | `yes` | κV monthly FTS is also time-reversed (no sign flip) by `reverse_fts_time!` |
| `PARTITION` | `1x1` | Serial (1 GPU) |
| `TM_SOURCE` | `const` | **Required** under `TRAF=yes` in first cut (snapshot/avg path not yet TRAF-aware) |
| `LINEAR_SOLVER` | `Pardiso` | (same as IAF) |
| `LUMP_AND_SPRAY` | `yes` | (same as IAF) |
| `MATRIX_PROCESSING` | `symdrop` | (same as IAF) |
| `INITIAL_AGE` | `TMage` | Warm-start NK from the TRAF TMsolve output (auto-resolved to `…_traf/` dir) |
| `GM_REDI` / `TBLOCKING` / `LOAD_BALANCE` / `ACTIVE_CELLS_MAP` / `TRACE_SOLVER_HISTORY` / `CHECK_BOUNDS` / `PLOT_TS` | (same as IAF) | |
| `GRID_HX/HY/HZ` | 7/7/2 | (same as IAF — grid + FTS already on disk from IAF preprocessing) |
| `MLD_TIME_WINDOW` | (unset) | (same as IAF) |

## Pipeline (JOB_CHAIN per driver invocation)

Each (PM, TW) needs: `TMbuild → TMsolve(const) → NK(const) → run1yrNK(const) → plotNK`. **No preprocessing, no `grid`/`vel`/`clo`/`diagnose_w` steps** — those are TRAF-agnostic and already on disk from the IAF runs. Encoded as:

```
JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK-plotNK
```

Under `TRAF=yes & TRAF_TM_SOURCE=invVMtV`, the `TMbuild` step short-circuits in [src/create_matrix.jl](../src/create_matrix.jl) — it loads the **forward** `M.jld2` from the non-`_traf` dir, computes `invVMtV = sparse(Diagonal(v.^-1)) * M' * sparse(Diagonal(v))`, and writes `invVMtV.jld2` into the `_traf` dir. No autodiff, no Oceananigans heavy setup. Downstream `TMsolve` and `NK` then read `invVMtV.jld2` from the `_traf` dir via the `M_basename` selector in [src/solve_matrix_age.jl](../src/solve_matrix_age.jl) and [src/solve_periodic_NK.jl](../src/solve_periodic_NK.jl).

If `invVMtV` ever fails to converge for NK, fall back to Option A by setting `TRAF_TM_SOURCE=M_traf` for that run — `TMbuild` then runs the full autodiff path with sign-flipped yearly velocities ([src/matrix_setup.jl](../src/matrix_setup.jl)) and writes `M_traf.jld2`. All downstream steps pick it up automatically.

## Submission strategy

The TRAF jobs depend on the corresponding IAF `TMbuild` jobs (forward `M.jld2`). Two options:

**Option 1 — chain via PBS deps** (preferred when launching from the same shell that submitted IAF):
Capture the IAF `TMbuild` job IDs (e.g. via `qstat -u $USER` or the manifest TOML) and pass them as the `TMBUILD_JOB` dependency env var when invoking the TRAF driver, so each TRAF `TMbuild` waits for the forward `TMbuild` to finish before short-circuiting.

**Option 2 — submit after IAF completes** (simpler):
Wait until all four IAF runs from [IAF_simulations.md](IAF_simulations.md) have produced their `outputs/{PM}/{EXP}/{TW}/TM/{MC}/const/M.jld2` (where `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`, **no** `_traf` suffix), then submit the four TRAF invocations independently.

**Per (PM, TW)** — one driver invocation each:
```bash
PARENT_MODEL=ACCESS-OM2-1 \
TIME_WINDOW=1968-1977 \
TRAF=yes \
VELOCITY_SOURCE=totaltransport W_FORMULATION=wdiagnosed \
ADVECTION_SCHEME=centered2 TIMESTEPPER=SRK3 TIMESTEP_MULT=12 \
MONTHLY_KAPPAV=yes TM_SOURCE=const INITIAL_AGE=TMage \
LUMP_AND_SPRAY=yes LINEAR_SOLVER=Pardiso MATRIX_PROCESSING=symdrop \
PARTITION=1x1 \
JOB_CHAIN=TMbuild-TMsolve-NK-run1yrNK-plotNK \
bash scripts/driver.sh
```

Repeat for the other three (PM, TW) combinations. Total: **4 driver invocations, 4 NK adjoint runs**.

## Critical files / paths

- Driver and ENV defaults
  - [scripts/driver.sh](../scripts/driver.sh) — JOB_CHAIN dispatcher; now exports `TRAF`, `TRAF_TM_SOURCE` in `COMMON_VARS`
  - [scripts/env_defaults.sh](../scripts/env_defaults.sh) — appends `_traf` to MODEL_CONFIG when `TRAF=yes`
  - [model_configs/ACCESS-OM2-1.sh](../model_configs/ACCESS-OM2-1.sh) — gpuvolta, WALLTIME_NK=03:00:00
  - [model_configs/ACCESS-OM2-025.sh](../model_configs/ACCESS-OM2-025.sh) — gpuhopper, WALLTIME_NK=48:00:00

- Source (TRAF-aware — see "Implementation" section below)
  - [src/shared_utils/config.jl](../src/shared_utils/config.jl) — `build_model_config` adds `_traf` suffix
  - [src/shared_utils/data_loading.jl](../src/shared_utils/data_loading.jl) — `reverse_fts_time!` helper (in-place, zero-alloc)
  - [src/setup_model.jl](../src/setup_model.jl) — applies `reverse_fts_time!` to every monthly FTS under TRAF
  - [src/create_matrix.jl](../src/create_matrix.jl) — top-level short-circuit for `invVMtV` synthesis
  - [src/matrix_setup.jl](../src/matrix_setup.jl) — sign-flips yearly u/v/w for the `M_traf` autodiff fallback
  - [src/solve_matrix_age.jl](../src/solve_matrix_age.jl), [src/solve_periodic_NK.jl](../src/solve_periodic_NK.jl) — pick `invVMtV.jld2` / `M_traf.jld2` via `M_basename`
  - [src/plot_standardrun_age.jl](../src/plot_standardrun_age.jl), [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) — `age_traf_…` filename infix + "TRAF age (time to re-emergence)" plot title

- Outputs land under (per run; `MC_TRAF = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12_traf`)
  - Adjoint matrix (built by TRAF TMbuild): `outputs/{PM}/{EXP}/{TW}/TM/{MC_TRAF}/const/invVMtV.jld2`
  - TRAF TM age (warm-start source for NK): `outputs/{PM}/{EXP}/{TW}/TM/{MC_TRAF}/const/steady_age_LSprec_Pardiso_symdrop.jld2`
  - TRAF NK age (target): `outputs/{PM}/{EXP}/{TW}/periodic/{MC_TRAF}/NK/age_Pardiso_LSprec.jld2`
  - 1-year-from-periodic-sol (TRAF): `outputs/{PM}/{EXP}/{TW}/standardrun/{MC_TRAF}/age_from_periodic_sol_Pardiso_LSprec.jld2`
  - PNGs / MP4s carry the `age_traf_…` filename infix and "TRAF age (time to re-emergence)" titles, all under the `…_traf/` dirs
  - PBS logs: `logs/PBS/` and Julia logs under `logs/julia/{PM}/{EXP}/{TW}/...` with `${MODEL_CONFIG}` (which now contains `_traf`) embedded in filenames

## "Clean rebuild" semantics

Same as IAF: don't delete anything; each step rewrites at its fixed `_traf`-tagged path. Re-running the chain end-to-end refreshes every TRAF artifact for this configuration. The forward (`MC`, no `_traf`) tree is untouched by TRAF jobs — the read of forward `M.jld2` is read-only.

If a previous TRAF run wrote at the same `MC_TRAF` path but with a different code version, it will be overwritten — confirm `GIT_COMMIT` in the manifest TOML matches expectations.

## Verification

After all 4 invocations have completed:

1. **`invVMtV.jld2` exists for all 4 runs**:
   ```bash
   MC_TRAF=totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12_traf
   for PM in ACCESS-OM2-1 ACCESS-OM2-025; do
     for TW in 1968-1977 1999-2008; do
       ls outputs/$PM/*/$TW/TM/$MC_TRAF/const/invVMtV.jld2
     done
   done
   ```

2. **NK adjoint age output exists for all 4 runs**:
   ```bash
   for PM in ACCESS-OM2-1 ACCESS-OM2-025; do
     for TW in 1968-1977 1999-2008; do
       ls outputs/$PM/*/$TW/periodic/$MC_TRAF/NK/age_Pardiso_LSprec.jld2
     done
   done
   ```

3. **NK convergence** — tail each TRAF NK log for "Newton converged" and final residual:
   ```bash
   ls logs/julia/ACCESS-OM2-1/*/1968-1977/periodic/NK/*traf*.out | xargs -I{} tail -20 {}
   # (repeat for the other 3 combos)
   ```

4. **plotNK output** — confirm `age_traf_*` PNGs/MP4s were produced under each `outputs/.../periodic/$MC_TRAF/NK/` with "TRAF age (time to re-emergence)" in the titles.

5. **FTS reversal smoke test** (one-off, optional but recommended for first submission):
   In a REPL after `setup_model.jl` runs under `TRAF=yes`, load the corresponding non-TRAF FTS in a sibling session and check `parent(u_ts_traf[m]) ≈ -parent(u_ts_fwd[N+1-m])` (and the analogous identities for v, η, T, S, κV) for a few snapshot indices `m`.

6. **Cross-run deliverable** — once all four `age_Pardiso_LSprec.jld2` files exist under `…_traf/NK/`, plus the forward equivalents from IAF, derive the surface ventilation fraction
   ```
   V↓ = A⁻¹ V (age_traf / sink_timescale)
   ```
   (sink_timescale = 3Δt from `age_parameters` in [setup_model.jl](../src/setup_model.jl)) in a notebook / small comparison script. **Out of scope for this plan**; this is the downstream deliverable the runs feed.

## Implementation done before submission

Unlike the IAF plan (which only needed small driver fixes), TRAF required a feature-level pipeline change. Summary of edits landed in this session (see plan file `/home/561/bp3051/.claude/plans/i-want-to-set-breezy-wombat.md` for the full design rationale and backward-compat audit):

| File | Change |
|---|---|
| `scripts/env_defaults.sh` | New `TRAF`, `TRAF_TM_SOURCE` defaults + validation + export; `_traf` appended to `MODEL_CONFIG` when `TRAF=yes` |
| `scripts/driver.sh` | New `TRAF`, `TRAF_TM_SOURCE` defaults; both added to `COMMON_VARS` (passed via qsub `-v`) |
| `src/shared_utils/config.jl` | `build_model_config` mirrors the shell-side `_traf` suffix rule |
| `src/shared_utils/data_loading.jl` | New `reverse_fts_time!(fts; flip_sign=false)` — in-place, zero-alloc, pairwise swap |
| `src/setup_model.jl` | Parses `TRAF`; hard-errors on `wprescribed + TRAF=yes`; calls `reverse_fts_time!` after every monthly FTS load (u, v: `flip_sign=true`; η, T, S, κV: `flip_sign=false`) |
| `src/matrix_setup.jl` | Sign-flips yearly u/v (and w when wprescribed) under `TRAF=yes & TRAF_TM_SOURCE=M_traf` (Option A only) |
| `src/create_matrix.jl` | Top-level short-circuit for Option B (load forward M, synthesize `invVMtV = D(v⁻¹) Mᵀ D(v)`, save in `_traf` dir, `exit()`); save filename swap for Option A (`M_traf.jld2` vs `M.jld2`) |
| `src/solve_matrix_age.jl`, `src/solve_periodic_NK.jl` | Validate `TM_SOURCE=const` under TRAF; pick `M.jld2` / `invVMtV.jld2` / `M_traf.jld2` via `M_basename` |
| `src/shared_utils/analysis_and_plotting.jl` | `animate_zonal_averages` and `animate_depth_slices` gain `tracer_title::AbstractString = "age"` kwarg (default preserves current titles bit-identically) |
| `src/plot_standardrun_age.jl`, `src/plot_periodic_1year_age.jl` | Under TRAF, label gets `age_traf_…` infix and `tracer_title = "TRAF age (time to re-emergence)"` passed to animation helpers |

**Backward-compat invariant**: every change reduces to a no-op when `TRAF=no` (the default). Audited per-edit in the plan file; sanity-tested by `bash -n` on the shell scripts and `Meta.parseall` on every edited Julia file.

## Submitted runs

*To be filled in after submission (in a separate session).* Mirror the IAF table format once jobs are queued:

| (PM, TW) | Driver invocation | Jobs | TMbuild job | NK_c job | Plot job |
|---|---|---|---|---|---|
| OM2-1, 1968-1977 | TBD | 5 | TBD | TBD | TBD |
| OM2-1, 1999-2008 | TBD | 5 | TBD | TBD | TBD |
| OM2-025, 1968-1977 | TBD | 5 | TBD | TBD | TBD |
| OM2-025, 1999-2008 | TBD | 5 | TBD | TBD | TBD |
