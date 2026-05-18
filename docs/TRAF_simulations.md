# TRAF_simulations.md — periodic NK adjoint age across TW × resolution

## Context

The companion to [IAF_simulations.md](IAF_simulations.md). Same 2×2 (PM × TW) grid, same model physics, **but with the Time-Reversed Adjoint Flow flag `TRAF=yes`** so we compute the adjoint age (a.k.a. "time to re-emergence", Holzer & Hall 2000 / Khatiwala 2003) under each circulation. Together with the forward NK ages from the IAF runs, this gives us the surface ventilation volume per unit area `𝒱ꜜ = A⁻¹ V (age_traf / sink_timescale)` (Pasquier *et al.* 2024, *JGR-Oceans*, doi:10.1029/2024JC021043). Units: m³/m² = m — without the `A⁻¹` factor, `V (age_traf / sink_timescale)` is the ocean volume ventilated per surface grid cell; `A⁻¹` converts that to a per-unit-area quantity. (File/variable name for downstream scripts: `calVdown`.)

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

## Dependencies on IAF runs

TRAF jobs read but do not (re)generate anything in the `preprocessed_inputs/` tree. Specifically, each TRAF (PM, TW) run reads:

- **Forward `M.jld2`**: `outputs/{PM}/{EXP}/{TW}/TM/{MC}/const/M.jld2`, produced by IAF `TMbuild` (with `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`, **no** `_traf` suffix). Required for `invVMtV` synthesis.
- **Six monthly FTS files**, all under `preprocessed_inputs/{PM}/{EXP}/{TW}/monthly/`:
  - `u_from_total_transport_monthly.jld2` (reversed and sign-flipped in memory)
  - `v_from_total_transport_monthly.jld2` (reversed and sign-flipped in memory)
  - `eta_monthly.jld2` (reversed only)
  - `temp_monthly.jld2` (reversed only)
  - `salt_monthly.jld2` (reversed only)
  - `kappa_v_monthly.jld2` (reversed only)

Within an IAF driver invocation the DAG `prep → grid → vel → clo → diagnose_w → TMbuild` makes all of these implicit `afterok` dependencies. But TRAF is a **separate driver invocation** that does not auto-inherit those PBS deps. Two coordination strategies (expanded under "Submission strategy" below):

- **Option 1 — PBS deps**: pass `TMBUILD_JOB=<iaf-tmbuild-id>` to the TRAF driver invocation. This is sufficient: IAF's `TMbuild` has its own `afterok` dependency on `clo`+`diagw` (and transitively on `prep`/`grid`/`vel`), so all six monthly FTS files are guaranteed present whenever the TRAF `TMbuild` is allowed to start.
- **Option 2 — wait**: confirm all four IAF `TMbuild` jobs have produced their forward `M.jld2`, then verify the six monthly FTS files exist on disk (one-liner below) before submitting TRAF.

Pre-submission existence check (Option 2):
```bash
MC=totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12
for PM in ACCESS-OM2-1 ACCESS-OM2-025; do
  for TW in 1968-1977 1999-2008; do
    for f in u_from_total_transport v_from_total_transport eta temp salt kappa_v; do
      ls preprocessed_inputs/$PM/*/$TW/monthly/${f}_monthly.jld2
    done
    ls outputs/$PM/*/$TW/TM/$MC/const/M.jld2
  done
done
```

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

## Pre-submission validation — FTS reversal smoke test

Before submitting any TRAF NK run, submit and pass the `trafftsrev` test driver step. This loads each of the six monthly FTS files that TRAF reads (u, v, η, T, S, κV) one by one, also loads the corresponding non-reversed FTS, applies `reverse_fts_time!` with the matching `flip_sign` choice, and verifies (a) bit-exact `parent`-array equality at every snapshot index — `parent(rev[i]) == sign · parent(fwd[N+1-i])` — and (b) interpolated agreement `interior(rev[Time(t)]) ≈ sign · interior(fwd[Time(T - t)])` at 22 mid-snapshot clock times (avoiding exact snapshot times to sidestep an asymmetric Oceananigans code path in `fts[Time(t)]` between the n₁ == n₂ and n₁ ≠ n₂ cases — TRAF simulations always go through the n₁ ≠ n₂ linear-blend path, so mid-snapshot times are the relevant check).

The test is OM2-1-only — `reverse_fts_time!` is grid-agnostic, so the OM2-1 run is sufficient as a smoke test for OM2-025 as well. Run once per TIME_WINDOW:

```bash
PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1968-1977 JOB_CHAIN=trafftsrev bash scripts/test_driver.sh
PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1999-2008 JOB_CHAIN=trafftsrev bash scripts/test_driver.sh
```

Pass criterion: log under `logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/{TW}/test/traf_fts_reversal_*.log` shows all `@test` passing (no `Test Failed`), and ends with `Done running test/test_traf_fts_reversal.jl`.

If a test fails, do **not** submit any TRAF NK run — investigate the reversal helper or the FTS load path first.

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

5. **Cross-run deliverable** — once all four `age_Pardiso_LSprec.jld2` files exist under `…_traf/NK/`, plus the forward equivalents from IAF, derive the surface ventilation volume per unit area
   ```
   𝒱ꜜ = A⁻¹ V (age_traf / sink_timescale)
   ```
   (units m³/m² = m; `sink_timescale = 3Δt` from `age_parameters` in [setup_model.jl](../src/setup_model.jl)) in a notebook / small comparison script (suggested variable name `calVdown`). **Out of scope for this plan**; this is the downstream deliverable the runs feed.

(The previous bullet 5 — a REPL-style one-off FTS reversal smoke test — has been superseded by the automated `trafftsrev` test driver step under "Pre-submission validation" above.)

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

### Attempt 1 — `GIT_COMMIT=cd80157` (2026-05-16, all 4 NK_c failed Exit 1)

All four NK_c jobs hit `Scalar indexing is disallowed` at setup. Root cause: `reverse_fts_time!` used a `@simd` scalar-indexed pairwise-swap loop, which works on CPU (the `trafftsrev` test architecture) but errors on GPU (the production architecture). OM2-025 NK_c (still queued) and the downstream run1yrNK_c/plotNK (held) were cancelled to save H200 SU. Fix landed in commit f26fadc (broadcast `.=` swap with a `similar(parent(fts[1]))` temp buffer).

| (PM, TW) | TMbuild | NK_c | Outcome |
|---|---|---|---|
| OM2-1, 1968-1977 | 168464264 | 168464267 | NK_c Exit 1 (scalar indexing) |
| OM2-1, 1999-2008 | 168464274 | 168464277 | NK_c Exit 1 (scalar indexing) |
| OM2-025, 1968-1977 | 168464308 | 168464311 | NK_c cancelled while queued |
| OM2-025, 1999-2008 | 168464316 | 168464319 | NK_c cancelled while queued |

### Attempt 2 — `GIT_COMMIT=f26fadc` (2026-05-16)

Resubmitted after the GPU-safe `reverse_fts_time!` fix. Same 6-job chain per (PM, TW); 24 PBS jobs total.

| (PM, TW) | GPU queue | TMbuild | TMslv_c | TMslv_cG | NK_c | run1yrNK_c | plotNK |
|---|---|---|---|---|---|---|---|
| OM2-1, 1968-1977 | gpuvolta | 168481391 ✓ | 168481392 ✓ | 168481393 ✗ | 168481394 ✓ | 168481395 ✓ | 168481396 ✓ |
| OM2-1, 1999-2008 | gpuvolta | 168481406 ✓ | 168481407 ✓ | 168481408 ✗ | 168481409 ✓ | 168481411 ✓ | 168481412 ✓ |
| OM2-025, 1968-1977 | gpuhopper | 168481413 ✓ | 168481414 ✓ | 168481415 ✗ | 168481416 ✓ | 168481417 ✓ | 168481418 ⚠ |
| OM2-025, 1999-2008 | gpuhopper | 168482434 ✓ | 168482435 ✓ | 168482436 ✗ | 168482437 ⚠ | 168482438 ✓ | 168482439 ⚠ |

**OM2-1 — converged.**

| (PM, TW) | Φ! calls | NK_c walltime | Volume-weighted mean TRAF age (yr) | `age_Pardiso_LSprec.jld2` |
|---|---:|---:|---:|---|
| OM2-1, 1968-1977 | 67 | 25:29 | **839.12** | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12_traf/NK/age_Pardiso_LSprec.jld2` |
| OM2-1, 1999-2008 | 70 | 25:53 | **886.39** | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12_traf/NK/age_Pardiso_LSprec.jld2` |

**OM2-025 — 3/4 deliverables in place; 1 NK stalled.**

| (PM, TW) | NK_c | Φ! calls | Wall | retcode | Volume-weighted mean TRAF age (yr) |
|---|---|---:|---:|---|---:|
| OM2-025, 1968-1977 | 168481416 | 120 | 05:05:48 | Success | **892.89** |
| OM2-025, 1999-2008 | 168482437 | 102 | 04:19:10 | **Stalled** | 2.4e-18 (garbage) |

### Issues to address before declaring the campaign complete

1. **OM2-025 / 1999-2008 NK stalled — forward simulation blew up in Φ! call #1**: starting from `age = 0` (because the warm-start file was missing — see issue 3 below), Φ! call #1 went numerically unstable at cell `(i=1288, j=1047, k=36)` (high latitude, near but not at the j=Ny=1080 tripolar fold). `max(age)` exploded to `6.8e+25` yr by sim iter 122 (0.083 yr in), then bounced through `1e+44`, `1e+45`, `1e+68` across the rest of the year. NK never recovered (initial residual `1.0e+76`, final `2.4e+70` after 102 outer iterations, retcode `Stalled`). The corresponding IAF run at the same (PM, TW) converged (see [IAF_simulations.md](IAF_simulations.md)), so the velocities are sound — the issue is TRAF-specific. Plausible root causes to investigate: a single bad cell in the time-reversed `(u, v, η)` that breaks the diagnosed-`w` continuity at that location; or a stability margin that's tight enough for `1999-2008` velocities under TRAF at OM2-025 resolution that the AB2/SRK3 step needs a smaller `TIMESTEP_MULT`. **Likely next step**: probe `(u, v, η)` at `(1288, 1047, 36)` and neighbours for the reversed FTS to look for NaNs / huge values, and/or rerun with `TIMESTEP_MULT=6` for this one (PM, TW).

2. **`plotNK` killed at walltime (-29) for both OM2-025 chains** (168481418, 168482439): wall request was `01:00:00` but each actually needed 01:00:15 / 01:00:51. The 1968-1977 chain's `age_Pardiso_LSprec.jld2` exists and is plottable; just bump `WALLTIME_PLOTNK` in [model_configs/ACCESS-OM2-025.sh](../model_configs/ACCESS-OM2-025.sh) (e.g. to `02:00:00`) and rerun `JOB_CHAIN=plotNK` for the converged chain.

3. **`TMage` warm-start was not used in ANY of the 4 chains**: each NK log shows the warning `INITIAL_AGE=TMage but no matrix age file found in …/TM/…_traf/const — starting from zeros`. Root cause: NK_c only `afterok`s TMbuild, not TMslv_c, so it can (and did) start before TMslv_c finished writing the warm-start file. For OM2-1 (TMslv_c ~3-6 min vs NK_c ~25 min) this happened to work out fine — NK converged from zeros. For OM2-025/1999-2008 it may have been the difference between convergence and divergence (a sensible warm-start would have kept the iterate in a stable region). Fix: add `TMslv_c` (or `TMslv_c + TMslv_cG` once 5ad89c3 lands) to NK_c's `afterok` dependency list in [scripts/driver.sh](../scripts/driver.sh).

**Known orthogonal failure — `TMslv_cG` (GPU TMsolve comparison) — FIXED in commit 5ad89c3 for future runs:** all four `TMslv_cG` jobs failed Exit 1 with `ArgumentError: No file exists at given path: …/TM/…_traf/const/M.jld2`. Root cause: [src/solve_matrix_age_gpu.jl](../src/solve_matrix_age_gpu.jl) hardcoded `M.jld2` rather than dispatching via `M_basename` like its CPU sibling [src/solve_matrix_age.jl](../src/solve_matrix_age.jl) — under `TRAF=yes & TRAF_TM_SOURCE=invVMtV` the only file in the `_traf/const/` dir is `invVMtV.jld2`. Did **not** block the NK chain (NK_c only `afterok`s TMbuild, and reads `invVMtV.jld2` via its own `M_basename` dispatch). Commit 5ad89c3 mirrors the `TRAF`/`TRAF_TM_SOURCE`/`M_basename` block into `solve_matrix_age_gpu.jl`; will take effect on the next TRAF rerun.

Manifests:
- OM2-1 / 1968-1977 — `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/manifests/20260516T220607_1879521.toml`
- OM2-1 / 1999-2008 — `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/manifests/20260516T220758_1883576.toml`
- OM2-025 / 1968-1977 — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/manifests/20260516T220819_1884908.toml`
- OM2-025 / 1999-2008 — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/manifests/20260516T221502_1904462.toml`

### Attempt 3 — `GIT_COMMIT=d40e13a` + follow-ups (2026-05-18)

After diagnosing Attempt 2, we made three fixes:
- f26fadc — `reverse_fts_time!` GPU-safe (broadcast `.=` swap; landed before Attempt 2).
- 5ad89c3 — `solve_matrix_age_gpu.jl` now dispatches `M_basename` like its CPU sibling so `TMslv_cG` no longer dies on `invVMtV.jld2`.
- d40e13a — `NK_c` now `afterok`-depends on `TMslv_c` (in addition to `TMbuild`) so the `INITIAL_AGE=TMage` warm-start is actually on disk by the time NK opens it; OM2-025 `plotNK` resources bumped to `02:00:00 / 24 CPU / 96 GB`.

We also confirmed via `test/probe_traf_bad_cell.jl` (job 168615803) that the input data at the 1999-2008 blow-up cell `(1288, 1047, 36)` and its face neighbours is clean: zero NaN/Inf/large values across the whole parent (incl. halos) for u/v/η/T/S/κV, and `reverse_fts_time!` produces bit-exact mirror values at every probed cell. So the blow-up was not due to a sign bug, NaN, or a corrupt input.

**6 driver invocations.** Four with `TIMESTEPPER=AB2 TIMESTEP_MULT=4` (full chain; new `MC_TRAF = totaltransport_wdiagnosed_centered2_AB2_mkappaV_DTx4_traf` — same per-year stage count as SRK3 + M=12, more conservative effective Δt). One SRK3 + M=12 NK rerun for OM2-025/1999-2008 (just `NK-run1yrNK-plotNK`, using the existing TMbuild artefacts + the disk-resident warm-start). One SRK3 + M=12 `plotNK`-only rerun for OM2-025/1968-1977 to complete the 5 missing MP4s.

**Attempt 3a — AB2 M=4 full chain (4 invocations, 24 PBS jobs):**

| (PM, TW) | TMbuild | TMslv_c | TMslv_cG | NK_c | run1yrNK_c | plotNK |
|---|---|---|---|---|---|---|
| OM2-1, 1968-1977 | 168619937 | 168619939 | 168619940 | 168619941 | 168619942 | 168619944 |
| OM2-1, 1999-2008 | 168619962 | 168619963 | 168619964 | 168619965 | 168619966 | 168619967 |
| OM2-025, 1968-1977 | 168620006 | 168620008 | 168620009 | 168620010 | 168620012 | 168620014 |
| OM2-025, 1999-2008 | 168620112 | 168620113 | 168620114 | 168620116 | 168620117 | 168620118 |

**Attempt 3b — SRK3 M=12, NK rerun for OM2-025/1999-2008 (3 PBS jobs):**

| step | job | uses |
|---|---|---|
| NK_c | 168620173 | existing `invVMtV.jld2` + disk-resident `steady_age_*.jld2` warm-start |
| run1yrNK_c | 168620174 | afterok NK_c |
| plotNK | 168620175 | afterok run1yrNK_c |

**Attempt 3c — SRK3 M=12, plotNK only for OM2-025/1968-1977 (1 PBS job):**

| step | job | uses |
|---|---|---|
| plotNK | 168620191 | existing converged `age_Pardiso_LSprec.jld2`; finishes the 5 MP4s that the original 01:00:00-walltime job (168481418) didn't complete |

Manifests:
- OM2-1 / 1968-1977 (3a) — `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/manifests/20260518T110439_2335292.toml`
- OM2-1 / 1999-2008 (3a) — `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/manifests/20260518T110510_2339993.toml`
- OM2-025 / 1968-1977 (3a) — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/manifests/20260518T110529_2344043.toml`
- OM2-025 / 1999-2008 (3a) — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/manifests/20260518T110549_2346922.toml`
- OM2-025 / 1999-2008 (3b) — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/manifests/20260518T110628_2354101.toml`
- OM2-025 / 1968-1977 (3c) — `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/manifests/20260518T110657_2358935.toml`
