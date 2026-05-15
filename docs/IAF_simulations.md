# IAF_simulations.md — periodic NK age across TW × resolution

## Context

We want to compare periodic steady-state age as a function of (a) forcing decade and (b) parent-model resolution, using the Newton-Krylov solver. The four NK age solves below give us a 2×2 comparison:

|              | TW=1968-1977 | TW=1999-2008 |
|--------------|--------------|--------------|
| OM2-1        | run 1        | run 2        |
| OM2-025      | run 3        | run 4        |

Resolution effect: row-by-row (run 1 vs 3, run 2 vs 4). Time-window effect: column-by-column (run 1 vs 2, run 3 vs 4). The whole pipeline is re-run from scratch (overwriting any pre-existing files) so no stale inputs/outputs leak into the comparison.

All four runs use the same model physics so the only differences are PARENT_MODEL, TIME_WINDOW, and the GPU queue (gpuvolta for OM2-1, gpuhopper for OM2-025). Each run is serial (1 GPU, PARTITION=1x1).

## Agreed configuration (per-run ENV)

The unified MODEL_CONFIG tag is `totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`.

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
| `VELOCITY_SOURCE` | `totaltransport` | Total mass transports (uses `total_transport_uv` MOM files) |
| `W_FORMULATION` | `wdiagnosed` | Diagnose w via Oceananigans continuity (default) |
| `ADVECTION_SCHEME` | `centered2` | Centered 2nd-order — cheapest, lowest dissipation |
| `TIMESTEPPER` | `SRK3` | SplitRungeKutta-3; works with M=12 Δt multiplier |
| `TIMESTEP_MULT` | `12` | Per docs/timestep_multiplier.md — optimal for OM2-1 and OM2-025 |
| `MONTHLY_KAPPAV` | `yes` | Time-varying monthly κV from MLD FTS |
| `PARTITION` | `1x1` | Serial (1 GPU) |
| `TM_SOURCE` | `const` | NK driven by single Jacobian from yearly-averaged fields |
| `LINEAR_SOLVER` | `Pardiso` | Default direct sparse solver (preconditioner factorization) |
| `LUMP_AND_SPRAY` | `yes` | Driver default; uses Bardin-2014 coarsening preconditioner |
| `MATRIX_PROCESSING` | `symdrop` | Keep (i,j) only if (j,i) also exists, then drop zeros — enforces structural symmetry for Pardiso |
| `INITIAL_AGE` | `TMage` | Warm-start NK from TM-derived age (requires TMsolve) |
| `GM_REDI` | `no` | Default — no isopycnal diffusion |
| `TBLOCKING` | `no` | Serial — no temporal blocking |
| `GRID_HX/HY/HZ` | 7/7/2 | New defaults after lowering `GRID_HZ` 7→2 (commit 551a023) — Hz=2 matches what all existing 3D FTS already carry |
| `LOAD_BALANCE` | `no` | Only meaningful for multi-rank |
| `ACTIVE_CELLS_MAP` | `yes` | Default |
| `TRACE_SOLVER_HISTORY` | `no` | Skip per-iteration trace files / plotNKtrace |
| `CHECK_BOUNDS` | `no` | Default |
| `PLOT_TS` | `no` | Default — no T/S animations |
| `MLD_TIME_WINDOW` | (unset) | Same as TIME_WINDOW; outputs land in production tree, not test/ |

## Pipeline (JOB_CHAIN per driver invocation)

Each (PM, TW) needs: `prep → grid → vel + clo → diagnose_w → TMbuild → TMsolve(const) → NK(const) → run1yrNK(const) → plotNK`. Encoded as:

```
JOB_CHAIN=preprocessing-TMbuild-TMsolve-NK-run1yrNK-plotNK
```

(`preprocessing` expands to `prep-grid-vel-clo-diagnose_w-partition`; with PARTITION=1x1 the partition step is a no-op.)

## Submission strategy (grid sharing across TWs)

Grid is per-experiment (PM × EXP) and shared across TIME_WINDOWs. To avoid two concurrent grid builds writing to the same `preprocessed_inputs/{PM}/{EXP}/grid.jld2`, chain the second TW onto the first invocation's grid job via `GRID_JOB=<jobid>`:

**Per PARENT_MODEL** (`ACCESS-OM2-1` and `ACCESS-OM2-025`):
1. **First TW (1968-1977)** — full pipeline incl. grid:
   ```bash
   PARENT_MODEL=ACCESS-OM2-1 \
   TIME_WINDOW=1968-1977 \
   VELOCITY_SOURCE=totaltransport W_FORMULATION=wdiagnosed \
   ADVECTION_SCHEME=centered2 TIMESTEPPER=SRK3 TIMESTEP_MULT=12 \
   MONTHLY_KAPPAV=yes TM_SOURCE=const INITIAL_AGE=TMage \
   LUMP_AND_SPRAY=yes LINEAR_SOLVER=Pardiso MATRIX_PROCESSING=symdrop \
   PARTITION=1x1 \
   JOB_CHAIN=preprocessing-TMbuild-TMsolve-NK-run1yrNK-plotNK \
   bash scripts/driver.sh
   ```
   Capture the printed `grid` job ID → `$GRID_JOB_OM2_1`.

2. **Second TW (1999-2008)** — reuse grid via env var:
   ```bash
   GRID_JOB=$GRID_JOB_OM2_1 \
   PARENT_MODEL=ACCESS-OM2-1 \
   TIME_WINDOW=1999-2008 \
   <same flags as above> \
   JOB_CHAIN=prep-vel-clo-diagnose_w-partition-TMbuild-TMsolve-NK-run1yrNK-plotNK \
   bash scripts/driver.sh
   ```
   (Note: `grid` omitted from JOB_CHAIN; vel/clo deps include `$GRID_JOB`.)

Repeat both invocations with `PARENT_MODEL=ACCESS-OM2-025` (gpuhopper picked up automatically from `model_configs/ACCESS-OM2-025.sh`).

Total: 4 driver invocations (2 per PM), 4 NK runs.

## Critical files / paths

- Driver and ENV defaults
  - [scripts/driver.sh](../scripts/driver.sh) — JOB_CHAIN dispatcher
  - [scripts/env_defaults.sh](../scripts/env_defaults.sh) — flag defaults and MODEL_CONFIG tag construction
  - [model_configs/ACCESS-OM2-1.sh](../model_configs/ACCESS-OM2-1.sh) — gpuvolta, WALLTIME_NK=03:00:00
  - [model_configs/ACCESS-OM2-025.sh](../model_configs/ACCESS-OM2-025.sh) — gpuhopper, WALLTIME_NK=48:00:00

- Source (no edits — read for reference)
  - [src/setup_model.jl](../src/setup_model.jl) — model construction (uses all the flags above)
  - [src/solve_periodic_NK.jl](../src/solve_periodic_NK.jl) — NK driver
  - [src/solve_matrix_age.jl](../src/solve_matrix_age.jl) — TM age solve
  - [src/create_matrix.jl](../src/create_matrix.jl) — Jacobian build (TMbuild)

- Outputs land under (per run; `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`)
  - TM age (warm-start source): `outputs/{PM}/{EXP}/{TW}/TM/{MC}/const/steady_age_LSprec_Pardiso_symdrop.jld2` (filename = `steady_age_$(coarse_tag)_$(LINEAR_SOLVER)_$(MATRIX_PROCESSING).jld2`, per [src/solve_matrix_age.jl:88](../src/solve_matrix_age.jl#L88))
  - NK age (target): `outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/age_Pardiso_LSprec.jld2` (filename = `age_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2`, per [src/solve_periodic_NK.jl:258](../src/solve_periodic_NK.jl#L258) — does NOT include MATRIX_PROCESSING)
  - 1-year-from-periodic-sol: `outputs/{PM}/{EXP}/{TW}/standardrun/{MC}/age_from_periodic_sol_Pardiso_LSprec.jld2`
  - PBS logs: `logs/PBS/` and Julia logs under `logs/julia/{PM}/{EXP}/{TW}/...`

## "Clean rebuild" semantics

Per the user: **don't delete anything; just overwrite.** Each step in the pipeline rewrites its output files at the fixed paths above (keyed by MODEL_CONFIG), so re-running the chain end-to-end refreshes every artifact for this configuration. Stale files at other MODEL_CONFIG tags are not affected and not relevant for this comparison.

Caveat: if a previous run wrote an output at exactly the same MODEL_CONFIG path but with a different code version, it will be overwritten — confirm `GIT_COMMIT` in the manifest TOML matches what you expect.

## Verification

After all 4 invocations have completed:

1. **Manifest sanity check** — every (PM, TW) should have a fresh manifest:
   ```bash
   for PM in ACCESS-OM2-1 ACCESS-OM2-025; do
     for TW in 1968-1977 1999-2008; do
       ls -lt outputs/$PM/*/$TW/manifests/ | head -3
     done
   done
   ```
   Each should show a recent TOML with `JOB_CHAIN`, `VELOCITY_SOURCE=totaltransport`, `TIMESTEPPER=SRK3`, `TIMESTEP_MULT=12`, `MONTHLY_KAPPAV=yes`, `TM_SOURCE=const`.

2. **NK output exists for all 4 runs**:
   ```bash
   MC=totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12
   for PM in ACCESS-OM2-1 ACCESS-OM2-025; do
     for TW in 1968-1977 1999-2008; do
       ls outputs/$PM/*/$TW/periodic/$MC/NK/age_Pardiso_LSprec.jld2
     done
   done
   ```

3. **NK convergence** — tail each NK log for "Newton converged" and final residual:
   ```bash
   ls logs/julia/ACCESS-OM2-1/*/1968-1977/periodic/NK/*.out | xargs -I{} tail -20 {}
   # (repeat for the other 3 combos)
   ```

4. **plotNK output** — confirm a `plotNK` PDF/PNG was produced under each `outputs/.../periodic/$MC/NK/` (the plot job's stdout will say where it wrote).

5. **Cross-run comparison** — Once all four `age_Pardiso_LSprec.jld2` files exist, load them in a notebook / a small `compare_ages.jl` script and produce TW-difference + resolution-difference age maps. (Out of scope for this plan; this is the deliverable the runs feed.)

## Pre-flight fixes applied before submission

Three small repo edits were needed to align defaults with the existing inputs and to make the driver forward our chosen flags:

| Commit | File | Change |
|---|---|---|
| 551a023 | `scripts/env_defaults.sh` | Lower `GRID_HZ` default 7 → 2 (matches saved FTS Hz=2 for 3D fields; grid Hz=7 was overkill) |
| b61eacb | `scripts/driver.sh` | Add `MATRIX_PROCESSING=${...:-raw}` and append to `COMMON_VARS` so the env var actually reaches PBS jobs |
| db03214 | `scripts/driver.sh` | Default `IMPLICIT_KAPPAV=${...:-yes}` to fix unbound-var crash; align driver's `GRID_HZ` default with `env_defaults.sh` (2) |

Halo audit of existing OM2-1 inputs (before submission, via JLD2 inspection):
- `grid.jld2`: Hx=13, Hy=13, Hz=7 (built for an earlier TBLOCKING=12 run)
- 3D FTS (u/v/w from totaltransport, kappa_v, mld): Hx=13, Hy=13, Hz=2
- Surface FTS (eta): Hx=13, Hy=13, Hz=7

Decision (per user): rebuild grid + all FTS from scratch with new defaults (Hx=Hy=7, Hz=2). All 4 invocations re-run preprocessing.

## Submitted runs (2026-05-15)

Driver commit at submission: `db03214` (chain summary printed `GRID halos=(7,7,2)` for all four).

| (PM, TW) | Driver invocation | Jobs | Grid job | NK_c job | Plot job |
|---|---|---|---|---|---|
| OM2-1, 1968-1977 | full pipeline incl. grid | 11 | `168394179` | `168394197` | `168394203` |
| OM2-1, 1999-2008 | `GRID_JOB=168394179`, no `grid` in chain | 10 | (chained on `168394179`) | `168394859` | `168394861` |
| OM2-025, 1968-1977 | full pipeline incl. grid | 11 | `168394981` | `168394990` | `168394993` |
| OM2-025, 1999-2008 | `GRID_JOB=168394981`, no `grid` in chain | 10 | (chained on `168394981`) | `168395138` | `168395142` |

Total: 42 PBS jobs. Manifests written under each `outputs/{PM}/{EXP}/{TW}/manifests/20260515T15…toml`. `scripts/runs/submissions.tsv` reconciled (PostToolUse hook + manual `bash scripts/runs/reconcile_submissions.sh`).

To track progress: `qstat -u $USER` (auto-reconciles submissions.tsv via the PostToolUse hook). To watch logs once jobs start running: `tail -f logs/PBS/<job>.OU`.
