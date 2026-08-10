# Converged OM2-01 periodic NK solve with an `upwind3` forward map

**Headline:** the periodic Newton–Krylov age solve **converges at 0.1°
(ACCESS-OM2-01)** with the forward map running the less-diffusive
`UpwindBiased(order=3)` (`upwind3`) advection, while NK reuses the **already-built
`upwind1` averaged transport matrix** as its preconditioner. Final
`vol_rms_drift = 3.4e-9`, `ReturnCode.Success`. This decouples the *simulated*
physics (`upwind3`, low numerical diffusion) from the *preconditioner* (`upwind1`,
whose implicit diffusion conditions the Jacobian well) — no new matrix build.

Window: `1968-1977`. Date: 2026-06. See
[`First_OM2-01_NK_solve.md`](First_OM2-01_NK_solve.md) for the original `upwind1`
solve and the cross-resolution JVP table (this doc adds the `upwind3` rows).

---

## 1. What "upwind3 forward / upwind1 preconditioner" means

The periodic NK solver drives `G(x) = Φ(x) − x = 0`, where `Φ!` is one year of GPU
forward integration of the age tracer. Two independent advection choices:

- **Forward map `Φ!`/`G!`** (`ADVECTION_SCHEME`): the scheme actually simulated.
  `upwind3` is chosen because `upwind1` is too diffusive; `upwind3` carries much
  less implicit numerical diffusion.
- **Preconditioner** (`TM_ADVECTION_SCHEME`): the coarsened transport matrix that
  accelerates GMRES. It only changes *how fast* GMRES converges, **not the fixed
  point**, so it can use a *different* scheme than the forward map. Here it reuses
  the `upwind1` averaged matrix, whose implicit diffusion better-conditions
  `∂G/∂x` (the reason `upwind1` unlocked OM2-01 in the first place).

`TM_ADVECTION_SCHEME=upwind1` swaps **only the advection token** of the
`MODEL_CONFIG` tag so NK points at the pre-built `upwind1` matrix:

| Role | MODEL_CONFIG |
|---|---|
| Forward map (`ADVECTION_SCHEME=upwind3`) | `cgridtransports_wparent_upwind3_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS` |
| Preconditioner (`TM_ADVECTION_SCHEME=upwind1`, `TM_SOURCE=avg`) | `cgridtransports_wparent_upwind1_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS` |

---

## 2. Prerequisites

1. **The `upwind1` averaged matrix must already exist** at the swapped tag:
   `…/TM/cgridtransports_wparent_upwind1_AB2_kH30_…_LBS/avg/M.jld2` (~39 GB, built
   run1yr-free by [`create_monthly_matrices.jl`](../src/create_monthly_matrices.jl);
   see `First_OM2-01_NK_solve.md` §5). Only `avg` is needed — there is no `const`
   OM2-01 matrix. If it is missing, build it first (that megamem job is the
   expensive part, not the NK).

2. **Grid halo ≥ 3 → rebuild at `GRID_HZ=4`.** `UpwindBiased(order=3)` has stencil
   buffer 2; with the immersed-boundary +1 that needs grid halo **≥ 3**. The
   default `GRID_HZ=2` (fine for `upwind1`) is too small and fails at model build.
   Use `GRID_HZ=4` (harmless larger halo, also serves WENO5). Rebuilding the grid
   means rebuilding the
   velocities, closures **and the partitioned per-rank data** — everything from
   `grid` onward. A larger halo is harmless to the interior, so the rebuilt
   grid/vel work for every scheme.

3. **`upwind3` is registered** (`ADVECTION_SCHEME=upwind3` →
   `UpwindBiased(order=3)` in [`config.jl`](../src/shared_utils/config.jl); the
   `TM_ADVECTION_SCHEME` whitelist accepts `upwind1`).

---

## 3. Pipeline

OM2-01 is distributed (`PARTITION=1x4`, four H200 ranks), so the chain is:

```
grid → vel → clo → partition → NK
```

- `grid`/`vel`/`clo` are **serial** (a single Julia process); they pin
  `PARTITION=1x1` internally so they don't inherit the model's `1x4` run
  partition. `vel`/`clo` run on `hugemem` (512 GB).
- `partition` (megamem) slices the serial global velocity FTS into per-rank files
  for the `1x4` layout — required before any distributed run/NK. **This step is
  what a `GRID_HZ` change forces you to redo**; skipping it makes NK die with a
  `Partition halo-size mismatch` (stale Hz=2 rank files vs the Hz=4 grid).
- `diagnose_w` is **not** needed: the default is `W_FORMULATION=wprescribed` +
  `PRESCRIBED_W_SOURCE=parent`, so the forward map reads the parent `w` straight
  from `vel`.

No `TMbuild`/`TMsnapshot` is in the chain — the preconditioner matrix already
exists on disk and NK loads it via the `TM_ADVECTION_SCHEME` swap.

---

## 4. Convergence (multi-restart)

Each 48 h GPU job advances roughly **one Newton iteration** — `upwind3` costs
~2× the JVPs per iteration of `upwind1` (e.g. iter 1 = 23 JVPs vs `upwind1`'s 13),
so a full solve spans several restarts. `TRACE_SOLVER_HISTORY=yes` saves
`newton_iterate_NN.jld2` after each iteration; `INITIAL_AGE=latest` resumes from
the newest.

| Job | Kind | Outcome | `vol_rms_drift` |
|---|---|---|---|
| 171609127 | first NK | bus error (SIGBUS), no iterate saved | — |
| 171702799 | resubmit (`INITIAL_AGE=0`) | walltime; iter 1 done → `newton_iterate_01` | 0.98 → 1.25e-2 |
| 171938097 | restart (`INITIAL_AGE=latest`) | bus error, no new iterate | — |
| 172079349 | restart (`INITIAL_AGE=latest`) | walltime; iter 2 done → `newton_iterate_02` | 1.25e-2 → **1.89e-6** |
| 172371015 | restart (`INITIAL_AGE=latest`) | **converged, exit 0** | 1.89e-6 → **3.4e-9** ✓ |

Converged steady-state age (~13 GB):
`…/periodic/cgridtransports_wparent_upwind3_AB2_kH30_…_LBS/1x4/NK_Q4x4/age_Pardiso_Q4x4.jld2`

> Note: the solve drove past the ~1e-6 residual "floor" seen at OM2-1/OM2-025
> (`upwind3`/`upwind1`-avg bottomed out at 1.2–1.6e-6 there) down to 3.4e-9 —
> essentially the same final residual as the `upwind1` OM2-01 solve (3.0e-9). That
> confirms the ~1e-6 at coarser resolutions was just where those Newton loops
> *terminated*, not a preconditioner-mismatch limit.

### Restart protocol

- **Walltime kill (exit −29):** resubmit with `INITIAL_AGE=latest` — resumes from
  the newest saved iterate.
- **Bus error (SIGBUS, intermittent gpuhopper node fault):** if no *new* iterate
  was saved, a plain resubmit still resolves `INITIAL_AGE=latest` to the last good
  iterate, so the same restart command is safe either way. Re-running lands on a
  different node and usually clears it.

---

## 5. Reproduce / re-run for another experiment

Replace `TIME_WINDOW` (and `PARENT_MODEL` if applying at another resolution). The
per-model defaults (`cgridtransports`, `1x4`, `Q4x4`, κH30, gpuhopper, megamem
partition, 48 h NK) come from [`model_configs/ACCESS-OM2-01.sh`](../model_configs/ACCESS-OM2-01.sh).

```bash
# 0. (once) ensure the upwind1 averaged matrix exists for this window, else build it:
#    ADVECTION_SCHEME=upwind1 TM_SOURCE=avg LUMP_AND_SPRAY=4x4 \
#    TMSNAP_QUEUE=megamem TMSNAP_MEM=2990GB TMSNAP_NCPUS=48 \
#    WALLTIME_TM_SNAPSHOT=24:00:00 SAVE_INTERMEDIATE_MATRICES=no \
#    JOB_CHAIN=TMsnapshot bash scripts/driver.sh   # (see First_OM2-01_NK_solve.md §7)

# 1. rebuild grid→vel→clo→partition at GRID_HZ=4, then run NK.
#    PARTITION_MEM=1600GB covers the Hz=4 1x4 partition (default 400/rank = 1.6 TB).
GRID_HZ=4 ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
  PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=1968-1977 \
  JOB_CHAIN=grid-vel-clo-partition-NK bash scripts/driver.sh

# 2. each time the NK job hits walltime, restart from the latest Newton iterate:
GRID_HZ=4 ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
  INITIAL_AGE=latest PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=1968-1977 \
  JOB_CHAIN=NK bash scripts/driver.sh
```

If the grid/vel/clo already exist at `GRID_HZ=4` for this window, drop them from
the chain (`JOB_CHAIN=partition-NK`, or just `NK` if the partition is current).

The same recipe works at OM2-1/OM2-025 — there the forward-scheme rebuild is
cheap and OM2-1 (`1x1`) needs no `partition` step. Cross-resolution JVP costs
(all converged): OM2-1 ΣJVP 44 `[18,26,0]`; OM2-025 ΣJVP 55 `[23,32,0]`; OM2-01
as above. (Table in `First_OM2-01_NK_solve.md` §2.)

---

## 6. Resource notes / gotchas

- **Partition memory.** The partition step holds the full global velocity FTS
  (12 monthly snapshots × u/v/w) on **every** rank, so peak ≈ `RANKS × 12 ×
  global-field`. With the per-component `GC.gc()` fix
  ([`partition_data.jl`](../src/partition_data.jl)) the Hz=4 `1x4` partition used
  **764 GB** (was ~1.3 TB). Default is `PARTITION_MEM_PER_RANK=400` (→1.6 TB for
  `1x4`), which has headroom.

- **`plotNK` is heavy at 0.1°.** It renders full-res figures + animations from the
  25-snapshot 1-year FTS: needs **hugemem (~317 GB observed, request ≥512 GB) and
  a long walltime** (`PLOT_NK_QUEUE=hugemem PLOT_NK_MEM=512GB PLOT_NK_NCPUS=18
  WALLTIME_PLOT_NK=12:00:00`). The static panels are written first; the
  animations are the slow part and may not finish in one job.

- **DataDeps prompt.** `OceanBasins.oceanpolygons()` needs the IHO "Oceans and
  seas" polygon dataset; if it is not cached it blocks on an interactive `[y/n]`
  and the batch job dies (exit 1, ~2 GB, ~2 min). Compute nodes have no internet —
  pre-fetch once on a login node:
  `DATADEPS_ALWAYS_ACCEPT=true julia --project -e 'using OceanBasins; oceanpolygons()'`.

- **Post-NK diagnostics chain:**
  `JOB_CHAIN=run1yrNK-combine1yr-ventilation-plotNK-plotventilation`. The
  `plotNKtrace` step is currently broken (references a nonexistent
  `scripts/plotting/plot_trace_history_job.sh`) and, under `set -e`, aborts the
  chain — omit it, or submit `plotventilation` separately `afterok` the
  `ventilation` job.

---

## 7. Key job IDs (1968-1977, reproducibility)

| Job(s) | What | Result |
|---|---|---|
| 170345011 | OM2-01 `upwind1` monthly avg-matrix build (preconditioner) | 39 GB `avg/M.jld2` |
| 171607917 / 918 / 919 / 171609126 | grid / vel / clo / partition at `GRID_HZ=4` | partition 764 GB, exit 0 |
| 171702799 → 172079349 → 172371015 | NK first run + two `INITIAL_AGE=latest` restarts | **converged, drift 3.4e-9 ✓** |
| 171609127, 171938097 | intermittent SIGBUS (resubmitted) | no progress lost |
| 172582615–618 | run1yrNK → combine1yr → ventilation → plotNK (post-NK) | run1yr/combine OK; plot needs hugemem |
