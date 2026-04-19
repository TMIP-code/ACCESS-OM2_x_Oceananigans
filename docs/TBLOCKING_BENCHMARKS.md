# Temporal blocking (K-sub-step batching) for tracer advance

## Intent

On distributed tripolar runs with prescribed velocities, the only MPI
communication per tracer sub-step is the tracer halo exchange inside
`update_state!`. We reduce MPI traffic by a factor of `K` by running
`K` AB2 sub-steps on an extended halo between MPI exchanges — the same
"extended halo" technique Oceananigans already uses in
`iterate_split_explicit_in_halo!`. When the original per-step loop is
MPI-bound, this yields a near-proportional speedup.

Ported from the validated MWE in the Oceananigans dev tree:

- [/g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar.md](/g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar.md)
- [/g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar_distributed.jl](/g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar_distributed.jl)

The MWE reproduces serial results bit-for-bit on 1×4 and 2×2 partitions
for K=10, N=200 with a Gaussian blob on a toy tripolar grid
(`max_diff = 0.0` vs. serial).

Implemented here as [src/temporal_blocking.jl](../src/temporal_blocking.jl);
gated by env var `TBLOCKING=<K>` in
[src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl).

## Status — 1-year benchmarks (K=12)

Common config: `VELOCITY_SOURCE=totaltransport`,
`W_FORMULATION=wdiagnosed`, `ADVECTION_SCHEME=centered2`,
`TIMESTEPPER=AB2`, grid halos `H=(13,13,2)`.

### OM2-1 (Δt = 5400 s; 487 batches × K=12 sub-steps = 5844 steps)

`TIME_WINDOW=1968-1977`. Speedup is relative to the 1×1 (serial)
plain baseline (121.3 s). "Scaling eff." = `(speedup − 1) / (NGPUs − 1)`:
1.00 = perfect strong scaling, 0.00 = no speedup, negative = slowdown.
Undefined for 1×1 (NGPUs=1).

| Partition | GPU | Mode | Time (s) | Speedup | Scaling eff. | Job ID | Julia log | PBS log |
|-----------|-----|------|----------|---------|--------------|--------|-----------|---------|
| 1x1 (serial) | V100 | plain | 121.3 | 1.00× | — | 166460607 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166460607.gadi-pbs.log) | [link](../logs/PBS/166460607.gadi-pbs.OU) |
| 1x1 (serial) | V100 | K=12 | 120.9 | 1.00× | — | 166460594 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166460594.gadi-pbs.log) | [link](../logs/PBS/166460594.gadi-pbs.OU) |
| 1x2 | V100 | plain | 105.0 | 1.16× | 0.16 | 166460691 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166460691.gadi-pbs.log) | [link](../logs/PBS/166460691.gadi-pbs.OU) |
| 1x2 | V100 | K=12 | 103.8 | 1.17× | 0.17 | 166460610 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166460610.gadi-pbs.log) | [link](../logs/PBS/166460610.gadi-pbs.OU) |
| 1x4 | V100 | plain | 101.3 | 1.20× | 0.07 | 166460692 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166460692.gadi-pbs.log) | [link](../logs/PBS/166460692.gadi-pbs.OU) |
| 1x4 | V100 | K=12 | 98.6 | 1.23× | 0.08 | 166460612 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166460612.gadi-pbs.log) | [link](../logs/PBS/166460612.gadi-pbs.OU) |
| 2x2 | V100 | plain | 107.5 | 1.13× | 0.04 | 166460693 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166460693.gadi-pbs.log) | [link](../logs/PBS/166460693.gadi-pbs.OU) |
| 2x2 | V100 | K=12 | 97.6 | 1.24× | 0.08 | 166458644 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166458644.gadi-pbs.log) | [link](../logs/PBS/166458644.gadi-pbs.OU) |
| 1x8 | V100 | plain | — | — | — | pending | — | — |
| 1x8 | V100 | K=12 | — | — | — | 166462624 | [link](../logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166462624.gadi-pbs.log) | [link](../logs/PBS/166462624.gadi-pbs.OU) |

> **Why 1x8 and not 1x6.** gpuvolta's per-queue rule requires `ncpus` to
> be a multiple of 48 (one full node = 4 V100 + 48 CPUs) whenever a job
> spans more than one node. 6 GPUs × 12 CPUs/GPU = 72 CPUs is 1.5
> nodes — rejected. 1x8 (8 GPUs = 2 full nodes, 96 CPUs) fits cleanly
> and still exercises the >4-rank path.

### OM2-025 (Δt = 1800 s; 1461 batches × K=12 sub-steps = 17532 steps)

`TIME_WINDOW=1968-1977`. Speedup is relative to the 1×1 (serial)
plain baseline (498.5 s). "Scaling eff." = `(speedup − 1) / (NGPUs − 1)`:
1.00 = perfect strong scaling, 0.00 = no speedup, negative = slowdown.
Undefined for 1×1 (NGPUs=1).

| Partition | GPU | Mode | Time (s) | Speedup | Scaling eff. | Job ID | Julia log | PBS log |
|-----------|-----|------|----------|---------|--------------|--------|-----------|---------|
| 1x1 (serial) | H200 | plain | 498.5 | 1.00× | — | 166461744 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166461744.gadi-pbs.log) | [link](../logs/PBS/166461744.gadi-pbs.OU) |
| 1x1 (serial) | H200 | K=12 | 500.0 | 1.00× | — | 166461743 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166461743.gadi-pbs.log) | [link](../logs/PBS/166461743.gadi-pbs.OU) |
| 1x2 | H200 | plain | running | — | — | 166462103 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166462103.gadi-pbs.log) | [link](../logs/PBS/166462103.gadi-pbs.OU) |
| 1x2 | H200 | K=12 | 297.3 | 1.68× | 0.68 | 166461746 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166461746.gadi-pbs.log) | [link](../logs/PBS/166461746.gadi-pbs.OU) |
| 1x4 | H200 | plain | running | — | — | 166462104 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166462104.gadi-pbs.log) | [link](../logs/PBS/166462104.gadi-pbs.OU) |
| 1x4 | H200 | K=12 | 185.5 | 2.69× | 0.56 | 166461748 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166461748.gadi-pbs.log) | [link](../logs/PBS/166461748.gadi-pbs.OU) |
| 1x4 | V100 | plain | — | — | — | 166462783 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166462783.gadi-pbs.log) | [link](../logs/PBS/166462783.gadi-pbs.OU) |
| 1x4 | V100 | K=12 | — | — | — | 166462782 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166462782.gadi-pbs.log) | [link](../logs/PBS/166462782.gadi-pbs.OU) |
| 2x2 | H200 | plain | queued | — | — | 166462105 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_1yearfast_166462105.gadi-pbs.log) | [link](../logs/PBS/166462105.gadi-pbs.OU) |
| 2x2 | H200 | K=12 | 191.6 | 2.60× | 0.53 | 166461350 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166461350.gadi-pbs.log) | [link](../logs/PBS/166461350.gadi-pbs.OU) |

Updated as jobs complete.

### Why K=12 (not K=10)

Both resolutions: 1-year stop_time is pinned to `12 × prescribed_Δt`
from the FTS, which must divide the model Δt for the
`total_steps % K == 0` assertion to hold.

- OM2-1 at Δt=5400 s → **5844 steps/yr = 2²·3·487**. Divisors near 10: `{6, 12}`.
- OM2-025 at Δt=1800 s → **17532 steps/yr = 2²·3²·487**. Divisors near 10: `{6, 9, 12, 18}`.

K=10 doesn't divide either. The smallest clean K ≥ 10 common to both is
**12** (487 / 1461 batches respectively), which preserves ~92% MPI
reduction vs. plain `time_step!`.

## How to run

### 1. One-time preprocessing with wider halos

Halo size is baked into `grid.jld2` at build time and the partitioned
FTS files are sized to match. For `K` sub-steps per batch we need
`H ≥ K + 1` on each partitioned horizontal direction. For K=12 that's
`Hx = Hy = 13` (`Hz=2` is always enough because z is Bounded and the
KP is never extended in z).

```bash
GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
  VELOCITY_SOURCE=totaltransport \
  PARTITION=2x2 \
  PARENT_MODEL=ACCESS-OM2-1 \
  JOB_CHAIN=grid-vel-clo-diagnose_w-partition \
  bash scripts/driver.sh
```

This rebuilds grid, monthly/yearly FTS, closures, and the 2×2
partitioned FTS. For additional partitions submit `JOB_CHAIN=partition`
separately:

```bash
for P in 1x2 1x4; do
  PARTITION=$P GRID_HX=13 GRID_HY=13 GRID_HZ=2 VELOCITY_SOURCE=totaltransport \
    PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=partition bash scripts/driver.sh
done
```

(Each partition writes to its own `preprocessed_inputs/.../partitions/{PxQ}/`
directory, so they can run in parallel.)

**Note:** `diagnose_w` is *not* a hard dependency of `partition` — the
w FTS that `partition` reads
([w_from_total_transport_monthly.jld2](../src/prep_velocities.jl#L161))
is produced by the `vel` step. `diagnose_w` produces a separate file
used only when `W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=diagnosed`.
Skip it (`JOB_CHAIN=grid-vel-clo-partition`) if you won't need it.

### 2. Submit the benchmark matrix

```bash
for P in 1x1 1x2 1x4 2x2; do
  for T in 12 no; do
    PARTITION=$P TBLOCKING=$T GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
      VELOCITY_SOURCE=totaltransport PARENT_MODEL=ACCESS-OM2-1 \
      JOB_CHAIN=run1yrfast bash scripts/driver.sh
  done
done
```

Each run reports `elapsed_seconds` at completion. Paths:

- Julia log: `logs/julia/ACCESS-OM2-1/{EXPERIMENT}/{TIME_WINDOW}/standardrun/totaltransport_wdiagnosed_centered2_AB2[_TB12]_1yearfast_<jobid>.gadi-pbs.log`
- PBS stdout/stderr: `logs/PBS/<jobid>.gadi-pbs.{OU,ER}`

## Correctness ingredients — every single one matters

Lifted from the MWE's "Key ingredients" section. Missing any one
produces a silent divergence that shows up in the 2×2 case first.

1. **Halo size ≥ K+1** on each partitioned horizontal direction. K halo
   points consumed per batch + 1 safety margin.
2. **Extended `KernelParameters`**: sub-step k spans
   `(1 - margin) : (N + margin)` with `margin = K - k + 1`. Sub-step 1
   is widest; sub-step K has margin 1.
3. **Bypass `complete_communication_and_compute_tracer_buffer!`**. Call
   `compute_hydrostatic_tracer_tendencies!(model, kp; active_cells_map=nothing)`
   and `compute_tracer_flux_bcs!(model)` directly — the standard path
   syncs MPI halos AND recomputes the buffer region with `:xyz` KPs,
   overwriting extended-KP work.
4. **Parent-array cache `G⁻ ← Gⁿ`**: `parent(G⁻) .= parent(Gⁿ)`, not
   `:xyz` / interior. AB2 reads G⁻ at halo cells on the next sub-step.
5. **`fill_halo_regions!(tracers; only_local_halos=true)` between
   sub-steps** — applies Periodic/Bounded/Fold BCs locally without
   triggering MPI.
6. **Extended-xy z-halo refill between sub-steps**. The standard z-fill
   covers only interior xy; corner cells at `(x-halo, y-halo, z-halo)`
   are otherwise never refreshed and diverge exponentially. Fix in
   [fill_z_halos_over_extended_xy!](../src/temporal_blocking.jl#L64).
7. **`implicit_step!` after each explicit AB2 step** per tracer —
   needed by the project's
   `VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization())`.
8. **`maybe_prepare_first_time_step!` at top of every batch** — emulates
   `time_step!`'s first-step handling for `last_Δt`.
9. **AB2 χ selection per sub-step** — identical formula to plain
   `time_step!`:
   `euler = (Δt ≠ clock.last_Δt); χ = euler ? -0.5 : timestepper.χ`.
10. **`update_state!` at the end of every batch** — the single MPI halo
    exchange for tracers (and any FTS internals). Also
    `fill_halo_regions!(model.timestepper.G⁻[name])` for each tracer so
    next batch's AB2 has consistent G⁻ halos after the MPI sync.
11. **Zipper sign flip on velocities**. Satisfied by
    [src/prep_velocities.jl:185-187](../src/prep_velocities.jl#L185)
    (u,v written with `FPivotZipperBoundaryCondition(-1)`; w with `+1`).
12. **Low-order advection only** — `Centered(order=2)` (B=1 stencil).
    WENO3/5 would need `K·B + 1` halos. Project already restricts to
    centered2 for matrix-density reasons.

## Project-specific handling

`multi_time_step!` bypasses `run!` / `Simulation` callbacks, so
anything those callbacks do needs an in-batch, MPI-free replacement.

**Enforced when TBLOCKING > 0** (asserts in
[src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl)):

- `ADVECTION_SCHEME=centered2`, `TIMESTEPPER=AB2`, `Hx ≥ K+1`,
  `Hy ≥ K+1`, `total_steps % K == 0`.

**Warned (not asserted):**

- `VELOCITY_SOURCE=totaltransport` recommended — with `GM_REDI=no` and
  `cgridtransports` you lose the GM eddy transport entirely.

**In-batch FTS field updates** (optional, passed via `fts_updates`
kwarg to `multi_time_step!`). Each does `set!(target, source_fts[t])`
plus `fill_halo_regions!(target; only_local_halos=true)`:

- `κV` when `MONTHLY_KAPPAV=yes` — replaces the `update_κV!` callback.
- `T, S` when `GM_REDI=yes` — replaces the `prescribe_TS!` callback.
  Untested; enable with caution.

**Intentionally NOT updated per sub-step:**

- **η** — wrapped in `PrescribedFreeSurface.displacement` as a
  `TimeSeriesInterpolation` (not a raw Field). Updating it requires
  `step_free_surface!`, but that alone doesn't refresh the diagnosed
  w or the zstar grid scaling. η/w/z-scaling resync once per batch via
  `update_state!`; since velocities vary on monthly timescales and
  K·Δt ≪ 1 month, the in-batch staleness is negligible.

## Implementation map

| What | Where |
|------|-------|
| `multi_time_step!` + helpers | [src/temporal_blocking.jl](../src/temporal_blocking.jl) |
| TBLOCKING branch + asserts | [src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl) |
| Configurable grid halos | [src/create_grid.jl:89-97](../src/create_grid.jl#L89) (`GRID_HX/HY/HZ` env) |
| Env var defaults + MODEL_CONFIG tag | [scripts/env_defaults.sh](../scripts/env_defaults.sh) (`_TB<K>` suffix) |
| COMMON_VARS propagation | [scripts/driver.sh](../scripts/driver.sh) |

## Constraints & open items

- Only the 1-year benchmark path is TBLOCKING-aware. `run_1year.jl`,
  `run_10years.jl`, `solve_periodic_NK.jl`, etc. still use the standard
  `run!(simulation)` path. If TBLOCKING pays off, the next step is to
  factor the batch loop into `periodic_solver_common.jl` so the NK
  forward map benefits.
- Correctness validation is inherited from the MWE, not re-run here.
  Follow-up: add a `TBLOCK_DUMP=yes` path that writes
  `interior(model.tracers.age)` and a `compare_tblock_tripolar.jl`-style
  diff to gate against silent mis-ports.
- `total_steps % K == 0` is a hard assert; for OM2-025 (different Δt)
  the valid K values change. Check the divisors of `stop_time / Δt`
  before picking K for a new resolution.

## References

- Malas, T., Hager, G., Ltaief, H., Stengel, H., Wellein, G., & Keyes, D. (2015). *Multicore-Optimized Wavefront Diamond Blocking for Optimizing Stencil Updates*. SIAM Journal on Scientific Computing, 37(4), C439–C464. DOI: [10.1137/140991133](https://doi.org/10.1137/140991133) — canonical journal reference for the combined diamond-tiling + wavefront temporal-blocking family that Oceananigans' `iterate_split_explicit_in_halo!` (and our `multi_time_step!`) descends from.
- Wellein, G., Hager, G., Zeiser, T., Wittmann, M., & Fehske, H. (2009). *Efficient Temporal Blocking for Stencil Computations by Multicore-Aware Wavefront Parallelization*. COMPSAC 2009. DOI: [10.1109/COMPSAC.2009.82](https://doi.org/10.1109/COMPSAC.2009.82) — multi-core wavefront paper that Malas et al. (2015) builds on.
- Wonnacott, D. (2000). *Using time skewing to eliminate idle time due to memory bandwidth and network limitations*. IPDPS 2000. DOI: [10.1109/IPDPS.2000.846009](https://doi.org/10.1109/IPDPS.2000.846009) — foundational "time skewing" idea that became temporal blocking.
