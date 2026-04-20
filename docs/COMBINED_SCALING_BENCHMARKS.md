# Combined strong-scaling optimisations

Three independent strong-scaling tricks for distributed Oceananigans
runs:

1. **Synchronised GC** ([SYNC_GC.md](SYNC_GC.md), [DISTRIBUTED_GC.md](DISTRIBUTED_GC.md)).
2. **Temporal blocking** ([TBLOCKING_BENCHMARKS.md](TBLOCKING_BENCHMARKS.md)).
3. **Land-mask-aware load balancing** (new — adapted from
   [CliMA/ClimaOcean.jl#665](https://github.com/CliMA/ClimaOcean.jl/discussions/665#discussioncomment-14737556)).

This doc covers the **combination** of all three on a single 1×2 OM2-025
H200 run, and the new wiring needed to make them stack. The per-trick
details stay in the docs above.

## Intent

Each trick on its own gave a modest gain on OM2-025 1×2 H200:

- Plain (no tricks): 337.4 s.
- Temporal blocking K=12: 297.3 s (1.13× over plain).

Sync-GC standalone numbers are profile-only (see SYNC_GC.md). Land-aware
load balancing has never been tried in this repo. Question: do they
compose, and how much do we get from all three?

## What changed in code

### 1. Sync-GC inside `multi_time_step!`

`multi_time_step!` bypasses the `Simulation` callback path, so the
`SYNC_GC_NSTEPS` callback installed in
[src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl) (lines
~110–120) never fired when `TBLOCKING > 0`. Now
[src/temporal_blocking.jl](../src/temporal_blocking.jl) accepts a
`sync_gc_nbatches::Int` kwarg and a shared `batch_index::Ref{Int}` and
fires `GC.gc(false)` immediately after the batch-end `update_state!`
collective:

```julia
batch_index[] += 1
if sync_gc_nbatches > 0 && batch_index[] % sync_gc_nbatches == 0
    GC.gc(false)
end
```

`update_state!` is already a collective-sync point on every rank, so
`GC.gc(false)` lands at the same wall-clock instant — collective pause
costs `max(GC_i)` instead of `sum(GC_i)`.

#### ⚠️ Unit change under TBLOCKING

`SYNC_GC_NSTEPS = N` means **every N batches** under TBLOCKING, **every N
steps** without. With K=12 and N=4: GC every 4 batches = every 48 raw
steps. Update SYNC_GC.md cross-references accordingly.

### 2. Land-mask-aware load balancing

New helper [src/shared_utils/load_balance.jl](../src/shared_utils/load_balance.jl):

```julia
compute_lb_y_sizes(grid_file, nranks_y; min_size=0) -> NTuple{nranks_y, Int}
```

Greedy split of `wet[j] = count(bottom[:,j] .< 0)` over `nranks_y`
contiguous y-slabs (each rank gets ≈ same wet-cell count). Adapted from
the `_assess_y_load!` + `calculate_local_N` snippet in the ClimaOcean
discussion thread, but operates on the saved `bottom` array on the CPU
(no kernel) — runs once at partition time on every rank, and is
deterministic so all ranks agree on `local_Ny` without any MPI
communication.

Wired into:

- [src/select_architecture.jl](../src/select_architecture.jl) — when
  `LOAD_BALANCE=yes`, builds `arch =
  Distributed(device, partition = Partition(y = Sizes(local_Ny...)))`.
- [src/partition_data.jl](../src/partition_data.jl) — same construction
  for the CPU-side preprocessing rank arch; output dirs gain a `_LB`
  suffix (`partitions/{PxQ}_LB/`) so LB and equal-split data sets coexist.
- [src/setup_model.jl](../src/setup_model.jl) — reads from the `_LB`
  partition dir when `LOAD_BALANCE=yes`.
- [scripts/env_defaults.sh](../scripts/env_defaults.sh) — new
  `LOAD_BALANCE=${LOAD_BALANCE:-no}` env var; appends `_LB` to
  `MODEL_CONFIG`.
- [scripts/driver.sh](../scripts/driver.sh) — propagates `LOAD_BALANCE`
  via `COMMON_VARS`.

**Bottom-sign assertion.** `compute_lb_y_sizes` errors loudly if
`sum(wet) == 0`: the saved `bottom` follows the Oceananigans convention
(z increases upward → `bottom < 0` ⇒ ocean), and we do not silently flip
the sign.

**Restriction.** Greedy y-slab only; `LOAD_BALANCE=yes` requires
`PARTITION_X=1` (errors otherwise).

### 3. Distributed code audit

The audit in step 4c of the planning checklist confirmed every
distributed-aware code path (grid construction, FTS slicing, temporal
blocking kernels) reads per-rank sizes from `arch.partition` /
`local_size` / `concatenate_local_sizes` / `size(grid)` — never assumes
equal `Ny ÷ py` slabs. No fixes were required.

## Phase A — combined-optimisations submission

Reusing existing baselines from
[TBLOCKING_BENCHMARKS.md](TBLOCKING_BENCHMARKS.md) for OM2-025 1×2 H200,
`TIME_WINDOW=1968-1977`. Same column structure as that doc's table.

Speedup baseline: 498.5 s (1×1 plain H200). Scaling eff. =
`(speedup − 1) / (NGPUs − 1)`.

| Partition | GPU | Mode | Time (s) | Speedup | Scaling eff. | Job ID | Julia log | PBS log |
|-----------|-----|------|----------|---------|--------------|--------|-----------|---------|
| 1x1 (serial) | H200 | plain | 498.5 | 1.00× | — | 166461744 | (see TBLOCKING_BENCHMARKS.md) | (see TBLOCKING_BENCHMARKS.md) |
| 1x2 | H200 | plain | 337.4 | 1.48× | 0.48 | 166462103 | (see TBLOCKING_BENCHMARKS.md) | (see TBLOCKING_BENCHMARKS.md) |
| 1x2 | H200 | K=12 | 297.3 | 1.68× | 0.68 | 166461746 | (see TBLOCKING_BENCHMARKS.md) | (see TBLOCKING_BENCHMARKS.md) |
| 1x2 | H200 | K=12 + syncGC=4 | TBD | TBD | TBD | A.1 | TBD | TBD |
| 1x2 | H200 | K=12 + syncGC=4 + LB | TBD | TBD | TBD | A.2 | TBD | TBD |

### Submission commands

**A.1 — K=12 + sync-GC N=4 (per-batch).** No new preprocessing needed
(reuses the `1x2` partition dir).

```bash
PARENT_MODEL=ACCESS-OM2-025 PARTITION=1x2 \
VELOCITY_SOURCE=totaltransport ADVECTION_SCHEME=centered2 TIMESTEPPER=AB2 \
TBLOCKING=12 GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
SYNC_GC_NSTEPS=4 \
JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

`MODEL_CONFIG` = `totaltransport_wdiagnosed_centered2_AB2_TB12` (no
`_syncGC` tag); jobid in the log filename keeps it disjoint from the
existing 297.3 s K=12 baseline.

**A.2 — K=12 + sync-GC N=4 + load-balance.** One-time prereq (login
node, ~minutes):

```bash
PARENT_MODEL=ACCESS-OM2-025 PARTITION=1x2 LOAD_BALANCE=yes \
GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
JOB_CHAIN=partition bash scripts/driver.sh
```

Then the run:

```bash
PARENT_MODEL=ACCESS-OM2-025 PARTITION=1x2 LOAD_BALANCE=yes \
VELOCITY_SOURCE=totaltransport ADVECTION_SCHEME=centered2 TIMESTEPPER=AB2 \
TBLOCKING=12 GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
SYNC_GC_NSTEPS=4 \
JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

`MODEL_CONFIG` = `totaltransport_wdiagnosed_centered2_AB2_TB12_LB`. Logs
land in a different file from A.1 thanks to the `_LB` suffix.

### Validation gates (cheap, run before the full A.1/A.2 jobs)

1. **A.1 itself is the validation for sync-GC-inside-TB.** It's a 5-min
   queue job. Look for the new `@info` line `Synchronized GC enabled
   (inside multi_time_step!): GC.gc(false) every 4 batches (= 48 raw
   steps)` and confirm `elapsed_seconds` is ≤ ~300 s.
2. **LB partition tuple.** Inspect the `partition` job log for
   `local_Ny=(...)`. Verify `sum == Ny = 1080` and both entries
   `≥ Hy + 2 = 15`. Expected: top rank larger than bottom (Arctic land
   mass ⇒ fewer wet cells in high-j slab ⇒ greedy widens it).
3. **A.2 runtime log.** Confirm the same `local_Ny` tuple appears in
   the simulation log (proves `setup_model.jl` recomputed the same
   split).

## Phase B — DEFERRED

> **⚠️ K and N tuning is out of scope for this change set.**
>
> When we re-open Phase B, the open questions are:
>
> - **K=24 is invalid** for OM2-025 (17532/24 = 730.5). Closest "double"
>   is **K=18** (17532/18 = 974). K=18 needs `GRID_HX=GRID_HY≥19`, which
>   means rebuilding the grid AND re-partitioning. The pre-partitioned
>   data for K=12 (with halos=13) lives in `partitions/1x2_LB/` and
>   would be clobbered by a K=18 partition step in the same dir. We need
>   to decide between: tagging the partition dir with the halo size
>   (`partitions/1x2_LB_H19/`); running K=18 in a separate workspace;
>   accepting the clobber and re-running A.2 + B(K=6) afterwards; or
>   another approach.
> - **K halved (K=6):** valid divisor (17532/6 = 2922). Halos
>   `HX=HY=13` already enough, so no re-preprocess needed.
> - **N halved / doubled:** trivial.

## Risks & open items

- **Sync-GC unit change under TB** (per-batch, not per-step) — see the
  warning above. SYNC_GC.md should grow a cross-reference here.
- **Fold halo on top rank under LB.** The `min_size = Hy + 2 = 15` floor
  in `compute_lb_y_sizes` keeps the warn at `grid.jl:479` silent. For
  OM2-025 1×2 this is far above any plausible imbalance — safeguard,
  not expected to fire.
- **Bottom-sign assertion fires** ⇒ stop and investigate; do **not**
  patch by flipping the sign.
- **Warmup wall-clock noise under LB.** Different per-rank Ny means
  slightly different JIT/alloc times. Not a correctness issue (the
  warmup batch syncs at `update_state!`), but worth a footnote in the
  table when interpreting A.2 numbers.
- **Halo-vs-partition coupling** — Phase B problem only.

## Post-Phase-A checklist

1. Fill in the `TBD` cells in the table above with `Time (s)`, `Speedup`,
   `Scaling eff.`, `Job ID`, and links to `Julia log` and `PBS log`
   (mirroring TBLOCKING_BENCHMARKS.md lines 39–81).
2. If A.2 ≥ A.1, the wet-cell skew on 1×2 OM2-025 isn't large enough to
   matter — record the finding here and shelve LB for this resolution.
3. (Defer) nsys profile run on the winning configuration to confirm GC
   NVTX events align across ranks at batch boundaries.
4. (Defer) Phase B K/N tuning per the warning above.
