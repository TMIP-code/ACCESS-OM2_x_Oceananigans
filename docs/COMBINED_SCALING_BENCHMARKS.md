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

New helper [src/shared_utils/load_balance.jl](../src/shared_utils/load_balance.jl)
supports two load proxies:

- **LBS** (`LOAD_BALANCE=surface`, tag `_LBS`): balance
  `wet[j] = count(bottom[:, j] .< 0)` — **surface wet columns only**.
  Treats a shallow column the same as a deep one.
- **LB** (`LOAD_BALANCE=cell`, tag `_LB`): balance the 3D wet-cell
  count `wet[j] = Σᵢ count(z_centers .> bottom[i, j])`. Weights each
  column by its z-depth. Matches what the ClimaOcean snippet does.

Both feed the same greedy splitter. The result — a tuple of per-rank
`local_Ny` — is passed to `Partition(y = Sizes(local_Ny...))`. `bottom`
(and for `:cell`, `z_faces`) are read from `grid.jld2` on every rank;
the computation is deterministic, so all ranks agree on `local_Ny`
without MPI communication.

Wired into:

- [src/select_architecture.jl](../src/select_architecture.jl) — when
  `LOAD_BALANCE ∈ (surface, cell)`, builds
  `arch = Distributed(device, partition = Partition(y = Sizes(local_Ny...)))`.
- [src/partition_data.jl](../src/partition_data.jl) — same construction
  for the CPU-side preprocessing rank arch; output dirs gain the mode
  tag (`partitions/{PxQ}_LB/` or `partitions/{PxQ}_LBS/`).
- [src/setup_model.jl](../src/setup_model.jl) — reads from the
  mode-tagged partition dir.
- [scripts/env_defaults.sh](../scripts/env_defaults.sh) —
  `LOAD_BALANCE=${LOAD_BALANCE:-no}` accepts `no | surface | cell`
  (`yes` is a deprecated alias for `surface`, preserving the original
  implementation's behaviour). Appends `_LB` or `_LBS` to
  `MODEL_CONFIG`.
- [scripts/driver.sh](../scripts/driver.sh) — propagates `LOAD_BALANCE`
  via `COMMON_VARS`.

**Bottom-sign assertion.** The load helper errors loudly if the computed
per-y-row load sums to zero: the saved `bottom` follows the Oceananigans
convention (z increases upward → `bottom < 0` ⇒ ocean), and we do not
silently flip the sign.

**Restriction.** Greedy y-slab only; both modes require `PARTITION_X=1`
(error otherwise).

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
| 1x2 | H200 | K=12 + syncGC=4 | 300.7 | 1.66× | 0.66 | 166492255 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_166492255.gadi-pbs.log) | [link](../logs/PBS/166492255.gadi-pbs.OU) |
| 1x2 | H200 | K=12 + syncGC=4 + LBS | 323.7 | 1.54× | 0.54 | 166492746 | [link](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_LB_1yearfast_166492746.gadi-pbs.log) | [link](../logs/PBS/166492746.gadi-pbs.OU) |

> **Labelling note.** The submitted job carries the `_TB12_LB` filename
> tag because it ran before the `LBS`/`LB` distinction existed — the
> original implementation was purely surface-wet-column based, now
> renamed to LBS. The on-disk partition dir has been renamed
> `1x2_LBS/` to match.

### Findings — 1×2

- **Sync-GC alone is a wash on 1x2 OM2-025 H200.** K=12+syncGC=4 lands
  at 300.7 s vs. the K=12 baseline of 297.3 s — well within run-to-run
  noise, neither gain nor regression. Interpretation: at 1x2 on H200 the
  per-step GC pressure is already low enough that forcing a collective
  pause at every 4th batch neither hides nor creates new stalls. Sync-GC
  may still matter at higher rank counts (1x4, 2x2) where stragglers
  have more ranks to block.
- **LBS partition is `local_Ny=(453, 627)`** (from
  [logs/julia/.../preprocess/partition_data_1x2_166492745.gadi-pbs.log](../logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/preprocess/partition_data_1x2_166492745.gadi-pbs.log)).
  Rank 0 (southern, dense ocean) gets 453 rows; rank 1 (northern,
  Arctic + land) gets 627. Both ≥ `Hy + 2 = 15`, so the fold-halo
  warning stays silent. `sum == 1080` checks out.
- **LBS regresses by ~9% on 1x2 OM2-025 H200** (323.7 s vs. 297.3 s
  K=12 baseline; +26 s ≈ +8.9%). The surface-column imbalance was not
  the bottleneck on this partition — forcing an uneven split skews
  kernel runtimes on the larger slab (rank 1 has 627/1080 ≈ 58% of y)
  without a compensating gain. Candidate causes to check before
  retrying:
  - Per-rank kernel runtime scales with *total* cells (wet + dry) on a
    GPU (no masking in the horizontal), so wet-column balance ≠ runtime
    balance when masking is cheap.
  - The zipper-fold rank (north, rank 1 here) already has extra halo
    work; giving it the larger slab compounds that.
  - Non-round slab sizes (453, 627 vs. 540, 540) may harm occupancy.
- **Cell-based LB not yet tried at 1×2.** LBS's surface proxy may be a
  bad fit for the real load; cell-based LB weights each column by its
  depth and is the closer analogue of the ClimaOcean recipe. Probed
  first at 1×4 (below).

## Phase A — 1×4 extension

New re-benchmark at 1×4 H200 (where the y-imbalance bites harder per
rank). Reuses the existing `1x4` equal-split partition for the baseline
(297.3/plain from TBLOCKING_BENCHMARKS.md not applicable — that's 1×2;
the 1×4 plain baseline is 227.9 s, K=12 is 185.5 s).

| Partition | GPU | Mode | Time (s) | Speedup | Scaling eff. | Job ID | Julia log | PBS log |
|-----------|-----|------|----------|---------|--------------|--------|-----------|---------|
| 1x4 | H200 | plain | 227.9 | 2.19× | 0.40 | 166462104 | (see TBLOCKING_BENCHMARKS.md) | (see TBLOCKING_BENCHMARKS.md) |
| 1x4 | H200 | K=12 | 185.5 | 2.69× | 0.56 | 166461748 | (see TBLOCKING_BENCHMARKS.md) | (see TBLOCKING_BENCHMARKS.md) |
| 1x4 | H200 | K=12 + syncGC=4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1x4 | H200 | K=12 + syncGC=4 + LBS | TBD | TBD | TBD | TBD | TBD | TBD |
| 1x4 | H200 | K=12 + syncGC=4 + LB | TBD | TBD | TBD | TBD | TBD | TBD |

### Submission commands

Generic form (set `PARTITION` and `LOAD_BALANCE` as appropriate):

```bash
PARENT_MODEL=ACCESS-OM2-025 PARTITION=<1x2|1x4> \
[LOAD_BALANCE=surface|cell] \
VELOCITY_SOURCE=totaltransport ADVECTION_SCHEME=centered2 TIMESTEPPER=AB2 \
TBLOCKING=12 GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
SYNC_GC_NSTEPS=4 \
JOB_CHAIN=[partition-]run1yrfast bash scripts/driver.sh
```

Add the `partition-` prefix (or submit `JOB_CHAIN=partition` separately
first) whenever the required partition dir is missing. Mode tag
suffixes: `_LB` for cell, `_LBS` for surface, empty for equal-split.

## Phase B — DEFERRED

> **⚠️ K and N tuning is out of scope for this change set.**
>
> When we re-open Phase B, the open questions are:
>
> - **K=24 is invalid** for OM2-025 (17532/24 = 730.5). Closest "double"
>   is **K=18** (17532/18 = 974). K=18 needs `GRID_HX=GRID_HY≥19`, which
>   means rebuilding the grid AND re-partitioning. The pre-partitioned
>   data for K=12 (with halos=13) lives in `partitions/1x2_LBS/` (and
>   whatever LB/plain variants) and would be clobbered by a K=18
>   partition step in the same dirs. We need
>   to decide between: tagging the partition dir with the halo size
>   (`partitions/1x2_LBS_H19/` etc.); running K=18 in a separate
>   workspace; accepting the clobber and re-running afterwards; or
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
