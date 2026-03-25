# GPU Benchmark Results

1-year forward-map walltime (no output writers, no diagnostics).
Benchmark script: `src/run_1year_benchmark.jl`, submitted via `JOB_CHAIN=run1yrfast`.

## ACCESS-OM2-1 (360x301x50)

| Partition | gpuvolta (V100, 96GB) | gpuhopper (H200, 256GB) |
|-----------|----------------------|------------------------|
| 1x1       | **41.8 s**           | **13.8 s**             |
| 2x2       | **38.0 s**           | **37.5 s** (121.3 s outlier, node-dependent) |
| 1x2       | NaN at iter 300      | NaN at iter 300        |
| 1x4       | NaN at iter 300      | NaN at iter 300        |

### Closure variants (1x1, V100)

| Config | 1yr benchmark | Slowdown | Notes |
|--------|--------------|----------|-------|
| Baseline (horizontal + vertical diffusion) | **41.9 s** | 1.0× | |
| Monthly κV (`MONTHLY_KAPPAV=yes`) | **69.8 s** | 1.7× | Time-varying κV from monthly MLD via callback |
| GM-Redi (`GM_REDI=yes`) | **327.2 s** | 7.8× | IsopycnalSkewSymmetricDiffusivity with prescribed T/S |
| GM-Redi + monthly κV | **342.1 s** | 8.2× | Both features combined |

GM-Redi uses `SeawaterBuoyancy(LinearEquationOfState())` with T/S prescribed from monthly FTS via `IterationInterval(1)` callback. The isopycnal slope computation dominates the cost.

### Full 1-year runs (with JLD2 output writers)

| Partition | GPU  | Wall time | Status          | Job ID    | Notes                              |
|-----------|------|-----------|-----------------|-----------|-------------------------------------|
| 1x1       | V100 | 6m58s     | Complete        | 163403568 | Baseline                            |
| 2x2       | V100 | 9m21s     | Complete        | 163393978 | Old monolithic writer               |
| 2x2       | V100 | timeout   | Crash           | 163403571 | `output_prefix` bug (now fixed)     |
| 1x2       | V100 | timeout   | Stuck at iter 0 | 163403573 | JLD2 distributed output hang        |

## ACCESS-OM2-025 (1440x1081x50)

| Partition | gpuvolta (V100, 96GB) | gpuhopper (H200, 256GB)        |
|-----------|----------------------|--------------------------------|
| 1x1       | timeout (FTS load)   | **511.2 s** (8.5 min)          |
| 2x2       | timeout (FTS load)   | NaN at iter 900-1900 (when FTS loading fits in walltime) |
| 1x2       | NaN at iter 400      | NaN at iter 400                |
| 1x4       | NaN at iter 400      | NaN at iter 400                |

**No multi-GPU OM2-025 run has completed 1 year without NaN.**
All 2x2 runs that finished FTS loading hit NaN (iter 900-1900).
All 1x2/1x4 runs NaN at iter 400.

## Known Issues

### NaN in distributed runs
- **1xN partitions (y-only split):** All 1x2 and 1x4 produce NaN within a few hundred
  iterations for both models. Likely an Oceananigans bug in the distributed zipper
  (`distributed_zipper.jl`) for y-only partitioned `RightFaceFolded` grids.
- **OM2-025 2x2:** NaN at iter 900-1900. The higher-resolution grid may amplify
  partition-boundary instabilities. OM2-1 2x2 is fine, suggesting the issue is
  resolution-dependent or related to how the fold row is partitioned.
- **OM2-1 2x2:** No NaN — the only multi-GPU config that works for simulation.

### FTS loading bottleneck (OM2-025 distributed)
Distributed FTS loading (`load_fts` via CPU grid + `fold_set!` x 12 months x 4 fields)
takes ~25-30 minutes for OM2-025, consuming the entire 30-minute walltime. Each rank
loads the full global FTS (~2.3 GB/field) on CPU then copies its partition.
Potential fixes: increase walltime, load only local partition, or pre-partition files.

### V100 OM2-025 timeouts
V100 single-GPU runs for OM2-025 exceed 30 minutes of walltime. FTS loading alone
is too slow on V100. Only H200 has enough memory bandwidth for OM2-025 1x1.

### JLD2OutputWriter deadlocks on distributed GPU grids (fixed)
`serializeproperty!` for `DistributedGrid` creates a new `Distributed(CPU(); ...)`
which triggers blocking MPI collectives that deadlock. Workaround: `including=[]`.
Filed as [CliMA/Oceananigans.jl#5410](https://github.com/CliMA/Oceananigans.jl/issues/5410).

### `output_prefix` UndefVarError (fixed)
After switching to per-field output writers, the log message still referenced
`output_prefix` which was scoped inside the loop. Fixed to use `age_output_dir`.

### Rank-to-position mapping bug (fixed)
`load_distributed_part` used row-major rank mapping (`i = mod(rank, px)`) but
Oceananigans uses column-major (`i = div(rank, Ry*Rz)`). Fixed to match
Oceananigans' `rank2index`: `i = div(rank, py), j = mod(rank, py)`.

## MPI Socket Binding Impact (OM2-1 V100 2x2)

| Batch   | Elapsed (s) | Socket binding | Job ID    |
|---------|-------------|----------------|-----------|
| Before  | 265.6       | No             | 163178328 |
| Before  | 267.8       | No             | 163278510 |
| After   | 38.0        | Yes            | 163403472 |

**7x speedup** from `--bind-to socket --map-by socket`.
All scripts now include this flag for multi-GPU runs.

### H200 2x2 performance variability (OM2-1)
H200 2x2 shows high variance: 37.5s vs 121.3s for identical workloads on
different nodes. Likely caused by NVLink topology or inter-socket placement.

## Environment

- Oceananigans v0.105.4 (fork: `briochemc/Oceananigans.jl`, branch `bp/offline-ACCESS-OM2`)
- Julia 1.12.5
- CUDA 12.9.0, OpenMPI 5.0.8
- NCI Gadi (PBS Pro)
