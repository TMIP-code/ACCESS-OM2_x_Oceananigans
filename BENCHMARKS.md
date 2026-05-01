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

## ACCESS-OM2-01 (3600×2700×75)

First successful 0.1° (eddy-resolving) runs. Experiment
`01deg_jra55v140_iaf_cycle4`, TIME_WINDOW `1958-1987`, Δt = 400 s
(78900 steps for 1 yr). Submitted via the standard `JOB_CHAIN=run1yr`
path (`scripts/standard_runs/run_1year.sh`, with JLD2 output writers
enabled), at commit `016505b`.

Model config flags (all defaults):

- `VELOCITY_SOURCE=cgridtransports`
- `W_FORMULATION=wdiagnosed` (model diagnoses w internally, so the
  `diagnose_w` preprocessing step is not needed for the run)
- `ADVECTION_SCHEME=centered2`
- `TIMESTEPPER=AB2`
- `GM_REDI=no`, `MONTHLY_KAPPAV=no`
- `eta_t` source: cycle4 saves `sea_level = eta_t + p_atm/(ρ₀g)`
  rather than `eta_t`; `periodicaverage.py` falls back to `sea_level`
  with the IB-correction caveat documented in
  [src/periodicaverage.py](../src/periodicaverage.py).

### 1-year forward-map walltime

| Partition | GPU                      | 1-yr integration | Total wall | Setup | Steps | Δt    | Status   | Job ID    |
|-----------|--------------------------|------------------|------------|-------|-------|-------|----------|-----------|
| 1×4       | gpuhopper (4× H200)      | **1.670 h**      | 1:54:33    | ~14 m | 78900 | 400 s | Complete | 167345793 |
| 1×8       | gpuhopper (8× H200, 2 nodes) | **1.670 h**  | 2:00:37    | ~20 m | 78900 | 400 s | Complete | 167345796 |
| 1×8       | gpuvolta (8× V100)       | OOM at FTS load  | 0:16:38    | —     | —     | —     | Failed   | 167416139 |

**Strong-scaling note:** 1×8 didn't reduce model integration time over
1×4 (both 1.670 h). The setup phase (FTS loading on 2 nodes vs. 1)
costs the extra ~6 min observed in total wall. Either the workload
doesn't scale past 4 H200 ranks at this resolution, or MPI/halo
communication absorbs the extra GPUs — worth profiling separately.

### PBS resource usage

| Partition | Job ID    | Mem requested | Mem used | NCPUs | NGPUs | Walltime requested | Walltime used | CPU time | Comment         |
|-----------|-----------|---------------|----------|-------|-------|--------------------|---------------|----------|-----------------|
| 1×4       | 167345793 | 1024 GB       | **320 GB** (31 %) | 48 | 4 | 16:00:00       | 01:54:33      | 07:30:00 | 1 node, gpuhopper |
| 1×8       | 167345796 | 2048 GB       | **382 GB** (19 %) | 96 | 8 | 16:00:00       | 02:00:37      | 16:27:00 | 2 nodes, gpuhopper |

Memory headroom is large in both cases — could shrink mem requests if
SU pressure becomes an issue. Walltime budget is ~10× too generous;
~3:00:00 would suffice with margin.

### What didn't work

- **`diagnose_w` on a single H200** (job 167339769) OOMed at FTS load:
  attempted to allocate 78 GiB on top of 83 GiB usage, exceeding the
  140 GB H200 limit. With `wdiagnosed` we don't actually consume
  `w_from_mass_transport` from preprocessing, so this step can be
  dropped from the OM2-01 chain. If we ever switch to `wprescribed`
  with a diagnosed source, `diagnose_w` itself will need to be
  partitioned (or moved to CPU).
- **1×8 on gpuvolta** (job 167416139) OOMed at distributed FTS load
  in `setup_model.jl:142`. V100 has only 32 GB GPU memory per card —
  too small for any reasonable per-rank slab of a 3600×2700×75 grid
  at 1×8. Higher `1×N` (e.g. 1×16, 2×8) would need testing on
  gpuvolta if that path mattered.

### Preprocessing resource usage (for context)

The pipeline upstream of `run1yr` for OM2-01 cycle4 / TIME_WINDOW
`1958-1987`:

| Step      | Job ID    | Queue   | NCPUs | Mem req | Mem used | Walltime  | SUs   | Notes |
|-----------|-----------|---------|-------|---------|----------|-----------|-------|-------|
| prep      | 166489939 | megamem | 32    | 2048 GB | 943 GB   | 7:48:53   | 1250  | `periodicaverage.py` — monthly clim + yearly avg |
| grid      | 167339766 | express | 12    | 47 GB   | 1.6 GB   | 0:01:18   | small | `create_grid.jl` |
| vel       | 167339767 | hugemem | 16    | 512 GB  | 339 GB   | 0:58:43   | medium| `prep_velocities.jl` |
| clo       | 167345779 | hugemem | 16    | 512 GB  | 340 GB   | 0:39:48   | medium| `build_closures.jl` (47 GB express OOMed; bumped to vel envelope) |
| partition | 167345792 | megamem | 4     | 1800 GB | 1.50 TB  | 0:37:13   | medium| 1×4 — each rank loads global FTS |
| partition | 167345794 | megamem | 8     | 3000 GB | 2.05 TB  | 0:50:09   | medium| 1×8 — same pattern, 8 ranks   |

Compared to OM2-025 baselines (158 GB / 36 min for `prep`, 47 GB / 49
min for `vel`), OM2-01 lands at roughly the predicted ×6 memory and
×10 walltime ratios.

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
