# GPU Profiling with NVIDIA Nsight Systems

## Quick start

Profile any benchmark run by adding `PROFILE=yes`:

```bash
# Serial (1 GPU)
PARENT_MODEL=ACCESS-OM2-1 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Distributed (4 GPUs, 2x2)
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Distributed (4 GPUs, 1x4) — needs partition step if 1x4 data doesn't exist
PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x4 PROFILE=yes JOB_CHAIN=partition-run1yrfast bash scripts/driver.sh

# OM2-025 serial needs 1hr walltime (nsys finalization is slow for large grids)
PARENT_MODEL=ACCESS-OM2-025 WALLTIME_RUN_1YEAR=01:00:00 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Custom step count (default: 20 when PROFILE=yes)
PARENT_MODEL=ACCESS-OM2-025 BENCHMARK_STEPS=50 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

## How it works

Profiling uses two mechanisms:

1. **`CUDA.@profile external=true`** in `run_1year_benchmark.jl` wraps the benchmark
   section with `cudaProfilerStart()`/`cudaProfilerStop()`.
2. **`--capture-range=cudaProfilerApi --capture-range-end=stop`** in the nsys command
   tells nsys to only record data between those API calls.

This captures **only the 20 benchmark time steps** — setup, FTS loading, model
construction, and JIT warmup are excluded. Profile files are small (1-4 MB) instead
of hundreds of MB or GB.

The pattern follows [ETH's Julia+CUDA+MPI profiling approach](https://github.com/eth-vaw-glaciology/course-101-0250-00).

## What gets captured

- **Serial runs**: `nsys profile --trace=nvtx,cuda` wraps the Julia process.
- **Distributed runs**: All ranks are profiled via `bash -c` wrapper, each producing
  its own `.nsys-rep` with MPI tracing (`--trace=nvtx,cuda,mpi`).
- **GC tracing**: `JULIA_NVTX_CALLBACKS=gc` traces Julia garbage collection events
  as NVTX markers (visible in the GUI timeline).
- **Step count**: `BENCHMARK_STEPS=20` (default when `PROFILE=yes`) limits the
  benchmark to 20 time steps after 3 warmup steps.

## Output files

Profiles are saved in the log directory:
```
logs/julia/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/standardrun/
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile.nsys-rep           # serial
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_rank0.nsys-rep     # distributed, rank 0
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_rank1.nsys-rep     # distributed, rank 1
    ...
```

Typical file sizes (20 steps):
- Serial: ~1.2 MB
- Distributed: ~3 MB per rank

## Viewing profiles

### CLI summary (on Gadi compute node)

`nsys stats` is too memory-intensive for login nodes. Run it on a compute node:

```bash
# Interactive session
qsub -I -P y99 -l mem=47GB -q express -l walltime=00:30:00 \
     -l ncpus=12 -l storage=gdata/y99+scratch/y99

# Load cuda module for nsys
module load cuda/12.9.0

# Full summary (CUDA API + GPU kernels + memory ops)
nsys stats profile.nsys-rep

# GPU kernel summary only
nsys stats --report=cuda_gpu_kern_sum profile.nsys-rep

# CUDA API summary only
nsys stats --report=cuda_api_sum profile.nsys-rep

# MPI event summary (distributed runs only)
nsys stats --report=mpi_event_sum profile_rank0.nsys-rep
```

A batch script is available: `scripts/run_nsys_stats_batch.sh` (update the job IDs
in the `profiles` array, then `qsub scripts/run_nsys_stats_batch.sh`).

### GUI (on local machine)

1. Download [Nsight Systems](https://developer.nvidia.com/nsight-systems) (free)
2. Copy the `.nsys-rep` files from Gadi:
   ```bash
   scp gadi:/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia/.../profile*.nsys-rep .
   ```
3. Open with `nsys-ui profile.nsys-rep`
4. For distributed runs, open multiple rank files as a "multi-report view" to compare ranks

The GUI timeline shows:
- **Green bars**: GPU kernel execution
- **Orange bars**: CUDA memcpy (GPU-CPU transfers)
- **NVTX ranges**: GC events (from `JULIA_NVTX_CALLBACKS=gc`)
- **Gaps**: Idle time (sync, kernel launch overhead)
- **MPI rows** (distributed): MPI send/recv/waitall calls per rank

## Key metrics to look for

| Metric | What it tells you |
|--------|-------------------|
| `compute_hydrostatic_free_surface_Gc!` | Tracer tendency computation time |
| `compute_w_from_continuity!` | w diagnosis time (dominant in distributed runs) |
| `solve_batched_tridiagonal_system` | Free-surface solver time |
| `MPI_Waitall` median | Per-exchange MPI synchronization cost |
| `MPI_Waitall` max | Worst-case stall (often caused by GC on another rank) |
| GC NVTX ranges | Julia GC pauses — these propagate as MPI stalls |
| `cuStreamSynchronize` time | CPU waiting for GPU |
| `cuLaunchKernel` count | Number of kernel launches per step |

## Profiling matrix

| Model | Partition | GPUs | GPU type |
|-------|-----------|------|----------|
| OM2-1 | serial | 1 | V100 |
| OM2-1 | 1x4 | 4 | V100 |
| OM2-1 | 2x2 | 4 | V100 |
| OM2-025 | serial | 1 | H200 |
| OM2-025 | 1x4 | 4 | H200 |
| OM2-025 | 2x2 | 4 | H200 |

**Note:** 1x4 partitions produce NaN at iter 300+ in full runs (see BENCHMARKS.md),
but 20 profiling steps complete safely.

## Technical notes

- **Distributed profiling** uses the `bash -c` wrapper pattern:
  `mpiexec ... bash -c "nsys profile --trace=nvtx,cuda,mpi ..."`.
  This lets nsys profile each MPI rank individually with per-rank output files.
- **Ranged capture** via `--capture-range=cudaProfilerApi` ensures only the benchmark
  time steps are recorded. `CUDA.@profile external=true` in the Julia script triggers
  `cudaProfilerStart()`/`cudaProfilerStop()` around the benchmark loop.
- **`JULIA_CUDA_MEMORY_POOL=none`** is set for MPI runs to avoid CUDA memory pool
  conflicts, but this can worsen GC pressure (more `cuMemFree` calls).
- **`BENCHMARK_STEPS`** env var controls the number of time steps (default 20 when
  `PROFILE=yes`). This overrides `N_MONTHS` for profiling runs.
- **GC and MPI interaction**: A GC pause on one rank blocks that rank's CPU thread,
  causing all other ranks to stall at the next `MPI_Waitall`. This is visible in the
  GUI as correlated GC NVTX ranges and long MPI_Waitall bars across ranks.
