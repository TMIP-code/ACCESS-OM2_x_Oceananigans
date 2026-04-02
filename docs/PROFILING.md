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

# Custom step count (default: 20 when PROFILE=yes)
PARENT_MODEL=ACCESS-OM2-025 BENCHMARK_STEPS=50 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

## What gets profiled

- **Serial runs**: `nsys profile` wraps the Julia process directly (`--trace=nvtx,cuda`).
- **Distributed runs**: All ranks are profiled via `bash -c` wrapper, each producing its own `.nsys-rep` file with MPI tracing (`--trace=nvtx,cuda,mpi`).
- **GC tracing**: `JULIA_NVTX_CALLBACKS=gc` is set automatically to trace Julia garbage collection events.
- **Step count**: `BENCHMARK_STEPS=20` (default when `PROFILE=yes`) limits the simulation to ~20 time steps after warmup, keeping profiles small.

## Output files

Profiles are saved in the log directory:
```
logs/julia/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/standardrun/
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile.nsys-rep           # serial
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_rank0.nsys-rep     # distributed, rank 0
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_rank1.nsys-rep     # distributed, rank 1
    ...
```

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

# Top-10 kernels
nsys stats --report=cuda_gpu_kern_sum profile.nsys-rep | head -20
```

### GUI (on local machine)

1. Download [Nsight Systems](https://developer.nvidia.com/nsight-systems) (free)
2. Copy the `.nsys-rep` file from Gadi:
   ```bash
   scp gadi:Projects/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia/.../profile.nsys-rep .
   ```
3. Open with `nsys-ui profile.nsys-rep`

The GUI timeline shows:
- **Green bars**: GPU kernel execution
- **Orange bars**: CUDA memcpy (GPU-CPU transfers)
- **NVTX ranges**: GC events, user annotations
- **Gaps**: Idle time (sync, kernel launch overhead)
- **MPI rows** (distributed): MPI send/recv/barrier calls per rank

## Key metrics to look for

| Metric | What it tells you |
|--------|-------------------|
| `cuStreamSynchronize` time | CPU waiting for GPU (overhead) |
| `cuLaunchKernel` count | Number of kernel launches per step |
| `compute_hydrostatic_free_surface_Gc!` | Tracer tendency computation time |
| `compute_w_from_continuity!` | w diagnosis time (should be 0 with prescribed w) |
| `_update_prescribed_∂t_σ!` | z-star update time |
| GC NVTX ranges | Julia garbage collection pauses |
| `cuMemcpy*` time | GPU-CPU data transfer overhead |
| MPI send/recv time | Communication overhead between ranks (distributed) |
| MPI barrier time | Synchronization overhead between ranks (distributed) |

## Profiling matrix

For a complete performance picture, profile these configurations:

| Model | Partition | GPUs | GPU type |
|-------|-----------|------|----------|
| OM2-1 | serial | 1 | V100 |
| OM2-1 | 1x4 | 4 | V100 |
| OM2-1 | 2x2 | 4 | V100 |
| OM2-025 | serial | 1 | H200 |
| OM2-025 | 1x4 | 4 | H200 |
| OM2-025 | 2x2 | 4 | H200 |

**Note:** 1x4 partitions produce NaN at iter 300+ in full runs (see BENCHMARKS.md), but ~20 profiling steps complete safely.

## Technical notes

- MPI tracing works via the `bash -c` wrapper pattern: `mpiexec ... bash -c "nsys profile --trace=nvtx,cuda,mpi ..."`. This lets nsys profile each MPI rank individually with per-rank output files.
- `JULIA_CUDA_MEMORY_POOL=none` is set for MPI runs to avoid CUDA memory pool conflicts.
- `BENCHMARK_STEPS` env var controls the number of time steps (default 20 when `PROFILE=yes`). This overrides `N_MONTHS` for profiling runs.
