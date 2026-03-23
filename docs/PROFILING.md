# GPU Profiling with NVIDIA Nsight Systems

## Quick start

Profile any benchmark run by adding `PROFILE=yes`:

```bash
# Serial (1 GPU)
PARENT_MODEL=ACCESS-OM2-1 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Distributed (4 GPUs)
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Prescribed w
PARENT_MODEL=ACCESS-OM2-025 W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=diagnosed \
    PARTITION=2x2 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

## What gets profiled

- **Serial runs**: `nsys profile` wraps the Julia process directly.
- **Distributed runs**: Only rank 0 is profiled (via a wrapper script). Other ranks run plain Julia. This avoids SIGTERM issues with `nsys` + `mpiexec` on PBS.
- **GC tracing**: `JULIA_NVTX_CALLBACKS=gc` is set automatically to trace Julia garbage collection events.
- **CUDA tracing**: `--trace=nvtx,cuda` captures GPU kernels, memory operations, and NVTX annotations.

## Output files

Profiles are saved in the log directory:
```
logs/julia/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/standardrun/
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile.nsys-rep           # serial
    {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_rank0.nsys-rep     # distributed
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

## Profiling matrix

For a complete performance picture, profile these configurations:

| Model | Partition | W formulation | GPU |
|-------|-----------|---------------|-----|
| OM2-1 | serial | diagnosed | V100 |
| OM2-1 | 2x2 | diagnosed | V100 |
| OM2-1 | serial | prescribed | V100 |
| OM2-1 | 2x2 | prescribed | V100 |
| OM2-025 | serial | diagnosed | H200 |
| OM2-025 | 2x2 | diagnosed | H200 |
| OM2-025 | serial | prescribed | H200 |
| OM2-025 | 2x2 | prescribed | H200 |

## Technical notes

- Profiles are written to `$PBS_JOBFS` (local SSD) during the run, then copied to persistent storage. This avoids network FS distortion.
- `--trace=mpi` does NOT work reliably with OpenMPI's `mpiexec` on PBS. MPI overhead must be inferred from CUDA memcpy patterns and idle gaps.
- `JULIA_CUDA_MEMORY_POOL=none` is set for MPI runs to avoid CUDA memory pool conflicts.
- `jobfs=40GB` is requested for profiled runs (profiles can be 5-10 GB).
