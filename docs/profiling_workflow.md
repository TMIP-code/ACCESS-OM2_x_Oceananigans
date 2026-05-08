# GPU Profiling Workflow

This guide covers setting up and running GPU profiling using **Nsight Systems (nsys)** for single-GPU runs and **Nsight Compute (ncu)** for kernel-level analysis and strong-scaling studies.

## Quick Start

### Nsight Systems (nsys) — Runtime Trace Profiling

Profile any benchmark run by adding `PROFILE=yes`:

```bash
# Serial (1 GPU)
PARENT_MODEL=ACCESS-OM2-1 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Distributed (4 GPUs, 2×2)
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Distributed (4 GPUs, 1×4)
PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x4 PROFILE=yes JOB_CHAIN=partition-run1yrfast bash scripts/driver.sh

# OM2-025 serial (1 hr walltime due to large grid finalization)
PARENT_MODEL=ACCESS-OM2-025 WALLTIME_RUN_1YEAR=01:00:00 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh

# Custom benchmark length (default: 20 steps)
PARENT_MODEL=ACCESS-OM2-025 BENCHMARK_STEPS=50 PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

### Nsight Compute (ncu) — Kernel-Level Roofline Analysis

For strong-scaling and kernel optimization:

```bash
# Template: K=24, proper warmup/steady-state separation
PARENT_MODEL=ACCESS-OM2-025 \
EXPERIMENT=025deg_jra55_iaf_omip2_cycle6 \
TIME_WINDOW=1968-1977 \
GRID_HX=25 GRID_HY=25 GRID_HZ=2 \
PARTITION=1x4 \
TBLOCKING=24 \
BENCHMARK_STEPS=240 \
NCU_SET=roofline \
NCU_KERNEL='gpu_compute_hydrostatic_free_surface_Gc_' \
NCU_SKIP=48 \
NCU_COUNT=48 \
GPU_QUEUE=gpuvolta \
JOB_CHAIN=run1yrncu \
bash scripts/driver.sh
```

## Nsight Systems (nsys) — Runtime Tracing

### How It Works

Two mechanisms capture only the benchmark region:

1. **`CUDA.@profile external=true`** in `run_1year_benchmark.jl` wraps the benchmark loop with `cudaProfilerStart()`/`cudaProfilerStop()`.
2. **`--capture-range=cudaProfilerApi`** in nsys only records data between those API calls.

**Result:** Small profiles (1–4 MB for serial, ~3 MB per rank for distributed) instead of hundreds of MB or GB.

### What Gets Captured

- **Serial runs**: `nsys profile --trace=nvtx,cuda` wraps the Julia process
- **Distributed runs**: All ranks profiled individually via `bash -c` wrapper, each produces `.nsys-rep` with MPI tracing (`--trace=nvtx,cuda,mpi`)
- **GC tracing**: `JULIA_NVTX_CALLBACKS=gc` traces Julia garbage collection as NVTX markers (visible in GUI timeline)
- **Step count**: `BENCHMARK_STEPS=20` (default when `PROFILE=yes`) limits benchmark to 20 steps after 3 warmup steps

**Alternative for Distributed Runs:** For better diagnosis of load imbalance and GPU synchronization issues, use `nsys profile --stats=true` instead. This captures a single unified timeline with all GPUs visible together, making it easy to spot which rank is stalling others. Recommended when investigating MPI communication bottlenecks or load-balance problems in multi-GPU runs.

### Output Files

Profiles saved in:
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

### Viewing Profiles

#### CLI Summary (on Gadi compute node)

`nsys stats` is memory-intensive for login nodes. Run on a compute node:

```bash
# Interactive session
qsub -I -P y99 -l mem=47GB -q express -l walltime=00:30:00 \
     -l ncpus=12 -l storage=gdata/y99+scratch/y99

# Load cuda module
module load cuda/12.9.0

# Full summary (CUDA API + GPU kernels + memory)
nsys stats profile.nsys-rep

# GPU kernel summary only
nsys stats --report=cuda_gpu_kern_sum profile.nsys-rep

# CUDA API summary only
nsys stats --report=cuda_api_sum profile.nsys-rep

# MPI event summary (distributed runs only)
nsys stats --report=mpi_event_sum profile_rank0.nsys-rep
```

#### GUI (on local machine)

1. Download [Nsight Systems](https://developer.nvidia.com/nsight-systems) (free)
2. Copy `.nsys-rep` files from Gadi:
   ```bash
   scp gadi:/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia/.../profile*.nsys-rep .
   ```
3. Open with `nsys-ui profile.nsys-rep`
4. For distributed runs with per-rank files: open multiple rank files as a multi-report view to compare

The GUI timeline shows:
- **Green bars**: GPU kernel execution
- **Orange bars**: CUDA memcpy (GPU-CPU transfers)
- **NVTX ranges**: GC events (from `JULIA_NVTX_CALLBACKS=gc`)
- **Gaps**: Idle time (sync, kernel launch overhead)
- **MPI rows** (distributed): MPI send/recv/waitall calls per rank

#### Unified Timeline for Distributed Profiling (Recommended for Load-Balance Analysis)

For diagnosing load imbalance and synchronization stalls across all GPUs on a single timeline:

```bash
# Modify the PBS script to use --stats=true instead of per-rank capture
# In scripts/standard_runs/run_1year.sh, replace the nsys command with:
nsys profile --stats=true --trace=nvtx,cuda,mpi --capture-range=cudaProfilerApi \
  --capture-range-end=stop mpiexec ... julia ...
```

This produces a **single `.nsys-rep` file with all GPUs on the same timeline**, making it immediately obvious:
- Which rank's GPU is slower/stalling others
- Whether load is evenly distributed
- Where MPI synchronization barriers cause idle time on fast ranks

This approach is **far superior for diagnosing scaling bottlenecks** compared to comparing separate per-rank traces. Use it when you see unexpected wall-time plateaus (e.g., 1×4→1×8 gives only 4% improvement instead of 2×).

### Key Metrics to Look For

| Metric | What it tells you |
|--------|-------------------|
| `compute_hydrostatic_free_surface_Gc!` | Tracer tendency computation time |
| `compute_w_from_continuity!` | w diagnosis time (dominant in distributed runs) |
| `solve_batched_tridiagonal_system` | Free-surface solver time |
| `MPI_Waitall` median | Per-exchange MPI synchronization cost |
| `MPI_Waitall` max | Worst-case stall (often caused by GC on another rank) |
| GC NVTX ranges | Julia GC pauses — propagate as MPI stalls |
| `cuStreamSynchronize` time | CPU waiting for GPU |
| `cuLaunchKernel` count | Number of kernel launches per step |

### Profiling Matrix

| Model | Partition | GPUs | GPU type |
|-------|-----------|------|----------|
| OM2-1 | serial | 1 | V100 |
| OM2-1 | 1×4 | 4 | V100 |
| OM2-1 | 2×2 | 4 | V100 |
| OM2-025 | serial | 1 | H200 |
| OM2-025 | 1×4 | 4 | H200 |
| OM2-025 | 2×2 | 4 | H200 |

**Note:** 1×4 partitions produce NaN at iter 300+ in full runs, but 20 profiling steps complete safely.

---

## Nsight Compute (ncu) — Kernel-Level Roofline Profiling

### Overview

For kernel-level performance analysis and strong-scaling studies. Critical: separate warmup from steady-state profiling.

### Critical Pitfalls to Avoid

#### 1. GPU Consistency Across Comparisons
- **Mistake**: K=12 (V100) vs K=24 (H200) — different GPUs confound results
- **Fix**: Force GPU queue explicitly: `GPU_QUEUE=gpuvolta` (V100) or `GPU_QUEUE=gpuhopper` (H200)
- **Why**: H200 has 2× memory bandwidth; metrics differ even if kernel scaling is identical

#### 2. Warmup Contamination
- **Mistake**: `BENCHMARK_STEPS=24` with `NCU_SKIP=0, NCU_COUNT=12` profiles first 12 launches (GPU JIT + cache cold-start)
- **Fix**: Use `BENCHMARK_STEPS = K × N_BATCHES` where `N_BATCHES ≥ 3` (first 1–2 for warmup, rest for profiling)
  - Example: K=24, 10 batches → `BENCHMARK_STEPS=240`, then `NCU_SKIP=48, NCU_COUNT=48` (skip batches 0–1, profile batches 2–3)
- **Why**: GPU JIT and L1/L2 caches stabilize after 2–3 kernel launches; measuring during warmup biases results

#### 3. Kernel Name Regex Matching
- **Mistake**: `NCU_KERNEL='gpu_compute_hydrostatic'` doesn't match `gpu_compute_hydrostatic_free_surface_Gc_` (exact suffix check)
- **Fix**: Use explicit kernel name or broader regex:
  - Exact: `gpu_compute_hydrostatic_free_surface_Gc_`
  - Regex: `.*Gc_$` (ends with Gc_)
  - Permissive: `.*Gc.*` (contains Gc)
- **Why**: ncu warns "No kernels were profiled" if filter misses target; always grep job log for available kernels

#### 4. Batch Structure and MPI Passes
- **Formula**: `TBLOCKING=K` requires `BENCHMARK_STEPS % K == 0` (divisibility constraint)
- **MPI passes**: `BENCHMARK_STEPS = K × N_BATCHES` → exactly N_BATCHES kernel launches, N_BATCHES – 1 inter-batch MPI halo exchanges
  - Example: K=12, BENCHMARK_STEPS=24 → 2 batches, 1 MPI pass
  - Example: K=24, BENCHMARK_STEPS=240 → 10 batches, 9 MPI passes (first 2 for warmup)
- **Why**: MPI happens between batches only; knowing batch count essential for interpreting communication overhead

### Checklist Before Submission

- [ ] GPU queue matches comparison target (both V100 or both H200, not mixed)
- [ ] `BENCHMARK_STEPS = TBLOCKING × N_BATCHES` where `N_BATCHES ≥ 3`
- [ ] `NCU_SKIP ≥ TBLOCKING × 2` (skip ≥2 batches for warmup)
- [ ] `NCU_KERNEL` matches actual kernel name (grep "gpu_" in first job log if unsure)
- [ ] Partition files exist with correct halos: `GRID_HX=TBLOCKING+1, GRID_HY=TBLOCKING+1`
- [ ] Same GPU queue for all configs being compared

### Template Job Submission

```bash
PARENT_MODEL=ACCESS-OM2-025 \
EXPERIMENT=025deg_jra55_iaf_omip2_cycle6 \
TIME_WINDOW=1968-1977 \
GRID_HX=<TBLOCKING+1> GRID_HY=<TBLOCKING+1> GRID_HZ=2 \
PARTITION=1x4 \
TBLOCKING=<K> \
BENCHMARK_STEPS=<K * N_BATCHES> \
NCU_SET=roofline \
NCU_KERNEL='gpu_compute_hydrostatic_free_surface_Gc_' \
NCU_SKIP=<TBLOCKING * 2> \
NCU_COUNT=<TBLOCKING * 2> \
GPU_QUEUE=gpuvolta \
JOB_CHAIN=run1yrncu \
bash scripts/driver.sh
```

### Extracting Metrics

```bash
module load cuda/12.9.0
ncu --import <path>.ncu-rep --print-summary per-kernel 2>&1 | head -100
```

Look for:
- **Kernel Duration (ms)** — per-kernel time; scale linearly with partition size if GPU-bound
- **Memory Throughput (%)** — target ≥ 40% for memory-bound kernels; compare across configs
- **Compute Throughput (%)** — should scale linearly with partition size
- **SM Frequency (GHz)**, **DRAM Frequency (MHz)** — check for thermal throttling or frequency shifts

---

## Gadi-Specific Considerations

### Partition Build Prerequisites

Distributed runs require pre-partitioned FTS files. Before submitting any distributed profiling jobs:

```bash
# Standard partition (e.g., 1x4)
PARENT_MODEL=ACCESS-OM2-1 GRID_HX=13 GRID_HY=13 PARTITION=1x4 JOB_CHAIN=partition bash scripts/driver.sh

# Load-balanced partition (use LOAD_BALANCE=cell, NOT PARTITION=1x4_LB)
PARENT_MODEL=ACCESS-OM2-1 GRID_HX=13 GRID_HY=13 PARTITION=1x4 LOAD_BALANCE=cell JOB_CHAIN=partition bash scripts/driver.sh
```

The `_LB` suffix is added automatically to the output directory; passing `PARTITION=1x4_LB` directly fails because bash arithmetic can't parse the suffix.

**Memory defaults:** Each partition build rank loads the full serial FTS into memory, so peak memory scales linearly with NRANKS. Model configs set `PARTITION_MEM_PER_RANK` (added 2026-05-08):

| Model | Per-rank | Queue | 1x2 | 1x4 | 1x8 |
|-------|----------|-------|-----|-----|-----|
| OM2-1 | 12 GB | express | 24GB/6cpu | 48GB/12cpu | 96GB/24cpu |
| OM2-025 | — (uses hugemem 192GB minimum, peak 220GB at 1x8) | hugemem | 192GB | 192GB | 256GB |
| OM2-01 | 350 GB | megamem | 700GB | 1.4TB | 2.8TB |

If a partition build dies with exit 137 (SIGKILL) and only some FTS rank files are written (e.g., u/v but no w/eta), it was OOM-killed. Bump `PARTITION_MEM_PER_RANK` in the model config.

### Pipeline Step Resource Requirements

Memory peaks observed across all historical runs in `logs/PBS/` (sized for safety, not just observed peak):

| Model | grid | vel | prep | partition |
|-------|------|-----|------|-----------|
| OM2-1 | express 47GB ✓ (peak 19GB) | express 47GB ✓ (peak 20GB) | normal 96GB ✓ (peak 61GB) | per-rank 12GB |
| OM2-025 | express 47GB ✓ (peak 3GB) | express **96GB** (peak 47GB cap-hit at default; bumped 2026-05-08) | hugemem 192GB ✓ (peak 164GB) | hugemem 192-256GB |
| OM2-01 | express 47GB ✓ (peak 12GB) | hugemem 512GB ✓ (peak 339GB) | megamem 2TB ✓ (peak 944GB) | megamem per-rank 350GB |

Always run `prep` (Python preprocessing) before `vel` for a new TIME_WINDOW — it creates the monthly NetCDFs that `vel` reads. Use `JOB_CHAIN=prep-grid-vel-partition` (or the shortcut `preprocessing-partition`) for a fresh time window.

### Temporary Directory for nsys Finalization

On Gadi, nsys finalization can fail with file descriptor exhaustion. Set:

```bash
export TMPDIR=$MYSCRATCH/tmp
```

Before submitting profiling jobs. This redirects nsys temporary files to your scratch directory instead of `/tmp`.

### MPI-Aware Profiling

For MPI+GPU profiling, use:

```bash
mpiexec --bind-to socket --map-by socket -n $NGPUS nsys profile --trace=nvtx,cuda,mpi ...
```

The `--bind-to socket --map-by socket` flags ensure each MPI rank runs on the CPU socket directly connected to its GPU, eliminating cross-socket communication overhead.

### GPU-Aware MPI

Verify CUDA-aware MPI is working by checking for `cuMemcpyDtoDAsync` calls (GPU→GPU IPC transfers) in the CUDA API summary. If absent, communication routes through CPU, degrading performance.

---

## Workflow Summary

### For Single-GPU or Quick Profiling

1. Add `PROFILE=yes` to your command
2. Run normally via `scripts/driver.sh`
3. Download `.nsys-rep` file
4. Open in GUI or run `nsys stats` on a compute node

### For Strong-Scaling or Kernel Optimization

1. Design job with correct TBLOCKING, BENCHMARK_STEPS, NCU_SKIP, NCU_COUNT
2. Verify GPU queue consistency across all runs
3. Submit via `run1yrncu` job chain
4. Extract roofline metrics with `ncu --import`
5. Compare per-kernel duration, memory throughput, compute throughput
6. Analyze scaling efficiency across partition sizes

### For Diagnosing Distributed Scaling Plateaus

When you see unexpected wall-time plateaus (e.g., 1×4→1×8 yields only 4% improvement instead of 2×):

1. Use `nsys profile --stats=true` to capture a **unified timeline with all GPUs visible together**
2. Download the single `.nsys-rep` file and open in GUI
3. Look for:
   - Which GPU(s) finish work early and idle waiting for `MPI_Waitall`
   - Whether load is evenly distributed across ranks
   - GC pauses on slow ranks blocking all others
   - Synchronization barriers that stall fast GPUs

This approach is **far more effective than comparing per-rank traces separately** for identifying communication bottlenecks and load-balance issues.

### Interpreting Results

- **Memory throughput 35–45%**: Memory-bound workload; scaling limited by memory bandwidth
- **Compute throughput ~22%**: Normal for memory-bound kernels; not a bottleneck
- **Kernel duration scales linearly with cell count**: GPU-bound, good strong-scaling
- **Wall-time improvement << kernel speedup**: MPI overhead or load imbalance dominates
- **Frequency throttling or thermal issues**: Check SM/DRAM frequency trends

See [profiling_results.md](profiling_results.md) for historical data and interpretation examples.
