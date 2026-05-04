# NCU Profiling Methodology

## Overview
Nsight Compute (ncu) kernel-level profiling for ACCESS-OM2 temporal-blocking simulations. Critical: separate warmup from steady-state profiling.

## Key Pitfalls to Avoid

### 1. **GPU Consistency Across Comparisons**
- **Mistake**: K=12 (V100) vs K=24 (H200) — different GPUs, confounded results
- **Fix**: Force GPU queue explicitly: `GPU_QUEUE=gpuvolta` (V100) or `GPU_QUEUE=gpuhopper` (H200)
- **Why**: H200 has 2× memory bandwidth; metrics will differ even if kernel scaling is identical

### 2. **Warmup Contamination**
- **Mistake**: `BENCHMARK_STEPS=24` with `NCU_SKIP=0, NCU_COUNT=12` profiles first 12 launches, likely JIT + cache cold-start
- **Fix**: Use `BENCHMARK_STEPS = K * N_BATCHES` where `N_BATCHES ≥ 3` (first 1–2 for warmup, rest for profiling)
  - Example: K=24, 10 batches → `BENCHMARK_STEPS=240`, then `NCU_SKIP=48, NCU_COUNT=48` (skip batches 0–1, profile batches 2–3)
- **Why**: GPU JIT and L1/L2 caches stabilize after a few kernel launches; measuring during warmup biases results

### 3. **Kernel Name Regex**
- **Mistake**: `NCU_KERNEL='gpu_compute_hydrostatic'` doesn't match `gpu_compute_hydrostatic_free_surface_Gc_` (exact suffix check)
- **Fix**: Use explicit kernel name or broader regex:
  - Exact: `gpu_compute_hydrostatic_free_surface_Gc_`
  - Regex: `.*Gc_$` (ends with Gc_)
  - Permissive: `.*Gc.*` (contains Gc)
- **Why**: ncu warns "No kernels were profiled" if filter misses target; check available kernels in job log

### 4. **Batch Structure and MPI Passes**
- **Formula**: `TBLOCKING=K` requires `BENCHMARK_STEPS % K == 0` (divisibility)
- **MPI passes**: With K=K_val and BENCHMARK_STEPS=K_val × N_BATCHES, you get exactly N_BATCHES kernel launches and N_BATCHES – 1 inter-batch MPI halo exchanges
  - Example: K=12, BENCHMARK_STEPS=24 → 2 batches, 1 MPI pass
  - Example: K=24, BENCHMARK_STEPS=240 → 10 batches, 9 MPI passes (warmup in early batches)

## Template Job Submission

```bash
# Full command for NCU profiling with proper warmup/steady-state separation
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

**Breakdown:**
- `TBLOCKING=24`: 24 substeps per batch
- `BENCHMARK_STEPS=240`: 10 batches total (240 ÷ 24)
- `NCU_SKIP=48`: Skip first 2 batches (0–47 launches) for warmup
- `NCU_COUNT=48`: Profile next 2 batches (launches 48–95) for steady-state
- `GPU_QUEUE=gpuvolta`: Force V100 (not H200 or other GPU)
- `NCU_KERNEL='gpu_compute_hydrostatic_free_surface_Gc_'`: Exact kernel name (check job log for available kernels)

## Extracting Metrics

```bash
module load cuda/12.9.0
ncu --import <path>.ncu-rep --print-summary per-kernel 2>&1 | head -100
```

Extract roofline metrics (GPU Speed Of Light Throughput):
- Memory Throughput (%) — target ≥ 40% for memory-bound kernels
- Compute Throughput (%) — compare across configs
- Duration (ms) — per-kernel time; scale linearly with partition size if GPU-bound

## Checklist Before Submission

- [ ] GPU queue matches comparison target (V100 for volta, H200 for hopper)
- [ ] `BENCHMARK_STEPS = TBLOCKING × N_BATCHES` where `N_BATCHES ≥ 3`
- [ ] `NCU_SKIP ≥ TBLOCKING × 2` (skip ≥2 batches for warmup)
- [ ] `NCU_KERNEL` matches actual kernel name (grep "gpu_" in job log after first failed submission)
- [ ] Partition files exist with correct halos: `GRID_HX=TBLOCKING+1, GRID_HY=TBLOCKING+1`
- [ ] Same GPU queue for all configs being compared

## References
- [PROFILING_RESULTS.md](PROFILING_RESULTS.md) — historical profiling data
- [run_1year_ncu.sh](../scripts/standard_runs/run_1year_ncu.sh) — NCU wrapper script
