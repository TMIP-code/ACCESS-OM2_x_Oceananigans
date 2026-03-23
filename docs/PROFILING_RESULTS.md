# Profiling Results

## Benchmark wall times

| Date | Model | Partition | W form. | GPU | Benchmark (s) | Speedup | Job ID | Note |
|------|-------|-----------|---------|-----|---------------|---------|--------|------|
| 2026-03-20 | OM2-1 | serial | diagnosed | 1× V100 | 41 | 1.0× | `163704544` | |
| 2026-03-20 | OM2-1 | 2x2 | diagnosed | 4× V100 | 41 | 1.0× | `163704958` | no scaling |
| 2026-03-20 | OM2-025 | serial | diagnosed | 1× H200 | 510 | 1.0× | `163704551` | |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× H200 | 468 | 1.09× | `163706476` | |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× V100 | 546 | 0.93× | `163707102` | slower than serial H200 |
| 2026-03-24 | OM2-025 | 2x2 | prescribed | 4× H200 | 432 | 1.18× | `163759028` | GC fix + prescribed w |

## Nsight Systems profile summaries

### OM2-1 serial diagnosed-w (1× V100) — Job `163718543`

**CUDA API Summary (top 5 by CPU time):**

| API call | Total (s) | Count | Avg (ms) | % of total |
|----------|-----------|-------|----------|------------|
| cuStreamSynchronize | 20.8 | 112K | 0.19 | 68.8% |
| cuLaunchKernel | 5.9 | 105K | 0.06 | 19.4% |
| cuMemcpyDtoHAsync | 1.1 | 12K | 0.09 | 3.5% |
| cuMemcpyHtoDAsync | 0.82 | 12K | 0.07 | 2.7% |
| cuMemAllocAsync | 0.67 | 12K | 0.06 | 2.2% |

**GPU Kernel Summary (top 5 by GPU time):**

| Kernel | Total (s) | Count | Avg (ms) | % of GPU |
|--------|-----------|-------|----------|----------|
| compute_hydrostatic_free_surface_Gc! | 2.9 | 6K | 0.49 | 38.8% |
| _compute_w_from_continuity! | 1.3 | 6K | 0.23 | 17.8% |
| _compute_w_from_continuity! (halo west) | 0.73 | 6K | 0.12 | 9.8% |
| _compute_w_from_continuity! (halo east) | 0.43 | 6K | 0.07 | 5.7% |
| _update_zstar_scaling! | 0.26 | 6K | 0.04 | 3.5% |

**Key insight:** w computation (interior + halos) = **33.3%** of GPU time. With prescribed w, this drops to ~0%.

### OM2-1 2x2 diagnosed-w (4× V100) — Job `163737132`

**CUDA API Summary (rank 0):**

| API call | Total (s) | Count | Avg (ms) | % of total |
|----------|-----------|-------|----------|------------|
| cuCtxGetId | 10.8 | 2.8M | 0.004 | 33.7% |
| cuLaunchKernel | 9.2 | 129K | 0.07 | 28.9% |
| cuStreamSynchronize | 5.3 | 23K | 0.23 | 16.7% |
| cuMemAllocAsync | 2.2 | 42K | 0.05 | 6.8% |
| cuMemFreeAsync | 1.5 | 42K | 0.04 | 4.7% |

**GPU Kernel Summary (rank 0, top 5):**

| Kernel | Total (s) | Count | Avg (ms) | % of GPU |
|--------|-----------|-------|----------|----------|
| compute_hydrostatic_free_surface_Gc! | 2.3 | 6K | 0.38 | 41.1% |
| _compute_w_from_continuity! (halo west) | 0.71 | 12K | 0.06 | 12.7% |
| _compute_w_from_continuity! (halo east) | 0.66 | 12K | 0.06 | 11.9% |
| _compute_w_from_continuity! | 0.64 | 6K | 0.11 | 11.5% |
| _compute_w_from_continuity! (halo north) | 0.51 | 6K | 0.09 | 9.1% |

**Key insight:** w computation total = **45.2%** of GPU time (worse than serial due to halo variants). `cuCtxGetId` at 33.7% and `cuMemAllocAsync`/`cuMemFreeAsync` at 11.5% suggest **GC-related overhead** from MPI allocations.

### Profiles pending (submitted 2026-03-24)

| Job ID | Model | Partition | W form. | GPU | Note |
|--------|-------|-----------|---------|-----|------|
| `163780377` | OM2-1 | serial | diagnosed | 1× V100 | Updated Oceananigans (GC fix) |
| `163780380` | OM2-1 | 2x2 | diagnosed | 4× V100 | Updated Oceananigans (GC fix) |
| `163780321` | OM2-025 | 2x2 | prescribed | 4× H200 | GC fix + prescribed w |
| `163780332` | OM2-025 | 2x2 | diagnosed | 4× H200 | Updated Oceananigans (GC fix) |
