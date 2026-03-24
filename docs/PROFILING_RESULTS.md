# Profiling Results

## Benchmark wall times

| Date | Model | Partition | W form. | GPU | Benchmark (s) | Speedup | Job ID | Note |
|------|-------|-----------|---------|-----|---------------|---------|--------|------|
| 2026-03-20 | OM2-1 | serial | diagnosed | 1× V100 | 41 | 1.0× | `163704544` | pre-GC fix |
| 2026-03-20 | OM2-1 | 2x2 | diagnosed | 4× V100 | 41 | 1.0× | `163704958` | pre-GC fix, no scaling |
| 2026-03-20 | OM2-025 | serial | diagnosed | 1× H200 | 510 | 1.0× | `163704551` | pre-GC fix |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× H200 | 468 | 1.09× | `163706476` | pre-GC fix |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× V100 | 546 | 0.93× | `163707102` | pre-GC fix, slower than serial H200 |
| 2026-03-24 | OM2-1 | serial | diagnosed | 1× V100 | 41 | 1.0× | `163780377` | GC fix (Oceananigans update) |
| 2026-03-24 | OM2-1 | 2x2 | diagnosed | 4× V100 | 39 | 1.05× | `163780380` | GC fix |
| 2026-03-24 | OM2-025 | 2x2 | prescribed | 4× H200 | 432 | 1.18× | `163759028` | GC fix + prescribed w |

## Nsight Systems profile summaries

### OM2-1 serial diagnosed-w (1× V100, pre-GC fix) — Job `163718543`

**CUDA API Summary:**

| API call | Total (s) | Count | % |
|----------|-----------|-------|---|
| cuStreamSynchronize | 20.8 | 112K | 68.8% |
| cuLaunchKernel | 5.9 | 105K | 19.4% |
| cuMemcpyDtoHAsync | 1.1 | 12K | 3.5% |
| cuMemcpyHtoDAsync | 0.82 | 12K | 2.7% |
| cuMemAllocAsync | 0.67 | 12K | 2.2% |

**GPU Kernel Summary:**

| Kernel | Total (s) | Count | % |
|--------|-----------|-------|---|
| compute_hydrostatic_free_surface_Gc! | 2.9 | 6K | 38.8% |
| _compute_w_from_continuity! (interior) | 1.3 | 6K | 17.8% |
| _compute_w_from_continuity! (halo west) | 0.73 | 6K | 9.8% |
| _compute_w_from_continuity! (halo east) | 0.43 | 6K | 5.7% |
| _update_zstar_scaling! | 0.26 | 6K | 3.5% |
| **w total** | **2.5** | | **33.3%** |

---

### OM2-1 serial diagnosed-w (1× V100, GC fix) — Job `163780377`

**CUDA API Summary:**

| API call | Total (s) | Count | % | Change vs pre-GC |
|----------|-----------|-------|---|-------------------|
| cuStreamSynchronize | 22.9 | 5K | 67.1% | count 112K→5K ↓↓ |
| cuLaunchKernel | 5.0 | 106K | 14.7% | |
| cuMemFree_v2 | 3.9 | 233 | 11.4% | new (GC frees) |
| cuMemcpyDtoHAsync | 1.1 | 2.5K | 3.1% | count 12K→2.5K ↓ |
| cuMemcpyHtoDAsync | 0.33 | 74 | 1.0% | count 12K→74 ↓↓↓ |

**GPU Kernel Summary:**

| Kernel | Total (s) | Count | % |
|--------|-----------|-------|---|
| compute_hydrostatic_free_surface_Gc! | 16.2 | 5.8K | 38.8% |
| _compute_w_from_continuity! | 14.8 | 5.9K | 35.4% |
| solve_batched_tridiagonal_system | 4.0 | 5.8K | 9.6% |
| _ab2_step_tracer_field! | 1.6 | 5.8K | 3.8% |
| _update_hydrostatic_pressure! | 1.6 | 5.9K | 3.8% |
| **w total** | **14.8** | | **35.4%** |

**Note:** w is now a single kernel (interior+halos combined) vs separate interior/halo kernels before.
GC-related `cuMemAllocAsync`/`cuMemFreeAsync` (11.5% before) eliminated; replaced by 233 `cuMemFree_v2` calls (still 11.4% due to GC pauses).

---

### OM2-1 2x2 diagnosed-w (4× V100, pre-GC fix) — Job `163737132`

**CUDA API Summary (rank 0):**

| API call | Total (s) | Count | % |
|----------|-----------|-------|---|
| cuCtxGetId | 10.8 | 2.8M | 33.7% |
| cuLaunchKernel | 9.2 | 129K | 28.9% |
| cuStreamSynchronize | 5.3 | 23K | 16.7% |
| cuMemAllocAsync | 2.2 | 42K | 6.8% |
| cuMemFreeAsync | 1.5 | 42K | 4.7% |

**GPU Kernel Summary (rank 0):**

| Kernel | Total (s) | Count | % |
|--------|-----------|-------|---|
| compute_hydrostatic_free_surface_Gc! | 2.3 | 6K | 41.1% |
| _compute_w_from_continuity! (halo west) | 0.71 | 12K | 12.7% |
| _compute_w_from_continuity! (halo east) | 0.66 | 12K | 11.9% |
| _compute_w_from_continuity! (interior) | 0.64 | 6K | 11.5% |
| _compute_w_from_continuity! (halo north) | 0.51 | 6K | 9.1% |
| **w total** | **2.52** | | **45.2%** |

**Note:** `cuMemAllocAsync`/`cuMemFreeAsync` at 42K calls each (11.5% combined) — MPI halo exchange allocates temporary GPU buffers. This is the GC overhead the upstream fix targets.

---

### OM2-1 2x2 diagnosed-w (4× V100, GC fix) — Job `163780380`

**CUDA API Summary (rank 0):**

| API call | Total (s) | Count | % | Change vs pre-GC |
|----------|-----------|-------|---|-------------------|
| cuCtxGetId | 1.7 | 25.6M | 30.3% | |
| cuLaunchKernel | 1.6 | 234K | 28.9% | |
| cuStreamGetCaptureInfo | 1.0 | 11.8M | 18.1% | new |
| cuStreamSynchronize | 0.5 | 18K | 8.9% | 5.3→0.5s ↓↓ |
| cuEventQuery | 0.2 | 308K | 3.5% | new |
| cuMemcpyDtoDAsync | 0.15 | 29K | 2.6% | new (GPU-GPU IPC) |
| cuMemcpyDtoHAsync | 0.15 | 447 | 2.6% | 23K→447 ↓↓ |
| cuMemcpyHtoDAsync | 0.12 | 578 | 2.1% | |
| cuMemAllocAsync | 0 | 0 | 0% | 42K→0 ↓↓↓ GONE |
| cuMemFreeAsync | 0 | 0 | 0% | 42K→0 ↓↓↓ GONE |
| cuMemAlloc_v2 | 0.03 | 698 | 0.6% | |
| cuMemFree_v2 | 0.02 | 698 | 0.3% | |

**GPU Kernel Summary (rank 0):**

| Kernel | Total (s) | Count | % |
|--------|-----------|-------|---|
| _compute_w_from_continuity! (interior) | 4.9 | 5.9K | 18.9% |
| compute_hydrostatic_free_surface_Gc! (main) | 4.5 | 5.8K | 17.7% |
| _compute_w_from_continuity! (halo west) | 4.1 | 11.7K | 16.0% |
| _compute_w_from_continuity! (halo east) | 4.0 | 11.7K | 15.7% |
| _compute_w_from_continuity! (halo north) | 3.6 | 11.7K | 14.1% |
| solve_batched_tridiagonal_system | 1.3 | 5.8K | 4.9% |
| **w total** | **16.6** | | **64.7%** |

**Key changes from GC fix:**
- `cuMemAllocAsync`/`cuMemFreeAsync` completely eliminated (42K calls → 0)
- `cuStreamSynchronize` dropped 5.3→0.5s (10× less sync overhead)
- `cuMemcpyDtoHAsync` dropped 23K→447 calls (GPU→CPU staging for MPI)
- New `cuMemcpyDtoDAsync` (29K calls) — GPU-GPU IPC transfers instead of staging through CPU
- w computation now 64.7% of GPU time (the dominant bottleneck for scaling)

---

### OM2-025 serial diagnosed-w (1× H200, pre-GC fix) — Job `163729267`

**CUDA API Summary:**

| API call | Total (s) | Count | % |
|----------|-----------|-------|---|
| cuStreamSynchronize | 291.9 | 5.3K | 58.7% |
| cuLaunchKernel | 122.5 | 316K | 24.6% |
| cuMemFree_v2 | 70.0 | 467 | 14.1% |
| cuMemcpyDtoHAsync | 7.6 | 2.6K | 1.5% |
| cuMemcpyHtoDAsync | 1.9 | 74 | 0.4% |

**GPU Kernel Summary:**

| Kernel | Total (s) | Count | % |
|--------|-----------|-------|---|
| compute_hydrostatic_free_surface_Gc! | 220.8 | 17.5K | 42.7% |
| _compute_w_from_continuity! | 192.8 | 17.5K | 37.3% |
| solve_batched_tridiagonal_system | 43.5 | 17.5K | 8.4% |
| _ab2_step_tracer_field! | 18.5 | 17.5K | 3.6% |
| _update_hydrostatic_pressure! | 10.1 | 17.5K | 2.0% |
| **w total** | **192.8** | | **37.3%** |

**Note:** `cuMemFree_v2` at **70s** (14.1%) — GC overhead scales super-linearly with grid size (17.9× for 14.4× more cells vs OM2-1).

---

### OM2-025 2x2 diagnosed-w (4× H200, pre-GC fix) — Job `163718549` (partial, ~4 steps)

Early attempt produced a 21MB profile with broken filename (`%q{}` not expanded).
Only captured initialization + ~4 time steps before crash, but proportions are revealing.
Later attempts (`163780321`, `163780332`) hit 30-min walltime before profile copy-back.

**CUDA API Summary (rank 0, ~4 steps only):**

| API call | Total (s) | Count | % |
|----------|-----------|-------|---|
| cuMemcpyDtoHAsync | 7.2 | 2,392 | 77.2% |
| cuMemcpyHtoDAsync | 1.9 | 73 | 20.1% |
| cuMemGetInfo_v2 | 0.07 | 19 | 0.7% |
| cuModuleLoadDataEx | 0.06 | 44 | 0.6% |
| cuStreamSynchronize | 0.05 | 4,784 | 0.5% |
| cuMemAlloc_v2 | 0.05 | 114 | 0.5% |
| cuLaunchKernel | 0.008 | 156 | 0.1% |

**GPU Kernel Summary (rank 0, ~4 steps):**

| Kernel | Total (s) | Instances | % |
|--------|-----------|-----------|---|
| _compute_w_from_continuity! | 0.036 | 4 | 22.5% |
| broadcast_kernel_cartesian | 0.030 | 44 | 18.6% |
| compute_hydrostatic_free_surface_Gc! | 0.012 | 1 | 7.4% |

**Key insight:** MPI data staging (`cuMemcpyDtoHAsync` + `cuMemcpyHtoDAsync`) = **97.3%** of CPU time!
GPU kernel time is only ~162ms while MPI staging overhead is 9.1s.
This confirms **GPU↔CPU data copies for MPI halo exchange dominate at OM2-025 resolution**.

---

### Cross-resolution comparison (serial, pre-GC fix)

| Metric | OM2-1 (V100) | OM2-025 (H200) | Ratio | Grid ratio |
|--------|-------------|----------------|-------|------------|
| Grid cells | 5.4M | 77.8M | — | 14.4× |
| GPU kernel time | 41.7s | 516.7s | 12.4× | |
| w computation | 2.5s | 192.8s | 77.1× | |
| Gc! (tendencies) | 2.9s | 220.8s | 76.1× | |
| `cuMemFree_v2` (GC) | 3.9s | 70.0s | 17.9× | super-linear |
| Benchmark walltime | 41s | 510s | 12.4× | |

---

## Profiles pending (submitted 2026-03-24, 1hr walltime)

| Job ID | Model | Partition | W form. | GPU | Note |
|--------|-------|-----------|---------|-----|------|
| `163785756` | OM2-025 | 2x2 | prescribed | 4× H200 | GC fix + prescribed w |
| `163785757` | OM2-025 | 2x2 | diagnosed | 4× H200 | GC fix only |
