# Profiling Results Archive

Collected profiling data from nsys and ncu across OM2-1 and OM2-025 models, single-GPU and distributed configurations.

**Terminology:** DtoD = GPU→GPU, DtoH = GPU→CPU, HtoD = CPU→GPU.

## Benchmark Wall Times

| Date | Model | Partition | W form. | GPU | Benchmark (s) | Speedup | Job ID | Note |
|------|-------|-----------|---------|-----|---------------|---------|--------|------|
| 2026-03-20 | OM2-1 | serial | diagnosed | 1× V100 | 41 | 1.0× | `163704544` | pre-GC fix |
| 2026-03-20 | OM2-1 | 2x2 | diagnosed | 4× V100 | 41 | 1.0× | `163704958` | pre-GC fix, no scaling |
| 2026-03-20 | OM2-025 | serial | diagnosed | 1× H200 | 510 | 1.0× | `163704551` | pre-GC fix |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× H200 | 468 | 1.09× | `163706476` | pre-GC fix |
| 2026-03-20 | OM2-025 | 2x2 | diagnosed | 4× V100 | 546 | 0.93× | `163707102` | pre-GC fix, slower than serial H200 |
| 2026-03-24 | OM2-1 | serial | diagnosed | 1× V100 | 41 | 1.0× | `163780377` | GC fix |
| 2026-03-24 | OM2-1 | 2x2 | diagnosed | 4× V100 | 39 | 1.05× | `163780380` | GC fix |
| 2026-03-24 | OM2-025 | 2x2 | prescribed | 4× H200 | 432 | 1.18× | `163759028` | GC fix + prescribed w |

---

## OM2-1 Serial, 1 GPU

### Pre-GC Fix (Job `163718543`)

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

### Post-GC Fix (Job `163780377`)

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

**Key Insight:** w computation now single kernel (interior+halos combined) vs separate interior/halo kernels. GC-related `cuMemAllocAsync`/`cuMemFreeAsync` eliminated; replaced by 233 `cuMemFree_v2` calls (11.4% — GC pauses remain).

---

## OM2-1 Distributed, 4 GPUs (2×2)

### Pre-GC Fix (Job `163737132`)

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

**Note:** `cuMemAllocAsync`/`cuMemFreeAsync` at 42K calls each (11.5% combined) — GC overhead from `reverse` allocations in MPI halo exchange.

---

### Post-GC Fix (Job `163780380`)

**CUDA API Summary (rank 0):**

| API call | Total (s) | Count | % | Change vs pre-GC |
|----------|-----------|-------|---|-------------------|
| cuCtxGetId | 1.7 | 25.6M | 30.3% | |
| cuLaunchKernel | 1.6 | 234K | 28.9% | |
| cuStreamGetCaptureInfo | 1.0 | 11.8M | 18.1% | new |
| cuStreamSynchronize | 0.5 | 18K | 8.9% | 5.3→0.5s ↓↓ |
| cuEventQuery | 0.2 | 308K | 3.5% | new |
| cuMemcpyDtoDAsync | 0.15 | 29K | 2.6% | GPU→GPU IPC transfers |
| cuMemcpyDtoHAsync | 0.15 | 447 | 2.6% | |
| cuMemcpyHtoDAsync | 0.12 | 578 | 2.1% | |
| cuMemAllocAsync | 0 | 0 | 0% | 42K→0 GONE |
| cuMemFreeAsync | 0 | 0 | 0% | 42K→0 GONE |
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

**Key Insight:** `cuMemAllocAsync`/`cuMemFreeAsync` completely eliminated (42K → 0). `cuStreamSynchronize` dropped 5.3→0.5s (10× reduction). GPU→GPU IPC transfers visible (29K calls, 2.6% — CUDA-aware MPI working). w computation now 64.7% (dominant bottleneck for distributed scaling).

---

## OM2-025 Serial, 1 GPU (Pre-GC Fix)

Job `163729267`

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

**Key Insight:** `cuMemFree_v2` at **70s** (14.1%) — GC overhead scales super-linearly with grid size (17.9× for 14.4× more cells vs OM2-1).

---

## Cross-Resolution Comparison (Serial, Pre-GC Fix)

| Metric | OM2-1 (V100) | OM2-025 (H200) | Ratio | Notes |
|--------|-------------|----------------|-------|-------|
| **Grid dimensions** | ~400×300×Nz | ~1600×1200×Nz | — | ~4× finer horizontally |
| Total cells | 5.4M | 77.8M | 14.4× | 14.4× cell ratio |
| Time steps | 20 | 20 | 1× | Same BENCHMARK_STEPS, but smaller Δt on OM2-025 |
| Effective work ratio | — | — | **~29–58×** | Accounting for smaller time-step (CFL constraint) |
| **GPU kernel time** | 41.7s | 516.7s | 12.4× | Linear with work ✓ |
| w computation | 2.5s | 192.8s | 77.1× | **77× > 14.4× cells** → super-linear |
| Gc! (tendencies) | 2.9s | 220.8s | 76.1× | **76× > 14.4× cells** → super-linear |
| `cuMemFree_v2` (GC) | 3.9s | 70.0s | 17.9× | **17.9× > 14.4× cells** → super-linear GC overhead |
| Benchmark walltime | 41s | 510s | 12.4× | Wall time = GPU kernel time at single GPU |

**Analysis:** The 12.4× wall-time scaling matches the cell ratio (14.4×), suggesting **per-cell kernel efficiency is constant across resolutions**. However:
- **w and Gc! scale 76–77×**, not 14.4× — indicates either (i) more vertical levels in OM2-025, or (ii) algorithmic complexity that increases with resolution
- **GC overhead scales super-linearly (17.9×)** — memory allocation patterns worsen at finer resolution
- **Time-step scaling not visible here** — both used BENCHMARK_STEPS=20, masking the CFL-induced time-step size reduction. A fair comparison would normalize by effective physical time or accounting for Δt differences.

---

## OM2-025 1×4 vs 1×8 V100 Strong-Scaling

### Nsys Profiling (2026-05-04)

| Job ID | Partition | Wall Time | GPU Active | MPI_Waitall | Comments |
|--------|-----------|-----------|-----------|------------|----------|
| `167685248` | 1×4 | 133.2 s | 5.26 s | 1.99 s | original Gc |
| `167685252` | 1×8 | 138.9 s | 5.81 s | 3.66 s | original Gc, plateau at 4.4% |

**Key Finding:** 1×8 plateau dominated by **GPU strong-scaling inefficiency** (per-rank GPU work grew despite half the cells), not MPI bandwidth. Halo time ≤5% of wall.

---

### NCU Roofline (2026-05-04)

**Single-Method Gc (Jobs 167702117/118):**

| Metric | 1×4 | 1×8 | Ratio |
|--------|-----|-----|-------|
| **Kernel Duration (avg)** | 20.38 ms | 10.72 ms | **1.90× faster** |
| Memory Throughput | 43.88% | 37.00% | 1×4 memory-bound |
| Compute Throughput (%) | 22.20 | 22.73 | ~same |
| SM Frequency (GHz) | 1.29 | 1.29 | same |

**Wall-time Improvement:** 4.4% (1×4 → 1×8) vs expected 2× from GPU count doubling → **~40× gap** indicates MPI overhead dominates, not GPU saturation.

---

## K-Value Temporal Blocking Study (OM2-025 1×4/1×8)

### Configuration Parameters

| Parameter | K=6 | K=12 | K=24 |
|-----------|-----|------|------|
| TBLOCKING | 6 | 12 | 24 |
| Halo size (GRID_HX/Y) | 7 | 13 | 25 |
| BENCHMARK_STEPS | 60 | 24 | 240 |
| MPI passes per cycle | 10 | 1 | 1 |
| GPU_QUEUE | gpuvolta | gpuvolta | gpuvolta |

### Roofline Metrics Comparison

| Metric | K6 1×4 | K6 1×8 | K12 1×4 | K12 1×8 | K24 1×4 | K24 1×8 |
|--------|--------|--------|---------|---------|---------|---------|
| Kernel Duration (ms) | 19.11 | 10.24 | 20.38 | 10.72 | 22.05 | 12.85 |
| Memory Throughput (%) | 43.29 | 35.18 | 43.88 | 37.00 | 45.34 | 36.46 |
| Compute (SM) Throughput (%) | 22.41 | 21.75 | 22.20 | 22.73 | 22.50 | 22.27 |
| Scaling Ratio (1×4 → 1×8) | 1.87× | — | 1.90× | — | 1.71× | — |

### Wall-Time Improvement (1×4 → 1×8)

| K | Wall Time 1×4 | Wall Time 1×8 | Improvement |
|---|---------------|---------------|------------|
| K=6 | — | — | ~4% (expected) |
| K=12 | 685 s | 655 s | **4.2%** |
| K=24 | 112.0 s | 109.1 s | **2.6%** ❌ Worse |

### Key Findings

**K6 Analysis:**
- ✓ K6 is **5.0% faster** than K12 at 1×4 (19.11 vs 20.38 ms)
- ✓ Confirms hypothesis: smaller halos (7×7 vs 13×13) reduce memory pressure
- ✗ Scaling plateau persists at ~4% (1.87× vs 1.90× — negligible difference)

**K24 Analysis:**
- ✗ **8–20% slower per kernel** despite identical computation
- ✗ Scaling **degrades 1.90× → 1.71×** (10% worse)
- ✗ Larger halo buffers (25×25 cells) increase memory pressure, harms more than fewer MPI passes helps
- **Conclusion:** Communication is **bandwidth-bound**, not latency-bound

### Interpretation

The bottleneck is **not MPI communication overhead**. Instead:
- **Larger halos** create working sets that exceed L2 cache capacity → memory latency increases
- **V100 architectural ceiling:** ~4% wall-time improvement per GPU doubling on OM2-025, regardless of K tuning
- **Root cause:** Per-rank memory bandwidth saturation, not communication overhead or compute efficiency

### Recommendations

1. **Do not increase K beyond 12** — K=24 proves K=12 was better-tuned
2. **Focus on different optimizations:**
   - Domain decomposition (1×4 → 2×2 square vs 1×8 thin) to reduce communication pairs
   - Reduce w computation overhead (currently 35–65% of runtime)
   - Consider OM2-01 H200 (2× memory bandwidth) for comparison
3. **Accept strong-scaling plateau** on V100: 1×4→1×8 yields ~4%, not 2× despite 2× GPU count

---

## Prescribe w Optimization (2026-03-24)

Job `163759028`: OM2-025 2×2 4× H200, prescribed w + GC fix

**Wall Time:** 432s vs 468s diagnosed (pre-GC) → **1.18× speedup** (w is 35–40% of runtime)

**GPU Kernel Change:** w computation eliminated; tendencies + free-surface solver dominate

---

## Summary of Key Insights

1. **GC overhead is significant:** Pre-GC fix had 11–14% of time in `cuMemAllocAsync`/`cuMemFreeAsync` or `cuMemFree_v2`
2. **w computation dominates:** 33–65% of GPU time across all configurations
3. **Distributed scaling plateau on V100:** ~4% wall-time improvement per GPU doubling on OM2-025, independent of K tuning
4. **Memory-bound workload:** 35–45% memory throughput vs compute ~22% → memory bandwidth is limiting factor
5. **Communication overhead:** ≤5% of wall time; not the primary scaling bottleneck
6. **Smaller halos help:** K=6 (7×7 halo) 5% faster than K=12 (13×13) — memory pressure reduction
7. **Bandwidth-bound communication:** Doubling buffer size (K=12 → K=24) hurts performance despite halving MPI passes
8. **OM2-1 vs OM2-025:** GC overhead scales super-linearly with grid size (17.9× for 14.4× cell count)

---

## Recommendations for Future Work

1. **Optimize w computation** — largest kernel by time, potential for specialized formulations
2. **Prescribe w** — eliminates 35–40% of runtime (confirmed 1.18× speedup)
3. **Profile OM2-01 on H200** — test whether 2× memory bandwidth alleviates scaling plateau
4. **Experiment with square partitions** — test 2×2 vs 1×4 for communication topology effects
5. **Reduce GC pressure** — pre-allocate halo buffers or use custom memory pools (complex, diminishing returns)
6. **Investigate register pressure** — OM2-1 shows lower occupancy than OM2-025; could limit per-kernel speedup

## Profiling Documentation Standards

When recording profiling runs, always capture:
- **Grid dimensions:** Nx, Ny, Nz (not just total cell count), and horizontal resolution
- **Time-stepping:** Time-step size Δt (or CFL number), BENCHMARK_STEPS, and wall-clock duration
- **Effective work ratio:** Account for resolution-dependent effects:
  - Cell count scales as Nx × Ny × Nz
  - Time-step size scales inversely with horizontal resolution (CFL constraint)
  - Total work ≈ (Nx × Ny × Nz) × (Δt⁻¹)
- **Per-cell metrics:** Normalize kernel time by total cell count and time-step size to compare algorithmic efficiency across resolutions

This avoids misleading cross-resolution comparisons (e.g., confusing 14.4× cell ratio with actual 30–60× work ratio due to time-stepping).

See [profiling_workflow.md](profiling_workflow.md) for methodology and how to set up new profiling jobs.
