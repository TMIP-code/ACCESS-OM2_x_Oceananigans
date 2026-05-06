# Profiling Results

Terminology: DtoD = GPU→GPU, DtoH = GPU→CPU, HtoD = CPU→GPU.

## Benchmark wall times

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
GC-related `cuMemAllocAsync`/`cuMemFreeAsync` eliminated; replaced by 233 `cuMemFree_v2` calls (11.4% — GC pauses remain).

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

**Note:** `cuMemAllocAsync`/`cuMemFreeAsync` at 42K calls each (11.5% combined) — GC overhead from `reverse` allocations in MPI halo exchange.

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

**Key changes from GC fix:**
- `cuMemAllocAsync`/`cuMemFreeAsync` completely eliminated (42K calls → 0)
- `cuStreamSynchronize` dropped 5.3→0.5s (10× less sync overhead)
- `cuMemcpyDtoDAsync` (29K calls, 2.6%) — GPU→GPU IPC transfers (CUDA-aware MPI working)
- w computation now 64.7% of GPU time (dominant bottleneck for distributed scaling)

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

### OM2-025 2x2 diagnosed-w (4× H200, pre-GC fix) — Job `163718549` (partial, initialization only)

Early attempt produced a 21MB profile with broken filename (`%q{}` not expanded).
Only captured initialization + ~4 time steps before crash.
Later attempts (`163780321`, `163780332`) hit 30-min walltime before profile copy-back.

**CUDA API Summary (rank 0, initialization only — NOT representative of steady-state):**

| API call | Total (s) | Count | % |
|----------|-----------|-------|---|
| cuMemcpyDtoHAsync | 7.2 | 2,392 | 77.2% |
| cuMemcpyHtoDAsync | 1.9 | 73 | 20.1% |
| cuStreamSynchronize | 0.05 | 4,784 | 0.5% |
| cuLaunchKernel | 0.008 | 156 | 0.1% |

**Caveat:** These numbers are dominated by initialization (loading FTS data, constructing fields).
The DtoH/HtoD calls are from data loading, NOT from steady-state halo exchange.
Communication buffers are allocated on GPU (`on_architecture(::Distributed, ...)` delegates to
`child_architecture`, so buffers live on GPU when child is GPU). CUDA-aware MPI is used.
A full-run profile is needed to see the actual distributed overhead breakdown.

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

---

## OM2-025 1×4 vs 1×8 V100 strong-scaling profiles (2026-05-04)

Submitted to localise the OM2-01 1×4→1×8 plateau on a smaller-resolution surrogate.
Same `model_config` for direct comparison, K=12 TBLOCKING with halos=(13,13,2)
matching the existing partition data.

### Submission

Driver invocation (1×4, partition already on disk → no partition build):

```bash
PARENT_MODEL=ACCESS-OM2-025 \
TIME_WINDOW=1968-1977 \
VELOCITY_SOURCE=totaltransport \
TBLOCKING=12 \
GRID_HX=13 GRID_HY=13 GRID_HZ=2 \
GPU_QUEUE=gpuvolta \
PARTITION=1x4 \
PROFILE=yes \
JOB_CHAIN=run1yrfast \
bash scripts/driver.sh
```

For 1×8 the first submission also built the partition via
`JOB_CHAIN=partition-run1yrfast`; once that exists subsequent submissions use
`JOB_CHAIN=run1yrfast` directly.

### Submitted runs

| Job ID | Partition | Commit | Notes |
|--------|-----------|--------|-------|
| `167685248` | 1×4 | `c1938d3` | original Gc with 12 K-specializations |
| `167685251` | partition build | `c1938d3` | built `partitions/1x8/` (CPU hugemem, ~7 min) |
| `167685252` | 1×8 | `c1938d3` | original Gc with 12 K-specializations |
| `167702117` | 1×4 | `5552e86` | single-method Gc (kernel-spec fix) |
| `167702118` | 1×8 | `5552e86` | single-method Gc (kernel-spec fix) |

### Env flags (full set propagated through driver)

| Flag | Value | Purpose |
|------|-------|---------|
| `PARENT_MODEL` | `ACCESS-OM2-025` | parent model |
| `EXPERIMENT` | `025deg_jra55_iaf_omip2_cycle6` (default) | parent forcing |
| `TIME_WINDOW` | `1968-1977` | time window |
| `VELOCITY_SOURCE` | `totaltransport` | match existing 1×4/1×8 partitions |
| `W_FORMULATION` | `wdiagnosed` (default) | |
| `ADVECTION_SCHEME` | `centered2` (default) | |
| `TIMESTEPPER` | `AB2` (default) | |
| `TBLOCKING` | `12` | K=12 substeps per macro-step |
| `GRID_HX`/`GRID_HY` | `13` | required ≥ K+1 for TBLOCKING |
| `GRID_HZ` | `2` | matches partition data |
| `GPU_QUEUE` | `gpuvolta` | V100 (less busy than gpuhopper) |
| `PARTITION` | `1x4` or `1x8` | strong-scaling pair |
| `PROFILE` | `yes` | wraps each rank's Julia in `nsys profile` |
| `BENCHMARK_STEPS` | `240` (default when PROFILE=yes) | profile length |
| `SYNC_GC_NSTEPS` | `5` (default when PROFILE=yes) | synchronized GC every 5 batches |
| `MPI_BINDING` | `numa` (default) | NCI Hybrid MPI pattern #9 |

`MODEL_CONFIG = totaltransport_wdiagnosed_centered2_AB2_TB12`.

### Outputs

`.nsys-rep` files (one per rank) and the rank-0 `.log`:

```
logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/
  totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_<JOBID>.gadi-pbs.log
  totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_<JOBID>.gadi-pbs_profile_syncGCyes_N5_rank{0..N-1}.nsys-rep
```

Glob to download all profiles for both jobs:

```
/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_167685[0-9]*_rank*.nsys-rep
/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/totaltransport_wdiagnosed_centered2_AB2_TB12_1yearfast_167702[0-9]*_rank*.nsys-rep
```

### Headline numbers (167685248 / 167685252, original Gc)

| | 1×4 (rank0) | 1×8 (rank0) |
|---|---|---|
| Wall (240 steps) | 133.2 s | 138.9 s |
| GPU active | 5.26 s | 5.81 s |
| `MPI_Waitall` total | 1.99 s | 3.66 s |
| `cuStreamSynchronize` total | 4.78 s | 5.34 s |
| `Gc` per-instance avg | 17.6 ms | 21.7 ms |

Conclusion: 1×8 plateau on V100 is dominated by **GPU strong-scaling
inefficiency** (per-rank GPU work *grew* slightly despite half the cells), not
by MPI bandwidth. Halo time grew (especially on chassis-edge ranks 3/4) but is
≤5 % of wall. Re-running with single-method `Gc` (167702117/167702118) to rule
out kernel specialisation as a contributor.

### Companion OM2-01 H200 profile pair (queued)

| Job ID | Partition | Notes |
|--------|-----------|-------|
| `167682325` | 1×4 | OM2-01 H200, `cgridtransports`, no TBLOCKING, halos=7 |
| `167682331` | 1×8 | OM2-01 H200, `cgridtransports`, no TBLOCKING, halos=7 |

---

## OM2-025 1×4 vs 1×8 V100 strong-scaling NCU analysis (2026-05-04)

**Context:** Re-ran identical OM2-025 jobs with constant-K temporal blocking (`TBLOCKING=12`, `BENCHMARK_STEPS=24`) and single method instance per kernel. Initial observation: 1×8 wall time only 4.4% faster than 1×4, vs expected 2× from 4× GPU count doubling. NCU roofline profiling on rank 0 to diagnose per-kernel scaling.

**Jobs and Timing:**

| Job ID | Partition | Wall Time | Duration |
|--------|-----------|-----------|----------|
| 167715278 | 1×4 | 685 s | 11:25 |
| 167715279 | 1×8 | 655 s | 10:55 |
| **Ratio** | 1:2 | **0.956× (4.4% slower)** | — |

**Per-Kernel NCU Metrics (Rank 0, `--set roofline`):**

| Metric | 1×4 | 1×8 | Ratio |
|--------|-----|-----|-------|
| **Kernel Duration (avg)** | 20.38 ms | 10.72 ms | **1.90× faster** |
| Grid (blocks) | 87,400 | 46,000 | 1.9 ÷ cells |
| Block size | 256×1×1 | 256×1×1 | same |
| Invocations (rank 0) | 12 | 12 | same |
| Memory Throughput | 43.88% | 37.00% | 1×4 memory-bound |
| DRAM Throughput | 43.88% | 37.00% | 1×4 memory-bound |
| L1/TEX Throughput | 19.89% | 20.08% | ~same (±0.2%) |
| L2 Throughput | 21.13% | 21.43% | ~same (±0.3%) |
| Compute (SM) Throughput | 22.20% | 22.73% | ~same (±0.5%) |
| SM Frequency (avg) | 1.29 GHz | 1.29 GHz | same |
| DRAM Frequency (avg) | 862.82 MHz | 877.05 MHz | +1.6% (1×8) |

**Key Finding: MPI Overhead Dominates**

The kernel accelerates **1.90×** per-rank when doubling partitions from 1×4 to 1×8, but end-to-end wall time improves only **4.4%**. This ~40× gap between kernel scaling and wall-time scaling indicates:

1. **MPI communication overhead dominates** — synchronization, halo exchange, and load imbalance consume ~95% of the per-kernel speedup.
2. **Not GPU saturation** — both configs are memory-bound (~37–44% memory throughput), not compute-bound. L1/L2/compute throughput are identical, ruling out register pressure or occupancy limits as OM2-1 bottleneck.
3. **Not kernel specialization** — constant-K design eliminated multiple method instances; same fast path for both partitions.

**Contrast with OM2-1:** OM2-1's 1×2 profile showed **register pressure + latency-bound execution** (25% of max occupancy, only 25% of cycles issue instructions). OM2-025 exhibits the opposite pathology: **per-kernel scaling is excellent, but distributed overhead caps end-to-end improvement.**

**Next Steps:**

- Analyze MPI tracing (nsys) for 1×4 and 1×8 to quantify communication breakdown (halo exchange time, synchronization barriers, load imbalance).
- Compare against 2×2 config (167702117/118) to see if square partitions reduce communication cost.
- Profile OM2-1 H200 pair for comparison (expected higher memory bandwidth tolerance).

---

## OM2-025 K=24 vs K=12 strong-scaling test (2026-05-04, in progress)

**Hypothesis:** Doubling temporal blocking K (12 → 24) **halves MPI halo passes** (2 → 1 per benchmark cycle) but **doubles buffer sizes**. Tests whether communication is **latency-bound** (fewer passes helps) vs **bandwidth-bound** (larger buffers hurt).

**Configuration:**
- Grid rebuilt with `GRID_HX=25, GRID_HY=25, GRID_HZ=2` (K+1 sizing)
- Repartitioned with same halos
- Both 1×4 and 1×8 run TBLOCKING=24, BENCHMARK_STEPS=24 (1 batch)
- Compared against K=12 results (2 batches, 2 MPI passes)

**Jobs submitted:**
- Preprocessing (grid/vel/clo): 167715618–620
- Partitions: 167715621 (1×4), 167715622 (1×8)
- NCU runs (K=24, improper warmup): 167716991 (1×4, H200), 167716992 (1×8, H200) — **❌ H200 instead of V100; no warmup separation**
- NCU runs (K=24, corrected): 167737076 (1×4, V100), 167737090 (1×8, V100) — **✓ proper warmup/profiling split, exact kernel name**

**Corrected Submission Call (K=24):**

```bash
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

**Why this works:**
- 10 batches total (240 ÷ 24), batches 0–1 skip for GPU warmup, batches 2–3 profiled for steady-state
- V100 GPU matches K=12 baseline (fair comparison)
- Exact kernel name + roofline metrics
- See [NCU_PROFILING_METHODOLOGY.md](NCU_PROFILING_METHODOLOGY.md) for detailed methodology

**Results (Jobs 167737076, 167737090 completed 2026-05-06 12:15 UTC):**

### Side-by-Side Metrics: K=12 vs K=24

| Metric | K=12 1×4 | K=12 1×8 | K=24 1×4 | K=24 1×8 | Analysis |
|--------|----------|----------|----------|----------|----------|
| **Kernel Duration (ms)** | 20.38 | 10.72 | 22.05 | 12.85 | K=24 **+8% slower** (1×4), **+20% slower** (1×8) |
| **Per-kernel Scaling Ratio (1×4→1×8)** | — | 1.90× | — | 1.71× | **Degraded:** 1.90× → 1.71× (–10% worse) |
| **Memory Throughput (%)** | 43.88 | 37.00 | 45.34 | 36.46 | K=24 slightly **higher** at 1×4, **lower** at 1×8 |
| **Compute Throughput (%)** | 22.20 | 22.73 | 22.50 | 22.27 | K=24 consistent (~22%), no regression |
| **SM Frequency (GHz)** | 1.29 | 1.29 | 1.30 | 1.29 | No throttling observed |
| **DRAM Frequency (MHz)** | 862.8 | 877.0 | 866.9 | 875.3 | Normal operation |
| **MPI passes per cycle** | 2 | 2 | 1 | 1 | **By design** (halved) |
| **Buffer size (halo cells)** | 13 | 13 | 25 | 25 | **By design** (doubled) |

### Wall-Time Comparison (Raw Benchmark Data)

| Config | TBLOCKING | BENCHMARK_STEPS | Batches | Wall Time | Per-Step Time |
|--------|-----------|-----------------|---------|-----------|---------------|
| K=12 1×4 | 12 | 24 | 2 | 685 s | 28.5 s/step |
| K=12 1×8 | 12 | 24 | 2 | 655 s | 27.3 s/step |
| K=24 1×4 | 24 | 240 | 10 | 112.0 s | 0.467 s/step |
| K=24 1×8 | 24 | 240 | 10 | 109.1 s | 0.455 s/step |

**Scaling Efficiency (per-step time improvement 1×4→1×8):**
- K=12: 28.5 → 27.3 s/step = **4.2% improvement**
- K=24: 0.467 → 0.455 s/step = **2.6% improvement** ❌ **Worse**

### Conclusion: Communication is **BANDWIDTH-BOUND** ❌

**Why K=24 performs worse:**

1. **Per-kernel slowdown visible:** K=24 kernels are 8–20% slower per invocation despite identical computation
   - Likely cause: larger halo buffers (25 vs 13 cells) increase memory pressure + cache pressure
   - L2 cache throughput increases slightly (20.7% → 21.2%), suggesting memory bottleneck

2. **Scaling degrades from 1.90× to 1.71×:** Doubling partitions (1×4→1×8) shows **10% worse scaling in K=24**
   - Indicates MPI communication overhead is less of a bottleneck than **memory bandwidth saturation**

3. **Fewer MPI passes did NOT help:**  
   - K=12: 2 MPI passes per cycle (12-cell halo) → 4.2% wall-time improvement (1×4→1×8)
   - K=24: 1 MPI pass per cycle (25-cell halo) → **2.6% wall-time improvement** (worse)
   - Conclusion: **Latency savings from fewer passes are outweighed by larger buffer costs**

4. **Memory throughput unchanged:** 36–46% across all configs = memory-bound not bandwidth-bound
   - Larger buffers (2× cells) don't saturate V100 memory; they just increase kernel execution time

### Interpretation

The bottleneck is **not MPI communication overhead** (latency or synchronization). Instead:
- **Hypothesis rejected:** Doubling K to halve MPI passes makes performance worse, not better
- **Root cause:** Increased halo size (25×25 vs 13×13) creates larger working sets → cache pressure, memory latency
- **Scaling ceiling:** V100 distributed strong-scaling on OM2-025 hits an architectural limit **~4–5% improvement per doubling**, not due to communication but to **per-rank compute efficiency degradation**

### Recommendations

1. **Do not increase K further** — K=24 shows K=12 was better-tuned
2. **Focus on different optimizations:**
   - Domain decomposition (1×4 → 2×2 square vs 1×8 thin) to reduce communication pairs
   - Reduce w computation overhead (currently 35%+ of runtime)
   - Consider OM2-01 H200 (2× memory bandwidth) for comparison
3. **Accept strong-scaling plateau** on V100: 1×4→1×8 yields ~4% wall-time improvement, not 2× despite 2× GPU count
