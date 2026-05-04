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
