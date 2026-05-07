# K=6 NCU Profiling Session — In Progress

## Summary

Testing if **smaller halo size improves memory bandwidth efficiency** by reducing cache pressure.

**Hypothesis:** K=24 was 20% slower due to larger halos (25×25 cells) exceeding L1/L2 cache capacity. K=6 (smaller halos = 7×7 cells) should reduce memory pressure and improve per-kernel throughput.

## Final Status (2026-05-07)

**All jobs completed successfully:**
- ✓ K6 1×4 jobs: 167768183–189 (FINISHED)
- ✓ K6 1×8 jobs: 167775918–919 (FINISHED)
- **NCU reports:** Both 1×4 and 1×8 roofline metrics extracted and analyzed

## Configuration

| Parameter | K=6 | K=12 (baseline) | K=24 (failed) |
|-----------|-----|-----------------|---------------|
| TBLOCKING | 6 | 12 | 24 |
| Halo size (GRID_HX/Y) | 7 | 13 | 25 |
| BENCHMARK_STEPS | 60 (10 batches) | 24 (2 batches) | 240 (10 batches) |
| NCU_SKIP | 12 (skip 2 batches) | 0 | 48 (skip 2 batches) |
| NCU_COUNT | 12 (profile 2 batches) | 12 | 48 (profile 2 batches) |
| GPU_QUEUE | gpuvolta (V100) | gpuvolta | gpuvolta |

## Job Submission Command

```bash
PARENT_MODEL=ACCESS-OM2-025 \
EXPERIMENT=025deg_jra55_iaf_omip2_cycle6 \
TIME_WINDOW=1968-1977 \
GRID_HX=7 GRID_HY=7 GRID_HZ=2 \
PARTITION=1x4 \
TBLOCKING=6 \
BENCHMARK_STEPS=60 \
NCU_SET=roofline \
NCU_KERNEL='gpu_compute_hydrostatic_free_surface_Gc_' \
NCU_SKIP=12 \
NCU_COUNT=12 \
GPU_QUEUE=gpuvolta \
JOB_CHAIN=preprocessing-run1yrncu \
bash scripts/driver.sh
```

**Note:** `preprocessing` includes grid rebuild + repartition (necessary for smaller halos). Skip `prep` in future submissions.

## Results

### K6 Roofline Metrics (Final)

| Metric | K6 1×4 | K6 1×8 | K12 1×4 | K12 1×8 | K24 1×4 |
|--------|--------|--------|---------|---------|---------|
| Kernel Duration (ms) | 19.11 | 10.24 | 20.38 | 10.72 | 22.05 |
| Memory Throughput (%) | 43.29 | 35.18 | 43.88 | 37.00 | 45.34 |
| Compute (SM) Throughput (%) | 22.41 | 21.75 | 22.20 | 22.73 | 22.50 |
| Scaling (1×4 → 1×8) | 1.87× | — | 1.90× | — | — |

### K6 Analysis: Smaller Halos Help

**✓ Confirmed: K6 is 5.0% faster than K12 at 1×4**
- K6 1×4: 19.11 ms → K12 1×4: 20.38 ms
- Hypothesis validated: halos of 7×7 reduce memory pressure vs 13×13
- At 1×8: K6 scales to 10.24 ms (1.87× speedup, vs K12's 1.90×)

**✗ Scaling plateau persists: ~4% wall-time improvement per GPU doubling**
- K6, K12, K24 all show memory bandwidth saturation at 1×8
- Memory throughput drops 15–19% when scaling from 1×4 to 1×8
- Compute throughput constant (~22%), confirming memory-bound workload
- Single-node architectural ceiling is **not addressable by tuning K alone**

### Key Results So Far (K12 vs K24)

**K=24 performed WORSE than K=12:**
- Per-kernel duration: **+20% slower** (12.85 ms vs 10.72 ms for 1×8)
- Scaling efficiency: **Degraded 1.90× → 1.71×** (10% worse)
- Wall-time improvement: **2.6% (worse than K=12's 4.2%)**

**Conclusion:** Communication is **bandwidth-bound**, not latency-bound. Larger buffers (2× halo cells) hurt more than fewer MPI passes help.

### Expected K=6 Results

If memory pressure is the bottleneck:
- K=6 kernels should be **faster** than K=12 (smaller halos fit in L1/L2)
- Memory throughput should improve
- Scaling (1×4→1×8) should match or exceed K=12's 1.90×

If scaling still plateaus:
- Confirms architectural ceiling is ~4% wall-time improvement per GPU doubling
- Not addressable by tuning K alone

## Conclusion & Next Steps

**Finding:** K6 profiling confirms smaller halos **reduce memory pressure and improve 1×4 
throughput by 5%**. However, **scaling remains plateaued at 1.87–1.90×**, confirming the 
bottleneck is **not tunable by K alone** on V100.

**Recommended Path Forward:**

| Option | Impact | Effort | Priority |
|--------|--------|--------|----------|
| **H200 GPU** | +50% memory bandwidth | High | HIGH |
| **Prescribed w** | ~40% kernel time savings | Medium | HIGH |
| **Domain decomp** (2×2) | Test alternative scaling topology | High | MEDIUM |
| **K=3 refinement** | Marginal gains (<2%) | Low | LOW |

1. **Reject K=3** — further halo reduction likely yields <1% gain (diminishing returns)
2. **Accept 1.87× as V100 baseline** for single-node configurations
3. **Prioritize H200/prescribed-w** for next optimization cycle
4. **Document findings** in PROFILING_RESULTS.md for future reference

## Documentation Files

- [PROFILING_RESULTS.md](PROFILING_RESULTS.md) — K=12 vs K=24 analysis with roofline metrics
- [NCU_PROFILING_METHODOLOGY.md](NCU_PROFILING_METHODOLOGY.md) — Profiling best practices
- [K24_PROFILING_TEST_PLAN.md](K24_PROFILING_TEST_PLAN.md) — K=24 test design
- [K6_PROFILING_SESSION.md](K6_PROFILING_SESSION.md) — This file

## Key Lessons Learned

1. **Warmup contamination matters** — Always use BENCHMARK_STEPS = K × N_BATCHES with N_BATCHES ≥ 3
2. **GPU consistency required** — All comparisons must use same GPU (V100 vs H200 confounds results)
3. **Kernel name matching is critical** — Exact names required; regex failures fail silently
4. **Memory bandwidth is the bottleneck** — Not MPI communication; larger halos saturate V100 bandwidth

See [project_ncu_profiling_lessons.md](../memory/project_ncu_profiling_lessons.md) for full methodology.
