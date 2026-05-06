# K=6 NCU Profiling Session — In Progress

## Summary

Testing if **smaller halo size improves memory bandwidth efficiency** by reducing cache pressure.

**Hypothesis:** K=24 was 20% slower due to larger halos (25×25 cells) exceeding L1/L2 cache capacity. K=6 (smaller halos = 7×7 cells) should reduce memory pressure and improve per-kernel throughput.

## Current Status (2026-05-06)

**Job Chain Running:**
- Job 167768183 (prep) — RUNNING (~54 min elapsed, near completion)
- Jobs 167768185-189 — HELD, waiting for prep to finish
- **Final NCU job: 167768189** (run1yrncu)

**Do NOT interrupt** — Python preprocessing can corrupt files if killed mid-run. Let it complete.

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

## Key Results So Far

### K=12 vs K=24 Findings

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

## Next Steps

1. **Wait for prep job to finish** (~15–30 min remaining estimate)
2. **Vel, clo, diagw, partition jobs will cascade** (~40–50 min total)
3. **K=6 NCU profiling will start** and run ~2–3 hours
4. **Extract metrics** once complete:
   ```bash
   bash scripts/extract_ncu_metrics.sh 167768189
   ```
5. **Compare K=6 vs K=12 vs K=24** in PROFILING_RESULTS.md
6. **Decide next steps** based on results:
   - If K=6 improves: test K=3 or investigate domain decomposition (1×4 → 2×2)
   - If K=6 matches K=12: accept scaling plateau, focus on other optimizations (prescribed w, H200 GPU)

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
