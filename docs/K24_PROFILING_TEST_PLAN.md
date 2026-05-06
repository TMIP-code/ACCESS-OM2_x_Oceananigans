# K=24 vs K=12 Strong-Scaling NCU Profiling Test Plan

## Objective
Determine if doubling temporal blocking `K` (12→24) improves GPU scaling efficiency by reducing MPI communication overhead, answering: **Is communication latency-bound (solvable) or bandwidth-bound (hard ceiling)?**

## Hypothesis
- K=12: 2 MPI passes per benchmark cycle, buffer size = 12 cells
- K=24: 1 MPI pass per benchmark cycle, buffer size = 24 cells
- **Prediction:** Fewer passes (1 vs 2) reduces synchronization latency → better wall-time scaling.
- **Alternative:** Larger buffers (2× bandwidth per pass) offsets latency savings → no improvement.

## Experimental Setup

### Configuration Comparison

| Aspect | K=12 Baseline | K=24 Test |
|--------|---------------|-----------|
| TBLOCKING | 12 | 24 |
| BENCHMARK_STEPS | 24 | 240 |
| Batches | 2 | 10 (first 2 warmup, measure batches 2–3) |
| MPI passes | 1 (between 2 batches) | 1 (between 10 batches, but only profiling batches 2–3) |
| Halo cells | 13 | 25 |
| GPU | V100 (gpuvolta) | V100 (gpuvolta) |
| Kernel target | `gpu_compute_hydrostatic_free_surface_Gc_` | `gpu_compute_hydrostatic_free_surface_Gc_` |

### Jobs Submitted

| Job ID | Partition | K | GPU | Status | Note |
|--------|-----------|---|-----|--------|------|
| 167715278 | 1×4 | 12 | V100 | Complete | K=12 baseline, 1×4 |
| 167715279 | 1×8 | 12 | V100 | Complete | K=12 baseline, 1×8 |
| **167737076** | **1×4** | **24** | **V100** | **Queued** | K=24 test, 1×4 |
| **167737090** | **1×8** | **24** | **V100** | **Queued** | K=24 test, 1×8 |

### Expected Metrics (from NCU roofline)

**Per-kernel metrics to extract:**
1. **Kernel Duration (avg ms)** — single kernel invocation time
2. **Memory Throughput (%)** — DRAM % of peak bandwidth
3. **Compute Throughput (%)** — SM compute unit utilization
4. **SM Frequency (GHz)** — GPU clock speed (check for throttling)
5. **DRAM Frequency (MHz)** — memory clock

## Analysis Plan

### Step 1: Extract Baseline K=12 Metrics (Already Available)

From PROFILING_RESULTS.md (jobs 167715278/279):
```
K=12, 1×4: Duration=20.38ms, Memory=43.88%, Compute=22.20%
K=12, 1×8: Duration=10.72ms, Memory=37.00%, Compute=22.73%
Scaling: 1.90× kernel speedup, 4.4% wall-time improvement
```

**Interpretation:**
- Per-kernel scales well (1.90×)
- Wall time barely improves (4.4%) → MPI overhead dominates (~95% of speedup lost)

### Step 2: Extract K=24 Metrics (Awaited from Jobs 167737076/090)

Run (once jobs complete):
```bash
module load cuda/12.9.0
ncu --import logs/julia/.../167737076*ncu-rep --print-summary per-kernel 2>&1 | head -150
ncu --import logs/julia/.../167737090*ncu-rep --print-summary per-kernel 2>&1 | head -150
```

### Step 3: Side-by-Side Comparison

Expected table:

| Metric | K=12 1×4 | K=12 1×8 | K=24 1×4 | K=24 1×8 | Analysis |
|--------|----------|----------|----------|----------|----------|
| Duration (ms) | 20.38 | 10.72 | ? | ? | Smaller = better per-kernel scaling |
| Memory (%) | 43.88 | 37.00 | ? | ? | >40% = memory-bound; <37% = compute-bound |
| Compute (%) | 22.20 | 22.73 | ? | ? | Should stay constant across K values |
| SM Freq (GHz) | 1.29 | 1.29 | ? | ? | Check for throttling differences |
| DRAM Freq (MHz) | 862.8 | 877.0 | ? | ? | Check for frequency shifts |

### Step 4: Interpret Results

**Success Criteria (each K=24 should improve K=12):**

1. **Per-kernel scaling is maintained or improves** → Duration ratio K=24(1×8)/K=24(1×4) should still be ~1.9×
   - If ratio degrades → MPI congestion visible in kernel execution
   
2. **Memory throughput stays consistent or increases** → K=24 memory ≥ 37% in 1×8 config
   - If drops below 37% → buffer-size overhead matters
   
3. **Wall-time scaling improves** → Need wall-time comparison (from PBS logs)
   - K=12: 655s (1×8) vs 685s (1×4) = 4.4% improvement
   - K=24: wall time ratio should be >4.4% (ideally 10–15%)

**Interpretation:**
- **Improvement on all three criteria** → Communication is latency-bound; fewer passes help
- **No improvement on wall time** → Communication is bandwidth-bound; larger buffers dominate
- **Partial improvement** → Communication is mixed-mode; may need domain-decomposition tuning

## Files and Commands

### Extraction Script
```bash
bash scripts/extract_ncu_metrics.sh 167737076
bash scripts/extract_ncu_metrics.sh 167737090
```

### Update PROFILING_RESULTS.md
Once results are available, fill in the K=24 section with side-by-side metrics.

## Follow-Up Actions

Based on results:

**If latency-bound (improvement observed):**
- Investigate further K>24 options (K=32, K=48) to push limit
- Estimate communication cost as function of K
- Consider domain-decomposition changes (1×4→2×2) to reduce MPI pairs

**If bandwidth-bound (no improvement):**
- Focus optimization on other bottlenecks (GPU memory pressure, w computation)
- Accept strong-scaling plateau as architectural limit
- Consider OM2-01 H200 comparison (2× memory bandwidth vs V100)

**Either way:**
- Document findings in PROFILING_RESULTS.md for future reference
- Update NCU_PROFILING_METHODOLOGY.md with lessons learned
