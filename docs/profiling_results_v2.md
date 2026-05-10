# Profiling Results v2: Systematic Benchmark Matrix

Wall-clock times from the 46-job profiling matrix defined in [profiling_plan.md](profiling_plan.md). Each cell is the elapsed walltime from PBS (`resources_used.walltime`).

- **bench** = full 1-year run via `run_1year_benchmark.jl` (no I/O), variable step count per model (5,844 / 17,532 / 78,894 steps)
- **nsys** = 240-step trace (360-step for OM2-01 +TB to satisfy K=18 divisibility)
- Time window: `1968-1977`, halos=13 (except OM2-01 +TB which used halos=19 for K=18)

---

## Phase 1: OM2-1 — V100 (`gpuvolta`)

| Partition | baseline bench | baseline nsys | +GC bench | +GC nsys | +TB bench (K=12) | +TB nsys (K=12) | +LB bench | +LB nsys |
|-----------|----------------|---------------|-----------|----------|-------------------|------------------|-----------|----------|
| 1x1 | 8m 12s | 8m 14s | — | — | — | — | — | — |
| 1x2 | 10m 57s | 9m 13s | 9m 20s | 9m 54s | **8m 43s** | 9m 35s | 10m 35s | 10m 41s |
| 1x4 | 9m 12s | 10m 0s | — | — | — | — | — | — |
| 1x8 | 9m 23s | 10m 29s | — | — | — | — | — | — |

---

## Phase 2: OM2-025 — V100 vs H200

### V100 (`gpuvolta`)

| Partition | baseline bench | baseline nsys | +GC bench | +GC nsys | +TB bench (K=12) | +TB nsys (K=12) | +LB bench | +LB nsys |
|-----------|----------------|---------------|-----------|----------|-------------------|------------------|-----------|----------|
| 1x2 | 25m 27s | 11m 59s | 25m 11s | 12m 14s | **24m 1s** | 11m 33s | 25m 12s | 12m 22s |

### H200 (`gpuhopper`)

| Partition | baseline bench | baseline nsys | +GC bench | +GC nsys | +TB bench (K=12) | +TB nsys (K=12) | +LB bench | +LB nsys |
|-----------|----------------|---------------|-----------|----------|-------------------|------------------|-----------|----------|
| 1x2 | 13m 41s | 10m 14s | 14m 7s | 10m 14s | **12m 54s** | 10m 2s | 14m 14s | 10m 41s |
| 1x4 | 11m 4s | 8m 55s | — | — | — | — | — | — |
| 1x8 | 12m 4s | 19m 3s | — | — | — | — | — | — |

---

## Phase 3: OM2-01 — H200 (`gpuhopper`)

| Partition | baseline bench | baseline nsys | +GC bench | +GC nsys | +TB bench (K=18*) | +TB nsys (K=18*) | +LB bench | +LB nsys |
|-----------|----------------|---------------|-----------|----------|--------------------|-------------------|-----------|----------|
| 1x2 | 3h 22m 3s | 19m 0s | 3h 24m 21s | 19m 5s | **3h 7m 7s** | 24m 23s | 3h 17m 56s | 22m 28s |
| 1x4 | 1h 52m 33s | 17m 20s | — | — | — | — | — | — |
| 1x8 | 2h 5m 2s | 31m 28s | — | — | — | — | — | — |

*K=18 used for OM2-01 because K=12 doesn't divide 78,894 steps/yr. Halos=19 partition built specifically for this run. nsys uses BENCHMARK_STEPS=360 (= 20×18) for 20 MPI passes.

---

## Derived: +Trick speedup vs baseline (1x2 bench)

Time relative to baseline at the same partition (1x2). Speedup = (baseline − trick) / baseline.

| Model | Hardware | Baseline | +GC | +TB | +LB |
|-------|----------|----------|-----|-----|-----|
| OM2-1 | V100 | 10m 57s | 9m 20s (**14.7%**) | 8m 43s (**20.4%**) | 10m 35s (3.3%) |
| OM2-025 | V100 | 25m 27s | 25m 11s (1.0%) | 24m 1s (**5.6%**) | 25m 12s (1.0%) |
| OM2-025 | H200 | 13m 41s | 14m 7s (−3.2%) | 12m 54s (**5.7%**) | 14m 14s (−4.0%) |
| OM2-01 | H200 | 3h 22m 3s | 3h 24m 21s (−1.1%) | 3h 7m 7s (**7.4%**) | 3h 17m 56s (2.0%) |

**Observations:**
- **+TB consistently helps** across all 4 (model, hardware) combinations: 5.6–20.4% speedup.
- **+GC and +LB are mostly noise** at 1x2 — sometimes a small win, sometimes a small loss. They will matter more at larger partition sizes where halo-exchange and load-imbalance overhead dominate, but Phase 2/3 only tested baseline beyond 1x2.
- **OM2-1 +TB win (20%)** is the largest: at 1° resolution, halo-exchange is a relatively large fraction of total time, so amortizing it across blocks pays off most.

---

## Derived: V100 vs H200 (OM2-025 1x2)

Speedup = V100 time / H200 time.

| Config | V100 bench | H200 bench | Speedup | V100 nsys | H200 nsys | Speedup |
|--------|-----------|-----------|---------|-----------|-----------|---------|
| baseline | 25m 27s | 13m 41s | **1.86×** | 11m 59s | 10m 14s | 1.17× |
| +GC | 25m 11s | 14m 7s | 1.78× | 12m 14s | 10m 14s | 1.20× |
| +TB | 24m 1s | 12m 54s | 1.86× | 11m 33s | 10m 2s | 1.15× |
| +LB | 25m 12s | 14m 14s | 1.77× | 12m 22s | 10m 41s | 1.16× |

**Observations:**
- **Bench speedup ~1.8× on H200** — close to the theoretical memory bandwidth ratio (H200 has ~2× HBM bandwidth over V100). Consistent with the workload being memory-bound.
- **nsys speedup only ~1.15–1.20×** — short profiled runs (240 steps) are dominated by warmup, JIT, and capture overhead, not steady-state compute. Bench is the right comparison for hardware throughput.

---

## Derived: Strong scaling on baseline bench

How wall time changes as partition grows. Ideal strong scaling would halve walltime with each doubling of GPUs.

### OM2-1 (V100)

| Partition | bench time | speedup vs 1x1 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x1 | 8m 12s | 1.00× | 1× | 100% |
| 1x2 | 10m 57s | 0.75× | 2× | **37%** |
| 1x4 | 9m 12s | 0.89× | 4× | 22% |
| 1x8 | 9m 23s | 0.87× | 8× | 11% |

### OM2-025 (H200)

| Partition | bench time | speedup vs 1x2 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x2 | 13m 41s | 1.00× | 1× | 100% |
| 1x4 | 11m 4s | 1.24× | 2× | **62%** |
| 1x8 | 12m 4s | 1.13× | 4× | 28% |

### OM2-01 (H200)

| Partition | bench time | speedup vs 1x2 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x2 | 3h 22m 3s | 1.00× | 1× | 100% |
| 1x4 | 1h 52m 33s | 1.79× | 2× | **90%** |
| 1x8 | 2h 5m 2s | 1.62× | 4× | 40% |

**Observations:**
- **Strong scaling is dramatically better at finer resolution.** OM2-01 1x2→1x4 hits 90% efficiency (near-ideal); OM2-1 1x1→1x2 is only 37%. This is the classic surface-area-to-volume effect: compute scales as Nx·Ny·Nz per rank, communication scales as halo area per rank — bigger problems amortize communication better.
- **All models plateau or regress between 1x4 and 1x8.** OM2-1 1x4→1x8 actually slows down. Likely a load-imbalance or halo-cost crossover, or H200 inter-node MPI overhead.
- **OM2-01 1x8 (2h 5m) is slower than 1x4 (1h 52m)** — same pattern. Distributed scaling beyond 4 GPUs is unproductive on this workload at current settings.

---

## Cross-resolution bench scaling (baseline, 1x2)

To put resolution in perspective: how much harder is each step at finer resolution?

| Model | Steps/yr | Bench (V100 1x2) | Bench (H200 1x2) | Time/step (V100) | Time/step (H200) |
|-------|---------:|------------------|------------------|------------------|------------------|
| OM2-1 | 5,844 | 10m 57s = 657s | — | **112 ms** | — |
| OM2-025 | 17,532 | 25m 27s = 1527s | 13m 41s = 821s | **87 ms** | **47 ms** |
| OM2-01 | 78,894 | — | 12,123s | — | **154 ms** |

**Observation:** Time per step *decreases* from OM2-1 → OM2-025 on V100 (112 → 87 ms), which is initially counter-intuitive given the larger grid. But OM2-1 1x2 has small per-rank work; relatively more time is spent in framework overhead per step. OM2-025 amortizes that overhead better. OM2-01 has the largest per-rank work and so the highest per-step cost — but it's still only ~3.3× slower per step than OM2-025 on H200 despite a 13.5× increase in step count (each step works on much bigger arrays).
