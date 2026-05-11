# Profiling Results v2: Systematic Benchmark Matrix

Wall-clock times for the **bench** runs from the 46-job profiling matrix defined in [profiling_plan.md](profiling_plan.md). All times are the **max across ranks** of the in-process timer at the end of the 1-year benchmark loop (extracted from the Julia log, **not** the PBS walltime — PBS walltime includes setup/JIT and is misleading especially for small models).

Times and plots extracted with [`scripts/plotting/plot_simtime_vs_walltime.py`](../scripts/plotting/plot_simtime_vs_walltime.py):
- Non-TB runs: from `Simulation is stopping after running for X seconds` (post-warmup, steady-state simulation loop only).
- TB runs: from `elapsed_seconds = X` in `Benchmark complete` (TB uses a custom loop with no per-rank stop line). **NB: TB wall includes the first-batch JIT cost** (~80s on OM2-1, smaller relative to steady-state on larger models), so direct baseline-vs-TB comparison overstates the JIT for small models.

Each cell shows max-rank simulation walltime + a per-rank time-vs-wall plot ([`docs/profiling_plots/`](profiling_plots/)). Time window: `1968-1977`, halos=13 throughout except OM2-01 +TB which uses halos=19 for K=18.

---

## Phase 1: OM2-1 — V100 (`gpuvolta`)

| Partition | baseline | +GC | +TB (K=12) | +LB |
|-----------|----------|-----|------------|-----|
| 1x1 | **38.6s**<br><img src="profiling_plots/OM2-1_1x1_baseline_167891368.png" width="240"> | — | — | — |
| 1x2 | **29.6s**<br><img src="profiling_plots/OM2-1_1x2_baseline_167891390.png" width="240"> | **28.9s**<br><img src="profiling_plots/OM2-1_1x2_GC_167891392.png" width="240"> | **1m 45.0s\***<br><img src="profiling_plots/OM2-1_1x2_TB_167891394.png" width="240"> | **27.0s**<br><img src="profiling_plots/OM2-1_1x2_LB_167931014.png" width="240"> |
| 1x4 | **23.2s**<br><img src="profiling_plots/OM2-1_1x4_baseline_167891869.png" width="240"> | — | — | — |
| 1x8 | **27.4s**<br><img src="profiling_plots/OM2-1_1x8_baseline_167891401.png" width="240"> | — | — | — |

\* +TB on OM2-1 is dominated by first-batch JIT (~80s) — see notes. Steady-state +TB step rate is actually faster than baseline (~21s for 92% of remaining work).

---

## Phase 2: OM2-025 — V100 vs H200

### V100 (`gpuvolta`)

| Partition | baseline | +GC | +TB (K=12) | +LB |
|-----------|----------|-----|------------|-----|
| 1x2 | **15m 26.4s**<br><img src="profiling_plots/OM2-025_V100_1x2_baseline_167950637.png" width="240"> | **15m 29.0s**<br><img src="profiling_plots/OM2-025_V100_1x2_GC_167950639.png" width="240"> | **14m 48.9s**<br><img src="profiling_plots/OM2-025_V100_1x2_TB_167950641.png" width="240"> | **15m 28.7s**<br><img src="profiling_plots/OM2-025_V100_1x2_LB_167950643.png" width="240"> |

### H200 (`gpuhopper`)

| Partition | baseline | +GC | +TB (K=12) | +LB |
|-----------|----------|-----|------------|-----|
| 1x2 | **5m 34.6s**<br><img src="profiling_plots/OM2-025_H200_1x2_baseline_167950650.png" width="240"> | **6m 1.2s**<br><img src="profiling_plots/OM2-025_H200_1x2_GC_167950652.png" width="240"> | **5m 7.2s**<br><img src="profiling_plots/OM2-025_H200_1x2_TB_167950654.png" width="240"> | **5m 40.1s**<br><img src="profiling_plots/OM2-025_H200_1x2_LB_167950656.png" width="240"> |
| 1x4 | **3m 48.8s**<br><img src="profiling_plots/OM2-025_H200_1x4_baseline_167950658.png" width="240"> | — | — | — |
| 1x8 | **3m 53.4s**<br><img src="profiling_plots/OM2-025_H200_1x8_baseline_167950660.png" width="240"> | — | — | — |

---

## Phase 3: OM2-01 — H200 (`gpuhopper`)

| Partition | baseline | +GC | +TB (K=18) | +LB |
|-----------|----------|-----|------------|-----|
| 1x2 | **3h 1m 47s**<br><img src="profiling_plots/OM2-01_1x2_baseline_167976668.png" width="240"> | **3h 3m 42s**<br><img src="profiling_plots/OM2-01_1x2_GC_167976670.png" width="240"> | **2h 45m 36s**<br><img src="profiling_plots/OM2-01_1x2_TB_168021140.png" width="240"> | **2h 55m 52s**<br><img src="profiling_plots/OM2-01_1x2_LB_167976674.png" width="240"> |
| 1x4 | **1h 37m 59s**<br><img src="profiling_plots/OM2-01_1x4_baseline_167976676.png" width="240"> | — | — | — |
| 1x8 | **1h 38m 27s**<br><img src="profiling_plots/OM2-01_1x8_baseline_167976678.png" width="240"> | — | — | — |

---

## Derived: +Trick speedup vs baseline (1x2 bench)

Speedup = (baseline − trick) / baseline. Negative = slower than baseline.

| Model | Hardware | Baseline | +GC | +TB | +LB |
|-------|----------|----------|-----|-----|-----|
| OM2-1 | V100 | 29.6s | 28.9s (**+2.4%**) | 1m 45.0s (**−255%, JIT-dominated**) | 27.0s (**+8.8%**) |
| OM2-025 | V100 | 15m 26.4s | 15m 29.0s (−0.3%) | 14m 48.9s (**+4.0%**) | 15m 28.7s (−0.2%) |
| OM2-025 | H200 | 5m 34.6s | 6m 1.2s (−7.9%) | 5m 7.2s (**+8.2%**) | 5m 40.1s (−1.6%) |
| OM2-01 | H200 | 3h 1m 47s | 3h 3m 42s (−1.1%) | 2h 45m 36s (**+8.9%**) | 2h 55m 52s (**+3.3%**) |

**Observations:**
- **+TB**: Net +4 to +9% speedup at OM2-025/OM2-01 once steady-state savings overcome the first-batch JIT cost (~80s absolute, small relative to total at these scales). On OM2-1 where the whole bench is only ~30s, the same JIT cost looks like a 255% regression.
- **+LB**: Marginal — wins at OM2-1 (+8.8%) and OM2-01 (+3.3%), neutral or slight loss at OM2-025. The trick targets cell-imbalance overhead, which is most pronounced at coarse resolution (more pole-relative variation per rank).
- **+GC**: Mostly noise (±2%) at most configs; meaningful loss on OM2-025 H200 (−7.9%) — possibly nondeterministic scheduling on that hardware.

---

## Derived: V100 vs H200 (OM2-025 1x2)

Speedup = V100 time / H200 time.

| Config | V100 | H200 | Speedup |
|--------|------|------|---------|
| baseline | 15m 26.4s | 5m 34.6s | **2.77×** |
| +GC | 15m 29.0s | 6m 1.2s | 2.57× |
| +TB | 14m 48.9s | 5m 7.2s | **2.89×** |
| +LB | 15m 28.7s | 5m 40.1s | 2.73× |

H200 is consistently ~2.7–2.9× faster than V100 for steady-state simulation. (The PBS-walltime ratio was only ~1.8× because Julia startup time is similar on both, diluting the speedup; on the actual simulation loop, the memory-bandwidth ratio dominates.)

---

## Derived: Strong scaling on baseline bench

How wall time changes as partition grows. Ideal strong scaling would halve walltime with each doubling of GPUs.

### OM2-1 (V100)

| Partition | bench time | speedup vs 1x1 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x1 | 38.6s | 1.00× | 1× | 100% |
| 1x2 | 29.6s | 1.30× | 2× | 65% |
| 1x4 | 23.2s | 1.66× | 4× | **42%** |
| 1x8 | 27.4s | 1.41× | 8× | 18% |

### OM2-025 (H200)

| Partition | bench time | speedup vs 1x2 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x2 | 5m 34.6s | 1.00× | 1× | 100% |
| 1x4 | 3m 48.8s | 1.46× | 2× | **73%** |
| 1x8 | 3m 53.4s | 1.43× | 4× | 36% |

### OM2-01 (H200)

| Partition | bench time | speedup vs 1x2 | ideal | efficiency |
|-----------|-----------|----------------|-------|-----------|
| 1x2 | 3h 1m 47s | 1.00× | 1× | 100% |
| 1x4 | 1h 37m 59s | **1.85×** | 2× | **93%** |
| 1x8 | 1h 38m 27s | 1.85× | 4× | 46% |

**Observations:**
- **OM2-01 1x2→1x4 hits 93% scaling efficiency** — near-ideal, the strongest result in the matrix. Finer resolution means more compute per rank, so halo-exchange overhead is proportionally smaller.
- **1x4→1x8 plateaus or regresses everywhere.** OM2-01 1x4 and 1x8 give the same wall time (1h 38m). OM2-1 actually slows down. This is the same pattern seen in the PBS-time analysis but now confirmed on the simulation loop alone — it's a genuine workload property, not a setup-time artifact.

---

## Cross-resolution per-step cost (baseline, 1x2)

| Model | Steps/yr | Bench (V100 1x2) | Bench (H200 1x2) | Time/step (V100) | Time/step (H200) |
|-------|---------:|------------------|------------------|------------------|------------------|
| OM2-1 | 5,844 | 29.6s | — | **5.1 ms** | — |
| OM2-025 | 17,532 | 926.4s | 334.6s | **52.8 ms** | **19.1 ms** |
| OM2-01 | 78,894 | — | 10,907s | — | **138.3 ms** |

**Observation:** Per-step cost scales roughly with grid volume:
- OM2-1 → OM2-025 (per-step): V100 5.1 → 52.8 ms = 10.3× slower per step (grid is ~25× larger, but per-step work per cell increases superlinearly with sub-step counts).
- OM2-025 → OM2-01 (per-step on H200): 19.1 → 138.3 ms = 7.2× slower per step.

Per-cell work per step has crept up at finer resolution, suggesting either non-uniform work distribution or more complex per-step kernels (free-surface sub-steps, halo exchanges) dominating.

---

## PBS walltime vs simulation walltime

The PBS walltimes used in earlier reports massively overstate the simulation cost for small models, because Julia startup, package loading, partition reading, and warmup can take many minutes regardless of model size. Setup tax breakdown:

| Config | PBS walltime | Sim walltime | Setup tax | Setup % |
|--------|-------------:|-------------:|----------:|--------:|
| OM2-1 1x2 baseline | 10m 57s | 29.6s | **10m 27s** | 95% |
| OM2-025 V100 1x2 baseline | 25m 27s | 15m 26s | 10m 1s | 39% |
| OM2-025 H200 1x2 baseline | 13m 41s | 5m 35s | 8m 6s | 59% |
| OM2-01 1x2 baseline | 3h 22m 3s | 3h 1m 47s | 20m 16s | 10% |

The setup tax is roughly model-independent at ~10–20 minutes (Julia init dominates), so it crushes OM2-1 walltimes but is a small fraction of OM2-01 walltimes.

**For all future benchmark analysis: use sim walltime, not PBS walltime.** The `plot_simtime_vs_walltime.py --no-plot` tool extracts it from the Julia logs in seconds.
