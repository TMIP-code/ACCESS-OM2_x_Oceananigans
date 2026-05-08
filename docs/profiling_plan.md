# Profiling Plan: Systematic Benchmark + nsys Matrix

## Context

Previous profiling was ad-hoc, making cross-configuration comparisons unreliable. Goal: a clean apples-to-apples matrix — one trick at a time, consistent base settings, both a full-year wall-time benchmark and a short nsys profile per entry. Start with OM2-1 to validate the workflow, then repeat for OM2-025 and OM2-01.

Tricks (GC sync, temporal blocking, load balancing) are only meaningful for distributed runs — never tested on 1x1. Each trick is tested only at the smallest distributed partition (1x2). Larger partitions (1x4, 1x8) run baseline-only to characterise raw scaling.

**This file is the cross-session results record — update it as results come in.**

---

## Baseline Configuration (all simulation jobs)

| Variable | Value |
|---|---|
| `VELOCITY_SOURCE` | `cgridtransports` |
| `W_FORMULATION` | `wdiagnosed` |
| `ADVECTION_SCHEME` | `centered2` |
| `TIMESTEPPER` | `AB2` |
| `TBLOCKING` | `no` |
| `SYNC_GC_NSTEPS` | (unset) |
| `PARTITION` | standard (no `_LB`) |
| `TIME_WINDOW` | `1968-1977` |
| `GRID_HX`, `GRID_HY` | `13` (all jobs; supports +TB) |

---

## Tricks (tested one at a time, only at 1x2)

| Name | Variable | Value |
|---|---|---|
| **+GC** | `SYNC_GC_NSTEPS` | `5` |
| **+TB** | `TBLOCKING` | `12` (requires halos ≥13; all partitions rebuilt with halos=13) |
| **+LB** | `PARTITION` | `{NxM}_LB` (`LOAD_BALANCE=cell`, cell-balanced y-partition) |

---

## Job Types (2 per non-empty cell)

| Type | JOB_CHAIN / flags | Purpose |
|---|---|---|
| **bench** | `run1yrfast` | full 1-year run; throughput measurement |
| **nsys** | `run1yrfast PROFILE=yes BENCHMARK_STEPS=240` | 240-step nsys trace |

**nsys flags (both serial and distributed):** use `--stats=true`.
- Serial: `nsys profile --stats=true --trace=nvtx,cuda --capture-range=cudaProfilerApi ...`
- Distributed: `nsys profile --stats=true --trace=nvtx,cuda,mpi --capture-range=cudaProfilerApi mpiexec ... julia ...` (single `.nsys-rep` with all ranks on the same timeline)

---

## Phase 1 (Validation): OM2-1 Matrix — V100 (`GPU_QUEUE=gpuvolta`)

Run these first. Goal is to confirm the pipeline works end-to-end — grid rebuild, partition builds, benchmark runs, nsys capture, unified timeline output — before committing compute to larger models. Analyse results before moving to Phase 2.

| Partition | baseline | +GC | +TB | +LB |
|-----------|----------|-----|-----|-----|
| **1x1** | bench + nsys | — | — | — |
| **1x2** | bench + nsys | bench + nsys | bench + nsys | bench + nsys |
| **1x4** | bench + nsys | — | — | — |
| **1x8** | bench + nsys | — | — | — |

**14 simulation jobs.**

### Build Results (OM2-1)

| Step | Partition | Job ID | Status | Elapsed | Notes |
|------|-----------|--------|--------|---------|-------|
| 1 | grid | 167855303 | ✓ | 3m 25s | |
| 1 | vel | 167855304 | ✓ | 7m 17s | |
| 1 | 1x2 | 167855305 | ✓ | 4m 13s | |
| 2 | 1x4 | 167861041 | ✓ | 7m 11s | |
| 2 | 1x8 | 167861043 | ✓ | 5m 17s | |
| 3 | 1x2_LB | 167878729 | ✗ | 4m 25s | OOM-killed during `w` file; only u/v written. **Needs rebuild with more memory.** |

### Simulation Results (fill in as jobs complete)

| Partition | Config | Job ID | Status | Wall time | Notes |
|-----------|--------|--------|--------|-----------|-------|
| 1x1 | baseline bench | 167891368 | ✓ | 8m 12s | |
| 1x1 | baseline nsys | 167891388 | ✓ | 8m 14s | |
| 1x2 | baseline bench | 167891390 | ✓ | 10m 57s | |
| 1x2 | baseline nsys | 167891391 | ✓ | 9m 13s | |
| 1x2 | +GC bench | 167891392 | ✓ | 9m 20s | |
| 1x2 | +GC nsys | 167891393 | ✓ | 9m 54s | |
| 1x2 | +TB bench | 167891394 | ✓ | 8m 43s | |
| 1x2 | +TB nsys | 167891398 | ✓ | 9m 35s | |
| 1x2 | +LB bench | 167891866 | ✗ | 4m 46s | Failed: incomplete 1x2_LB partition (missing eta/w files) |
| 1x2 | +LB nsys | 167891868 | ✗ | 4m 59s | Failed: incomplete 1x2_LB partition (missing eta/w files) |
| 1x4 | baseline bench | 167891869 | Q | — | |
| 1x4 | baseline nsys | 167891870 | Q | — | |
| 1x8 | baseline bench | 167891401 | Q | — | |
| 1x8 | baseline nsys | 167891402 | Q | — | |

---

## Phase 2: OM2-025 Matrix

V100 vs H200 hardware comparison at 1x2 only (with all tricks). Larger partitions baseline-only on H200.

| Partition | GPU | baseline | +GC | +TB | +LB |
|-----------|-----|----------|-----|-----|-----|
| **1x2** | V100 | bench + nsys | bench + nsys | bench + nsys | bench + nsys |
| **1x2** | H200 | bench + nsys | bench + nsys | bench + nsys | bench + nsys |
| **1x4** | H200 | bench + nsys | — | — | — |
| **1x8** | H200 | bench + nsys | — | — | — |

**20 simulation jobs.**

### Build Results (OM2-025)

| Step | Partition | GPU | Job ID | Status | Elapsed | Notes |
|------|-----------|-----|--------|--------|---------|-------|
| | | | | Q | — | Pending |

### Simulation Results (fill in as jobs complete)

| Partition | GPU | Config | Job ID | Wall time | Notes |
|-----------|-----|--------|--------|-----------|-------|
| 1x2 | V100 | baseline bench | | | |
| 1x2 | V100 | baseline nsys | | | |
| 1x2 | V100 | +GC bench | | | |
| 1x2 | V100 | +GC nsys | | | |
| 1x2 | V100 | +TB bench | | | |
| 1x2 | V100 | +TB nsys | | | |
| 1x2 | V100 | +LB bench | | | |
| 1x2 | V100 | +LB nsys | | | |
| 1x2 | H200 | baseline bench | | | |
| 1x2 | H200 | baseline nsys | | | |
| 1x2 | H200 | +GC bench | | | |
| 1x2 | H200 | +GC nsys | | | |
| 1x2 | H200 | +TB bench | | | |
| 1x2 | H200 | +TB nsys | | | |
| 1x2 | H200 | +LB bench | | | |
| 1x2 | H200 | +LB nsys | | | |
| 1x4 | H200 | baseline bench | | | |
| 1x4 | H200 | baseline nsys | | | |
| 1x8 | H200 | baseline bench | | | |
| 1x8 | H200 | baseline nsys | | | |

---

## Phase 3: OM2-01 Matrix — H200 (`GPU_QUEUE=gpuhopper`)

Skip 1x1 (doesn't fit on H200).

| Partition | baseline | +GC | +TB | +LB |
|-----------|----------|-----|-----|-----|
| **1x2** | bench + nsys | bench + nsys | bench + nsys | bench + nsys |
| **1x4** | bench + nsys | — | — | — |
| **1x8** | bench + nsys | — | — | — |

**12 simulation jobs.**

### Build Results (OM2-01)

| Step | Partition | Job ID | Status | Elapsed | Notes |
|------|-----------|--------|--------|---------|-------|
| | | | Q | — | Pending |

### Simulation Results (fill in as jobs complete)

| Partition | Config | Job ID | Wall time | Notes |
|-----------|--------|--------|-----------|-------|
| 1x2 | baseline bench | | | |
| 1x2 | baseline nsys | | | |
| 1x2 | +GC bench | | | |
| 1x2 | +GC nsys | | | |
| 1x2 | +TB bench | | | |
| 1x2 | +TB nsys | | | |
| 1x2 | +LB bench | | | |
| 1x2 | +LB nsys | | | |
| 1x4 | baseline bench | | | |
| 1x4 | baseline nsys | | | |
| 1x8 | baseline bench | | | |
| 1x8 | baseline nsys | | | |

---

## Total: 46 simulation jobs

---

## Prerequisites: Grid, Velocity, and Partition Builds

### Build order matters

The partition step reads halos from `grid.jld2`. The velocity FTS also reference the grid. So to get halos=13 throughout:

**Correct order:** rebuild `grid` (halos=13) → rebuild `vel` → rebuild `partition`

The DAG in the driver enforces this: `grid → vel → partition` (via PBS afterok). Running `JOB_CHAIN=grid-vel-partition` is the cleanest way to do this in one invocation. No `diagw` needed — it is downstream of `vel`, not between `vel` and `partition`.

### Build sequence per model

For each model (`PARENT_MODEL`, appropriate `GPU_QUEUE`):

```bash
# Step 1: rebuild grid + vel + standard partitions (sequential via PBS afterok)
GRID_HX=13 GRID_HY=13 PARTITION=1x2 JOB_CHAIN=grid-vel-partition bash scripts/driver.sh

# Step 2: remaining standard partitions (once grid+vel are done)
GRID_HX=13 GRID_HY=13 PARTITION=1x4 JOB_CHAIN=partition bash scripts/driver.sh
GRID_HX=13 GRID_HY=13 PARTITION=1x8 JOB_CHAIN=partition bash scripts/driver.sh

# Step 3: load-balanced partitions for Phase 1 (reuse rebuilt grid+vel from step 1)
# Note: +LB trick only tested at 1x2; larger partitions run baseline-only
GRID_HX=13 GRID_HY=13 PARTITION=1x2 LOAD_BALANCE=cell JOB_CHAIN=partition bash scripts/driver.sh
```

Steps 2 and 3 can only run after step 1 completes (grid and vel must be ready).

### Partition status before builds

| Model | Existing (halos likely 7) | To build with halos=13 |
|---|---|---|
| OM2-1 | `1x2`, `1x4`, `1x8`, `2x2` | rebuild all + add `1x2_LB`, `1x4_LB`, `1x8_LB` |
| OM2-025 | `1x2`, `1x2_LB`, `1x4`, `1x4_LB`, `1x8` | rebuild all + add `1x8_LB` |
| OM2-01 | unknown — verify first | likely rebuild/build all |

### TMPDIR for nsys on Gadi

nsys requires scratch-backed temp storage. Add to PBS nsys job scripts:

```bash
MYSCRATCH=/scratch/y99/bp3051
export TMPDIR=$MYSCRATCH/tmp && mkdir -p $TMPDIR
```

### nsys script change required

In `scripts/standard_runs/run_1year_benchmark.sh`:

```bash
# Serial: add --stats=true
nsys profile --stats=true --trace=nvtx,cuda \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    julia --project src/run_1year_benchmark.jl

# Distributed: replace per-rank wrapper with unified mpiexec command
nsys profile --stats=true --trace=nvtx,cuda,mpi \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    mpiexec $MPI_BIND_FLAGS -n $NGPUS julia --project src/run_1year_benchmark.jl
```

⚠️ Verify that `CUDA.@profile external=true` ranged capture works when nsys wraps `mpiexec`. If not, fall back to `--capture-range=none` with BENCHMARK_STEPS capping the run length.

---

## Sample Driver Commands

```bash
# --- OM2-1 ---

# baseline bench 1x1
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# baseline nsys 1x1
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# baseline bench + nsys 1x4
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x4 JOB_CHAIN=run1yrfast bash scripts/driver.sh
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x4 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# +GC bench + nsys 1x2
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 SYNC_GC_NSTEPS=5 JOB_CHAIN=run1yrfast bash scripts/driver.sh
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 SYNC_GC_NSTEPS=5 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# +TB bench + nsys 1x2
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 TBLOCKING=12 JOB_CHAIN=run1yrfast bash scripts/driver.sh
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 TBLOCKING=12 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# +LB bench + nsys 1x2
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2_LB JOB_CHAIN=run1yrfast bash scripts/driver.sh
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2_LB PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# --- OM2-025 ---

# baseline bench + nsys 1x2 V100 (V100 vs H200 hardware comparison)
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 JOB_CHAIN=run1yrfast bash scripts/driver.sh
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# +LB bench + nsys 1x2 H200
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuhopper GRID_HX=13 GRID_HY=13 PARTITION=1x2_LB JOB_CHAIN=run1yrfast bash scripts/driver.sh
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuhopper GRID_HX=13 GRID_HY=13 PARTITION=1x2_LB PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh
```
