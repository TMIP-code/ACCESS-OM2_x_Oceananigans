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
| **+LB** | `LOAD_BALANCE` | `cell` (cell-balanced y-partition; output dir gets `_LB` suffix automatically) |

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
| 3 | 1x2_LB | 167878729 | ✗ | 4m 25s | OOM-killed during `w` file; only u/v written. Default 8GB on express was insufficient. |
| 3 | 1x2_LB (rebuild) | 167929025 | ✓ | 4m 42s | After fix: PARTITION_MEM_PER_RANK=12 → 24GB/6cpu (commit 32ff1db). All 8 files written. |

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
| 1x2 | +LB bench (retry) | 167931014 | ✓ | 10m 35s | After 1x2_LB rebuild; uses `LOAD_BALANCE=cell` |
| 1x2 | +LB nsys (retry) | 167931019 | ✓ | 10m 41s | After 1x2_LB rebuild; uses `LOAD_BALANCE=cell` |
| 1x4 | baseline bench | 167891869 | ✓ | 9m 12s | |
| 1x4 | baseline nsys | 167891870 | ✓ | 10m 0s | |
| 1x8 | baseline bench | 167891401 | ✓ | 9m 23s | |
| 1x8 | baseline nsys | 167891402 | ✓ | 10m 29s | |

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

| Step | Partition | Job ID | Status | Elapsed | Notes |
|------|-----------|--------|--------|---------|-------|
| 1 | grid | 167933157 | ✓ | 12m 12s | |
| 1 | vel | 167933158 | ✓ | 18m 25s | **44/47GB cap-hit** (default at time of submission); fixed in commit 18ac934 → VEL_MEM=96GB |
| 1 | 1x2 | 167933159 | ✓ | 7m 46s | hugemem 192GB (peak 109GB) |
| 2 | 1x4 | 167940054 | ✓ | 7m 55s | hugemem, peak 195GB / 192GB (cap-hit) |
| 2 | 1x8 | 167940055 | ✓ | 8m 10s | hugemem, peak 268GB / 256GB (cap-hit) |
| 3 | 1x2_LB | 167940056 | ✓ | 6m 37s | hugemem 192GB (peak 93GB) |

### Simulation Results (fill in as jobs complete)

| Partition | GPU | Config | Job ID | Wall time | Notes |
|-----------|-----|--------|--------|-----------|-------|
| 1x2 | V100 | baseline bench | 167950637 | ✓ 25m 27s | |
| 1x2 | V100 | baseline nsys | 167950638 | ✓ 11m 59s | |
| 1x2 | V100 | +GC bench | 167950639 | ✓ 25m 11s | |
| 1x2 | V100 | +GC nsys | 167950640 | ✓ 12m 14s | |
| 1x2 | V100 | +TB bench | 167950641 | ✓ 24m 1s | |
| 1x2 | V100 | +TB nsys | 167950642 | ✓ 11m 33s | |
| 1x2 | V100 | +LB bench | 167950643 | ✓ 25m 12s | |
| 1x2 | V100 | +LB nsys | 167950645 | ✓ 12m 22s | |
| 1x2 | H200 | baseline bench | 167950650 | ✓ 13m 41s | |
| 1x2 | H200 | baseline nsys | 167950651 | ✓ 10m 14s | |
| 1x2 | H200 | +GC bench | 167950652 | ✓ 14m 7s | |
| 1x2 | H200 | +GC nsys | 167950653 | ✓ 10m 14s | |
| 1x2 | H200 | +TB bench | 167950654 | ✓ 12m 54s | |
| 1x2 | H200 | +TB nsys | 167950655 | ✓ 10m 2s | |
| 1x2 | H200 | +LB bench | 167950656 | ✓ 14m 14s | |
| 1x2 | H200 | +LB nsys | 167950657 | ✓ 10m 41s | |
| 1x4 | H200 | baseline bench | 167950658 | ✓ 11m 4s | |
| 1x4 | H200 | baseline nsys | 167950659 | ✓ 8m 55s | |
| 1x8 | H200 | baseline bench | 167950660 | ✓ 12m 4s | |
| 1x8 | H200 | baseline nsys | 167950661 | ✓ 19m 3s | |

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
| 0 | grid (initial, redundant) | 167933708 | ✓ | 3m 30s | submitted before realizing prep was needed |
| 0 | vel (initial) | 167933709 | ✗ | 5m 59s | exit 1 — missing monthly NCs (no prep step) |
| 0 | partition (initial) | 167933710 | (cancelled) | — | would have OOM'd at express 8GB default |
| 1 | prep | 167940352 | ✓ | 3h 44m | megamem 2TB |
| 1 | grid | 167940353 | ✓ | 3m 39s | afterok prep |
| 1 | vel | 167940354 | ✓ | 1h 0m | afterok prep+grid (hugemem 512GB) |
| 1 | 1x2 | 167940409 | ✓ | 43m | afterok grid+vel (megamem 1000GB / 15cpu) |
| 2 | 1x4 | 167961038 | ✓ | 41m 40s | megamem 1400GB / 21cpu, peak 1.34TB (96% — tight) |
| 2 | 1x8 | 167961039 | ✓ | 1h 21m | megamem 2800GB / 43cpu, peak 1.79TB |
| 3 | 1x2_LB | 167961040 | ✓ | 36m 39s | megamem 1000GB / 15cpu, peak 645GB |
| 4 | grid (halos=19) | 167994958 | ✓ | 4m 53s | rebuild for K=18; overwrites halos=13 grid.jld2 |
| 4 | vel (halos=19) | 167994959 | R | 56m+ | depends on new grid |
| 4 | 1x2 (halos=19) | 167994960 | H | — | depends on grid+vel; will support TBLOCKING=18 |

### Simulation Results (fill in as jobs complete)

| Partition | Config | Job ID | Wall time | Notes |
|-----------|--------|--------|-----------|-------|
| 1x2 | baseline bench | 167976668 | ✓ 3h 22m | halos=13 |
| 1x2 | baseline nsys | 167976669 | ✓ 19m 0s | halos=13 |
| 1x2 | +GC bench | 167976670 | ✓ 3h 24m | halos=13 |
| 1x2 | +GC nsys | 167976671 | ✓ 19m 5s | halos=13 |
| 1x2 | +TB bench | 167976672 | ✗ 17m 55s | failed: K=12 doesn't divide 78894 steps; redo with K=18 (halos=19) |
| 1x2 | +TB nsys | 167976673 | (superseded) 20m 46s | K=12 succeeded but redoing with K=18 for consistency |
| 1x2 | +LB bench | 167976674 | ✓ 3h 18m | halos=13 |
| 1x2 | +LB nsys | 167976675 | ✓ 22m 28s | halos=13 |
| 1x4 | baseline bench | 167976676 | ✓ 1h 53m | halos=13 |
| 1x4 | baseline nsys | 167976677 | ✓ 17m 20s | halos=13 |
| 1x8 | baseline bench | 167976678 | ✓ 2h 5m | halos=13 |
| 1x8 | baseline nsys | 167976679 | ✓ 31m 28s | halos=13 |
| 1x2 | +TB bench (retry) | TBD | Q | K=18, halos=19; pending grid rebuild |
| 1x2 | +TB nsys (retry) | TBD | Q | K=18, BENCHMARK_STEPS=360 (20×18, 20 MPI passes), halos=19 |

---

## Total: 46 simulation jobs

---

## nsys Profile Locations + scp Globs

Profiles are saved per rank under `logs/julia/{MODEL}/{EXPERIMENT}/{TW}/standardrun/`:
- Serial (1x1): `{MODEL_CONFIG}_1yearfast_{JOB_ID}.gadi-pbs_profile_syncGCyes_N5.nsys-rep`
- Distributed: `{MODEL_CONFIG}_1yearfast_{JOB_ID}.gadi-pbs_profile_syncGCyes_N5_rank{0..N-1}.nsys-rep`

`MODEL_CONFIG` defaults to `cgridtransports_wdiagnosed_centered2_AB2` and gets:
- `_TB12` / `_TB18` suffix when `TBLOCKING` is set
- `_LB` suffix when `LOAD_BALANCE=cell` is set

### Phase 1 (OM2-1) — V100, 7 nsys runs

Base: `logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/`

| Job ID | Config | MODEL_CONFIG suffix | Rank files |
|--------|--------|---------------------|------------|
| 167891388 | 1x1 baseline | (none) | (serial, no rank) |
| 167891391 | 1x2 baseline | (none) | rank0, rank1 |
| 167891393 | 1x2 +GC | (none) | rank0, rank1 |
| 167891398 | 1x2 +TB | `_TB12` | rank0, rank1 |
| 167931019 | 1x2 +LB | `_LB` | rank0, rank1 |
| 167891870 | 1x4 baseline | (none) | rank0..3 |
| 167891402 | 1x8 baseline | (none) | rank0..7 |

### Phase 2 (OM2-025) — V100 + H200, 10 nsys runs

Base: `logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/`

| Job ID | Config | MODEL_CONFIG suffix | Rank files |
|--------|--------|---------------------|------------|
| 167950638 | V100 1x2 baseline | (none) | rank0, rank1 |
| 167950640 | V100 1x2 +GC | (none) | rank0, rank1 |
| 167950642 | V100 1x2 +TB | `_TB12` | rank0, rank1 |
| 167950645 | V100 1x2 +LB | `_LB` | rank0, rank1 |
| 167950651 | H200 1x2 baseline | (none) | rank0, rank1 |
| 167950653 | H200 1x2 +GC | (none) | rank0, rank1 |
| 167950655 | H200 1x2 +TB | `_TB12` | rank0, rank1 |
| 167950657 | H200 1x2 +LB | `_LB` | rank0, rank1 |
| 167950659 | H200 1x4 baseline | (none) | rank0..3 |
| 167950661 | H200 1x8 baseline | (none) | rank0..7 |

### Phase 3 (OM2-01) — H200, 6 nsys runs (+ 1 pending K=18 retry)

Base: `logs/julia/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/`

| Job ID | Config | MODEL_CONFIG suffix | Rank files |
|--------|--------|---------------------|------------|
| 167976669 | 1x2 baseline | (none) | rank0, rank1 |
| 167976671 | 1x2 +GC | (none) | rank0, rank1 |
| 167976673 | 1x2 +TB (K=12, superseded) | `_TB12` | rank0, rank1 |
| 167976675 | 1x2 +LB | `_LB` | rank0, rank1 |
| 167976677 | 1x4 baseline | (none) | rank0..3 |
| 167976679 | 1x8 baseline | (none) | rank0..7 |
| TBD | 1x2 +TB (K=18 retry) | `_TB18` | rank0, rank1 (halos=19) |

### scp commands (run from local machine)

Set the remote root once:

```bash
GADI_ROOT="gadi:/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/logs/julia"
```

**Pull everything (all phases):**

```bash
mkdir -p nsys_profiles && cd nsys_profiles
scp -r "$GADI_ROOT/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/*_profile_*.nsys-rep" om2-1/
scp -r "$GADI_ROOT/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/*_profile_*.nsys-rep" om2-025/
scp -r "$GADI_ROOT/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/*_profile_*.nsys-rep" om2-01/
```

**Pull specific job (e.g. one row of the matrix):**

```bash
# Pattern: any rank for a given job ID
JOB=167891398  # OM2-1 1x2 +TB nsys
scp "$GADI_ROOT/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/*_${JOB}.gadi-pbs_profile_*.nsys-rep" .
```

**Pull just the V100 vs H200 1x2 baseline (Phase 2 hardware comparison):**

```bash
B="$GADI_ROOT/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun"
scp "$B/*_167950638.gadi-pbs_profile_*.nsys-rep" v100/    # V100 baseline
scp "$B/*_167950651.gadi-pbs_profile_*.nsys-rep" h200/    # H200 baseline
```

**Pull all +TB profiles across phases (cross-resolution comparison):**

```bash
scp "$GADI_ROOT/ACCESS-OM2-1/*/1968-1977/standardrun/*_TB12_*_167891398.gadi-pbs_profile_*.nsys-rep" .
scp "$GADI_ROOT/ACCESS-OM2-025/*/1968-1977/standardrun/*_TB12_*_{167950642,167950655}.gadi-pbs_profile_*.nsys-rep" .
scp "$GADI_ROOT/ACCESS-OM2-01/*/1968-1977/standardrun/*_TB18_*_profile_*.nsys-rep" .   # job ID TBD
```

Total size estimate: each rank file ≈ 3 MB → 23 profile job IDs × ~4 ranks avg ≈ **~300 MB total** for everything.

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
# (Use LOAD_BALANCE=cell — PARTITION=1x2_LB fails in arithmetic parsing)
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 LOAD_BALANCE=cell JOB_CHAIN=run1yrfast bash scripts/driver.sh
GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 LOAD_BALANCE=cell PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# --- OM2-025 ---

# baseline bench + nsys 1x2 V100 (V100 vs H200 hardware comparison)
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 JOB_CHAIN=run1yrfast bash scripts/driver.sh
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuvolta GRID_HX=13 GRID_HY=13 PARTITION=1x2 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh

# +LB bench + nsys 1x2 H200
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuhopper GRID_HX=13 GRID_HY=13 PARTITION=1x2 LOAD_BALANCE=cell JOB_CHAIN=run1yrfast bash scripts/driver.sh
PARENT_MODEL=ACCESS-OM2-025 GPU_QUEUE=gpuhopper GRID_HX=13 GRID_HY=13 PARTITION=1x2 LOAD_BALANCE=cell PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast bash scripts/driver.sh
```
