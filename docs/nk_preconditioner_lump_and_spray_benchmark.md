# NK preconditioner factorize+solve benchmark — OM2-01 default config

Benchmark of the Newton–Krylov (NK) **preconditioner** linear solve (the coarsened
lump-and-spray Pardiso factorization) as a function of the `LUMP_AND_SPRAY` coarsening
factor, for **ACCESS-OM2-01 in its default configuration**. Drives the choice of
`LUMP_AND_SPRAY` for OM2-01: finer coarsening → better preconditioner but larger coarse
matrix `Mc` and much larger Pardiso fill-in (the memory/time bottleneck of the NK solve).

## What is measured

The benchmark (`src/benchmark_precond_solve.jl`, PBS wrapper
`scripts/benchmarks/benchmark_precond_solve.sh`, driver step `TMprecbench`) **reproduces the
exact preconditioner matrix Q built by `src/solve_periodic_NK.jl`**:

```
coarsen the RAW M:   LUMP, SPRAY, v_c = lump_and_spray(wet3D, v1D, M; di, dj, dk=1)
                     Mc = LUMP * M * SPRAY
scale:               Q  = stop_time * Mc                 # stop_time = 1 yr; scaling is cosmetic for cost/memory
process:             Q  = process_sparse_matrix(Q, MATRIX_PROCESSING)   # symdrop
factorize + solve:   Pardiso REAL_SYM (mtype=1), nprocs=48
```

It times `init` (symbolic), the **1st solve** (= numeric factorization + solve), and a **2nd
solve** (factorization reused — what each GMRES preconditioner apply costs), and records peak
memory. One CPU job per coarsening factor, so PBS `resources_used.mem` cleanly attributes the
peak. It does **not** run a Newton solve.

> Note: this is the *preconditioner* solve. It is closely related to but not identical to the
> steady-state age `TMsolve` (`src/solve_matrix_age.jl`), which processes M *before* coarsening
> and solves `Mc \ rhs` without the `stop_time` scaling. We deliberately match the NK path.

## The configuration being benchmarked

**OM2-01 default config** (resolved by `scripts/env_defaults.sh` + `model_configs/ACCESS-OM2-01.sh`):

| Setting | Value |
|---|---|
| `PARENT_MODEL` | `ACCESS-OM2-01` |
| `EXPERIMENT` | `01deg_jra55v140_iaf_cycle4` |
| `TIME_WINDOW` | `1968-1977` |
| `VELOCITY_SOURCE` | `cgridtransports` |
| `W_FORMULATION` / `PRESCRIBED_W_SOURCE` | `wprescribed` / `parent` (→ tag `wparent`) |
| `ADVECTION_SCHEME` | `centered2` |
| `TIMESTEPPER` | `AB2` |
| `TIMESTEP_MULT` | `1` (Δt = 400 s; no `DTx` suffix) |
| `KAPPA_H`, `KAPPA_V_ML`, `KAPPA_V_BG` | `30`, `25e-3`, `75e-7` |
| `MONTHLY_KAPPAV` / `IMPLICIT_KAPPAV` | `yes` / `yes` (→ tag `mkappaV`) |
| `GM_REDI` | `no` (OM2-01 is eddy-resolving) |
| `LOAD_BALANCE` / `PARTITION` | `surface` (→ tag `LBS`) / `1x4` |
| `TM_SOURCE` | `const` |
| `LINEAR_SOLVER` / `MATRIX_PROCESSING` | `Pardiso` / `symdrop` |

**Resolved `MODEL_CONFIG`:**
```
cgridtransports_wparent_centered2_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS
```

**Transport matrix M** (built by job `169827616`, commit `8a35fc4`):
```
outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/<MODEL_CONFIG>/const/M.jld2
```
- size **351,532,308 × 351,532,308**, nnz **2,439,571,000** (~39 GB on disk)

## Results

48-CPU **hugemem** node, Pardiso `REAL_SYM`, `MATRIX_PROCESSING=symdrop`. Memory columns:
`maxrss` = Julia process peak (logged); **`job peak` = PBS `resources_used.mem`** (authoritative).

| LUMP_AND_SPRAY | di×dj | n_coarse | nnz(Q) | **factorize** (1st solve) | solve (reuse) | factor alloc | maxrss | **job peak (PBS)** | job |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **5x5** | 5×5 | 14,722,237 | 100,946,837 | 1234.4 s (20.6 min) | 18.9 s | 0.92 GB | 301.6 GiB | **304.1 GiB** | `169835787` |
| **4x4** | 4×4 | 22,751,548 | 156,472,744 | 2645.1 s (44.1 min) | 39.5 s | 1.42 GB | 472.3 GiB | **514.4 GiB** | `169835788` |
| **3x4** | 3×4 | 30,173,079 | 207,814,687 | 4115.6 s (68.6 min) | 72.9 s | 1.89 GB | 628.8 GiB | **671.3 GiB** | `169881992` |
| **4x3** | 4×3 | 30,166,648 | 207,783,382 | 4383.8 s (73.1 min) | 56.7 s | 1.89 GB | 647.2 GiB | **687.9 GiB** | `169881993` |
| **3x3** | 3×3 | 40,000,420 | 275,923,076 | 7078.8 s (118.0 min) | 95.5 s | 2.50 GB | 876.8 GiB | **878.6 GiB** | `169835789` |

(`di` = i/zonal coarsening, `dj` = j/meridional. All jobs Exit 0.)

### Takeaways

- **Steeply superlinear.** Factorize time ≈ doubles per step finer (1234 → 2645 → 7079 s);
  memory grows ~1.6×/step (304 → 514 → 879 GiB) while `n_coarse` grows only ~1.5×/step —
  classic sparse-LU fill-in blow-up.
- **di vs dj asymmetry.** 3×4 and 4×3 have near-identical size (n_coarse ≈ 30.17M, product 12)
  but **3×4 (coarsen less zonally) is ~6% cheaper** in factorize time (68.6 vs 73.1 min) and
  memory (671 vs 688 GiB). The tripolar grid is anisotropic; coarsening more in j fills in less.
- **Per-iteration cost is cheap.** The reused-factorization solve (what each GMRES apply costs)
  is 19–96 s; the trade-off is dominated by the one-time factorize + the memory ceiling, not the
  GMRES apply.
- **Fit on the NK node.** The NK preconditioner factorizes on **rank 0 (host MKL Pardiso,
  `nprocs=48`)** of a single gpuhopper node = 4× H200 + 48 CPU + **1024 GiB shared CPU RAM**
  (one cgroup; memory is *not* partitioned per rank). All factors above fit under 1024 GiB:
  5×5/4×4 with wide margin; **3×3 at ~879 GiB is tight** once the three GPU workers' host-side
  footprint is added (the running NK job sits at ~700 GiB total for 4×4). The GPU workers keep
  their big arrays in HBM, leaving most of the 1024 GiB for rank 0's factorization.

## How to (re)submit

All via `scripts/driver.sh` (clean git tree required; bakes `GIT_COMMIT` into every job and
records `scripts/runs/submissions.tsv` + a manifest TOML).

**Run/extend the benchmark sweep** (`LAS_VALUES` overrides the swept set; default `"5x5 4x4 3x3"`):
```bash
PARENT_MODEL=ACCESS-OM2-01 LAS_VALUES="5x5 4x4 3x3" \
  JOB_CHAIN=TMprecbench bash scripts/driver.sh
# add new factors, e.g.:
PARENT_MODEL=ACCESS-OM2-01 LAS_VALUES="3x4 4x3 2x2" \
  JOB_CHAIN=TMprecbench bash scripts/driver.sh
```
If M is not yet built, chain onto a running/finished `TMbuild` job:
```bash
TMBUILD_JOB=<jobid>.gadi-pbs PARENT_MODEL=ACCESS-OM2-01 LAS_VALUES="5x5 4x4 3x3" \
  JOB_CHAIN=TMprecbench bash scripts/driver.sh
```
Each benchmark job: **hugemem, 48 CPU, 1470 GB, walltime 06:00:00** (override with
`WALLTIME_PRECBENCH`). `DRY_RUN=yes` previews the qsub commands.

**Run a full NK solve at a given coarsening** (e.g. to test whether 3×3 survives on the node):
```bash
PARENT_MODEL=ACCESS-OM2-01 LUMP_AND_SPRAY=3x3 JOB_CHAIN=NK bash scripts/driver.sh
```
NK job: **gpuhopper, 1×4 (4 H200, 48 CPU, 1024 GiB), walltime 48:00:00**. (Default
`LUMP_AND_SPRAY` for OM2-01 is `4x4`; set it explicitly to override.)

## Where results land

```
outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/<MODEL_CONFIG>/const/benchmarks/
  precond_solve_Q{di}x{dj}_Pardiso_symdrop.tsv    # one header+row per coarsening (race-free)
  precond_solve_Q{di}x{dj}_Pardiso_symdrop.jld2   # same, structured (+ git_commit, model_config)
logs/julia/ACCESS-OM2-01/.../TM/benchmarks/        # full run logs
```
Combine the per-config TSVs:
```bash
column -t -s$'\t' <(cat outputs/.../const/benchmarks/precond_solve_*.tsv | awk 'NR==1||!/^tag\t/')
```
Authoritative peak memory per job (while still in PBS history, ~7 days):
```bash
qstat -fx <jobid>.gadi-pbs | grep resources_used.mem    # value in KiB
```
After jobs finish, reconcile the submissions index (fills exit_code, queue, mem/walltime used):
```bash
bash scripts/runs/reconcile_submissions.sh
```
