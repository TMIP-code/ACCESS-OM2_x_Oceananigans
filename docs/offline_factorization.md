# Offline preconditioner factorization — hand-off plan

**Goal:** factorize the NK lump-and-spray preconditioner matrix **Q offline on a big-memory
CPU queue, save the factors to disk, then have the GPU NK job load + reuse them** — so the NK
job never pays the factorization memory peak. Prove the save→load→reuse cycle on a small OM2-1
toy first, pick a solver that supports it, and benchmark the solver options.

> Scope note (from the requester): **don't over-invest.** The first concrete deliverable is the
> OM2-1 `LUMP_AND_SPRAY=2x2` toy that demonstrates save/load/reuse works and a solver-comparison
> table. Everything past that is contingent on the toy succeeding.

---

## 1. Why (motivation + hard numbers)

The NK preconditioner factorization is the binding constraint on `LUMP_AND_SPRAY` for OM2-01.
Standalone benchmark (see `docs/nk_preconditioner_lump_and_spray_benchmark.md`), 48-CPU hugemem,
Pardiso `REAL_SYM`, `symdrop`:

| LUMP_AND_SPRAY | n_coarse | factorize | **peak RAM (factorize)** |
|---|---:|---:|---:|
| 5×5 | 14.7M | 20.6 min | 304 GiB |
| 4×4 | 22.8M | 44.1 min | 514 GiB |
| 3×4 | 30.2M | 68.6 min | 671 GiB |
| 4×3 | 30.2M | 73.1 min | 688 GiB |
| 3×3 | 40.0M | 118 min | 879 GiB |

Real NK runs on **one gpuhopper node = 4× H200 + 48 CPU + 1024 GiB shared host RAM** (Pardiso
runs on rank 0 as host MKL, `nprocs=48`; memory is one shared cgroup, not per-rank). Observed:

- **4×4** (`169827617`): fits (~700 GB total) but **does not converge in the 48 h max walltime**
  — too-coarse preconditioner ⇒ too many GMRES Jv's (each Jv ≈ 1 h GPU forward run).
- **3×3** (`169881991`): **OOM / SIGKILL**, 995 GB used — does not fit 1024 GiB.
- **3×4** (`169902272`): **SIGBUS**, 857 GB used — at the memory edge (retries pending:
  `170033028` 3×4, `170033029` 4×3).

So we're squeezed: coarser-than-4×4 won't converge in time; finer-than-~3×4 OOMs. **The key
observation that motivates this work: peak RAM occurs *during* factorization (fill-in), while
the resulting factors are smaller.** If we factorize on **megamem (3 TB)** and reuse the saved
factors, the GPU NK job only needs `factor + LUMP/SPRAY + GPU-worker host footprint` — which
should let us run **3×3 (and even 3×2 / 2×3 / 2×2)** preconditioners that are impossible to
factorize on the GPU node today, giving far fewer GMRES iterations and convergence within 48 h.

---

## 2. The recipe — how to load M, build the coarsened Q, and apply it

This is captured from the benchmark we just built. **Canonical references** (read these first):

- `src/benchmark_precond_solve.jl` — the standalone CPU build of Q (mirrors NK exactly).
- `src/solve_periodic_NK.jl:135-223` — the in-NK build **and** the preconditioner application
  (`TMPreconditioner` + `ldiv!`).
- `src/shared_utils/matrix.jl` — `process_sparse_matrix`, `compute_and_save_coarsening`.
- `src/shared_utils/config.jl` — `parse_lump_and_spray`, `require_env`, `load_project_config`.
- `src/shared_utils/grid.jl` — `load_tripolar_grid`, `compute_wet_mask`, `compute_volume`.

### 2a. Build Q (CPU-only)

```julia
include("shared_functions.jl")                       # pulls in all shared_utils/*
(; parentmodel, experiment_dir, outputdir) = load_project_config()
year = 365.25 * 86400

model_config = require_env("MODEL_CONFIG")
TM_SOURCE    = require_env("TM_SOURCE")              # "const"
MATRIX_PROCESSING = require_env("MATRIX_PROCESSING") # "symdrop" (Pardiso REAL_SYM needs structural symmetry)
ls = parse_lump_and_spray()                          # LUMP_AND_SPRAY env -> (di, dj, dk=1, on, tag, dir_suffix)

matrices_dir = joinpath(outputdir, "TM", model_config)
M_dir        = joinpath(matrices_dir, TM_SOURCE)     # M lives at <M_dir>/M.jld2

# grid → wet mask → cell volumes
grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
v1D = interior(compute_volume(grid))[idx]

# load RAW M, coarsen, scale, process  (EXACTLY as solve_periodic_NK.jl:173-197)
M = load(joinpath(M_dir, "M.jld2"), "M")
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; ls.di, ls.dj, ls.dk)
Mc = LUMP * M * SPRAY
stop_time = year                                     # ⚠ see "exactness" note below
Q  = copy(Mc); Q.nzval .*= stop_time
Q  = process_sparse_matrix(Q, MATRIX_PROCESSING)
```

`compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; ls.di, ls.dj, ls.dk, ls.on, ls.tag)`
already **persists `LUMP.jld2`, `SPRAY.jld2`, `Mc.jld2`** to `matrices_dir/<tag>/` (e.g.
`.../TM/<MC>/Q3x3/`). So the **only new artifact** offline factorization must produce is the
**factor of Q**. Reuse this function rather than re-coarsening.

### 2b. How the preconditioner is applied inside NK (what reuse must reproduce)

From `src/solve_periodic_NK.jl:141-153` (`ldiv!`), with `Plprob` = factorized `LinearProblem(Q, …)`:

```
ldiv!(Pl, x):   b = LUMP * x            # restrict fine → coarse
                solve!(Pl.prob)          # u = Q⁻¹ b   ← THE reused factorization
                x = SPRAY * Pl.prob.u - x # prolong coarse → fine, minus identity  (P = S Q⁻¹ L − I)
```

So **reuse = (load Q-factor) + (load LUMP, SPRAY) + (apply the three lines above)**. To swap
offline factors into NK, replace the `init(LinearProblem(Q,…), linear_solver)` at
`solve_periodic_NK.jl:221-223` with a loader that returns an object whose `solve!`/`\` uses the
saved factor; `LUMP`/`SPRAY` are loaded from `matrices_dir/<tag>/`.

### 2c. ⚠ Exactness requirement

For a saved factor to be a valid preconditioner, **Q must be built bit-for-bit identically to
what `solve_periodic_NK.jl` builds**: same `M`, same `(di,dj,dk)`, same `MATRIX_PROCESSING`, and
the **same `stop_time`**. NK uses `stop_time = n_months * prescribed_Δt` (`src/setup_model.jl:433`,
= 1 yr for `N_MONTHS=12`), which the benchmark approximates as `year`. **Recommended refactor:**
extract the Q-construction (NK lines 173-197) into one shared helper (e.g. in
`shared_utils/matrix.jl`) called by **both** `solve_periodic_NK.jl` and the offline factorizer, so
they can never drift. The `stop_time` scalar is uniform on the nonzeros (cosmetic for *cost*) but
it changes the *numeric* factor, so it must match.

---

## 3. The technical crux — which solver can save/load factors

We currently go through LinearSolve's `MKLPardisoIterate` wrapper, which likely does **not**
expose the flags needed to dump/reload factors. Options, with what's known vs. to-verify:

> **CORRECTION (verified during Phase 1):** LinearSolve's `MUMPSFactorization`
> (PR #996) only does **in-process** factorize+solve — it exposes an `ooc` flag but
> **never calls JOB=7/JOB=8**, so it cannot save a factor in one job and reload it in
> another. **True cross-job save/restore lives only in `MUMPS.jl` directly**
> (`set_save_dir!`, `set_save_prefix!`, `set_job!(SAVE_DATA=7)` / `RESTORE_DATA=8`,
> `invoke_mumps!`), bypassing LinearSolve. That is what we implemented and verified.

| Solver | Save/load factors? | Notes |
|---|---|---|
| **MUMPS.jl (direct)** | ✅ **native** `JOB=7` (save) / `JOB=8` (restore) to disk | **Winner.** Via `MUMPS.jl` directly (NOT LinearSolve's wrapper). Parallel; scales to OM2-01. Needs `MUMPS.jl` + `MPI.Init()` (env_defaults.sh provides the MPI runtime in the job). |
| **PureUMFPACK.jl** | ✅ direct | `PureLU` is plain Julia data → `jldsave(F)` directly; reload + built-in `solve(F,b)`. Use `method=:multifrontal` (BLAS-3; the default `:gplu` is too slow and overran walltime). Unregistered (pinned rev). Serial → won't scale to 3×3. |
| **UMFPACK** (stdlib) | ⚠ indirect | Extract `F.L, F.U, F.p, F.q, F.Rs` (plain Julia objects) → JLD2 → reload → apply manually. Relation: `F.L*F.U == (F.Rs .* A)[F.p, F.q]`. Solve: `b′=F.Rs.*b; y=F.L\(b′[F.p]); z=F.U\y; x[F.q]=z`. Zero new deps; trusted reference. Serial → won't scale to 3×3. |
| **ParU** (`ParU_jll`) | ❓ | Parallel. In-memory C handle — serialization likely unsupported; not pursued. |
| **Pardiso (MKL)** | ❌ portable disk dump; ✅ **OOC** | The `pt` handle is in-process pointers, not portable. MKL Pardiso has **out-of-core mode (`IPARM[60]`)** that streams the factor to disk during factorization. OOC ≠ offline reuse: it leaves the factor *on disk* → disk-bound per-apply (bad for the apply-heavy NK preconditioner). Kept only as a cheap probe ("does 3×3 even fit in-job?"). |

## 3b. Phase 1 results — toy proven (OM2-1, `LUMP_AND_SPRAY=2x2`)

The save→load→reuse cycle works for **all three** solvers. Q built by the shared
`build_precond_Q` helper (710156×710156 coarse, nnz 4.8M). Each phase a separate
process (PBS via `JOB_CHAIN=TMofflinefact`); correctness gate `rel-err < 1e-8` vs an
in-process `lu(Q)\b` reference.

| solver | factorize | factor peak RAM | factor on disk | reuse load+apply | reuse RAM | rel-err | gate |
|---|--:|--:|--:|--:|--:|--:|:--:|
| **MUMPS** (JOB 7/8) | **31 s** | **14.0 GB** | **3.28 GB** | **4.0 s** | **4.25 GB** | 3.6e-14 | ✅ |
| PureUMFPACK (multifrontal) | 94 s | 25.5 GB | 11.71 GB | 9.0 s | 12.5 GB | 6.9e-14 | ✅ |
| UMFPACK (stdlib) | 152 s | 31.8 GB | 11.93 GB | 8.7 s | 12.7 GB | 1.5e-15 | ✅ |

**Conclusion: MUMPS-direct wins on every axis** (6× faster factorize, 3.6× smaller
factor, 3× less reuse RAM) and is the only one that scales to OM2-01 → **the Phase-2
production solver.** Reuse RAM ≈ factor-on-disk size (≪ factorization peak), confirming
the core premise: reload the small factor into RAM, never pay the fill-in peak on the
GPU node.

Code: `src/benchmark_offline_factor.jl` (+ `scripts/benchmarks/benchmark_offline_factor.sh`,
driver step `TMofflinefact`); shared Q build + UMFPACK helpers in `src/shared_utils/matrix.jl`.

---

## 4. Phase 1 — OM2-1 toy (`LUMP_AND_SPRAY=2x2`)

OM2-1 (1°) is tiny: M is ~10⁵–10⁶ wet cells, factorizes in seconds and fits express/normal — fast
iteration on the save/load mechanics. Matrices already exist on disk, e.g.
`outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/TM/<MC>/const/M.jld2` (pick the
default OM2-1 config that `env_defaults.sh` resolves; LUMP/Mc are already present for several
configs there).

**Write a script** `src/benchmark_offline_factor.jl` (model it on `src/benchmark_precond_solve.jl`)
that, for each solver in `{UMFPACK, ParU, MUMPS?}`:

1. Build Q via §2a (`LUMP_AND_SPRAY=2x2`).
2. **Factorize** — record time + `Sys.maxrss`.
3. **Save** the factor to `matrices_dir/<tag>/factor_<solver>.{jld2|native}` — record file size + save time.
4. In a **fresh Julia process**, **load** the factor + `LUMP`/`SPRAY` — record load time + RAM.
5. **Apply**: solve `Q u = b` for a test `b = LUMP*randn(Nidx)`; compute the full preconditioner
   action `SPRAY*u − x` and compare to an in-process reference solve. **Assert** relative error
   `< 1e-8` (this is the correctness gate).
6. Record solve time.

**Deliverable — solver comparison table:**

| solver | factorize time | factorize peak RAM | factor file size | load time | solve time | reuse correct? |
|---|---|---|---|---|---|---|

Run via a small PBS wrapper (`scripts/benchmarks/benchmark_offline_factor.sh`, normal/express,
CPU). No driver step needed yet for the toy.

---

## 5. Phase 2 — scale to OM2-01 (only if the toy works)

1. **Offline factorize on megamem (3 TB)** for the coarsenings impossible on the GPU node:
   `3x3` first, then `3x2`/`2x3`/`2x2` (better preconditioners → fewer GMRES iters). Use the
   winning solver from Phase 1. Save factor next to `LUMP/SPRAY/Mc` in `TM/<MC>/<tag>/`.
2. **Estimate NK-node fit:** `factor size (loaded) + LUMP + SPRAY + GPU-worker host footprint`
   must stay under 1024 GiB. The GPU workers' host footprint was ~150–340 GB in the 4×4 run; the
   loaded factor (no fill-in peak) should be far smaller than the 879 GiB *factorization* peak.
3. **Wire into the pipeline:** new driver step (e.g. `TMprecfact`, megamem CPU) that produces the
   factor; `solve_periodic_NK.jl` gains a `PRECOND_FACTOR=<path>` mode that loads the factor +
   LUMP/SPRAY instead of factorizing. Keep the existing in-job factorization as the default.

---

## 6. Related cheap win (independent of offline reuse)

In `solve_periodic_NK.jl`, **free `M` after Q is built** (`M = nothing; GC.gc()` before
factorization). The Jv's use the GPU forward model, not `M`, so the ~tens-of-GB sparse `M` on
rank 0 is dead weight during factorization. This alone lowers the NK-node peak and might let
3×4/4×3 run comfortably even without offline factors. Worth doing regardless.

---

## 7. Open questions for the implementing agent

- Is `MUMPSFactorization` available in our pinned `LinearSolve`? If not, cost of bumping +
  adding `MUMPS_jll`.
- Does Pardiso **OOC** (`IPARM[60]`, via `Pardiso.jl` directly) reduce the 3×3 peak enough to fit
  the GPU node **without** offline reuse? (Could make this whole effort unnecessary — check early.)
- Confirm the UMFPACK factor-reconstruction apply formula against an in-process `F\b` (the toy's
  correctness gate).
- Does `ParU_jll`/any ParU Julia wrapper expose a factor export, or is it in-memory only?

## 8. Conventions to follow

- Reuse existing helpers (don't re-implement coarsening / matrix processing / grid loading).
- Track wall time + `Sys.maxrss` (process) **and** PBS `resources_used.mem` (authoritative peak),
  as `benchmark_precond_solve.jl` does.
- Submit via `scripts/driver.sh` for commit tracking; for the toy a standalone PBS wrapper is fine.
- Save artifacts beside the existing `LUMP/SPRAY/Mc` in `TM/<MODEL_CONFIG>/<tag>/`.
