# NK solver at OM2-01: settings, code changes, coarsening sweep

## Context

OM2-01 (0.1°) is ~9.4× the cells of OM2-025 (6.25× horizontal at 2.5× linear, 1.5× vertical
at 75 vs 50 levels). The current direct memory/time anchor is **OM2-01 itself**:

- **OM2-01 TM age solve, 2×2 coarsening** (one-shot run): Pardiso factorisation needed
  **2.34 TB** of memory and took **~10 hours**.

That rules 2×2 out for NK at OM2-01: 2.34 TB exceeds the 1024 GB node, and a 10 h
factorisation would dominate the 48 h walltime even if it fit. The sweep has to start at
a coarsening that brings both the memory and the factorisation time down by ~10×.

Naive scaling from the 2×2 anchor (matrix dim ∝ 1/(di·dj); sparse-LU memory roughly N to
N^{4/3}, time roughly N^{3/2} to N^2 for 3D problems with nested dissection):

| Coarsening | dim ratio vs 2×2 | est. memory | est. factorisation time |
|---|---|---|---|
| 5×5 | 0.16  | 150–400 GB  | 15–40 min |
| 4×4 | 0.25  | 370–600 GB  | 40 min – 1.5 h |
| 3×3 | 0.44  | 770 GB – 1 TB | ~2–3 h |
| 6×6 | 0.11  | 130–260 GB  | 7–22 min |
| 7×7 | 0.08  | 90–190 GB   | ~5–15 min |

These are rough — Pardiso ordering, fill-in, and the 75 vertical levels (uncoarsened in z)
make either end of each range plausible.

The current code hardcodes `(di,dj,dk) = (2,2,1)` and treats `LUMP_AND_SPRAY` as a yes/no
flag — we need to generalise to `AxB`, sweep factors centred at 5×5, and tag outputs so
different factors don't overwrite each other. Goal: find the largest coarsening that
still converges within the Newton budget — walking finer (4×4, 3×3) if 5×5's preconditioner
is too weak, or coarser (6×6, 7×7, …) if memory or factorisation time blow up.

**Status**: 5×5 NK test job submitted by the user; results pending.

## Run config (env vars)

```
PARENT_MODEL          = ACCESS-OM2-01
EXPERIMENT            = 01deg_jra55v140_iaf_cycle4
TIME_WINDOW           = 1968-1977
VELOCITY_SOURCE       = cgridtransports
W_FORMULATION         = wprescribed
PRESCRIBED_W_SOURCE   = parent
TIMESTEPPER           = AB2
TIMESTEP_MULT         = 2
MONTHLY_KAPPAV        = yes
PARTITION             = 1x4
GPU_QUEUE             = gpuhopper
LOAD_BALANCE          = <TBD>      # one of: no | surface | cell | mix | minmax
LUMP_AND_SPRAY        = <AxB>      # only no | AxB (no `yes`); sweep 5x5 / 4x4 / 3x3
MATRIX_PROCESSING     = symdrop
LINEAR_SOLVER         = Pardiso
TM_SOURCE             = const
TRACE_SOLVER_HISTORY  = yes
INITIAL_AGE           = 0          # `latest` on a manual restart
```

`MODEL_CONFIG` resolves to `cgridtransports_wparent_centered2_AB2_mkappaV_DTx2`
(plus the `_LB?` suffix once `LOAD_BALANCE` is picked). `LUMP_AND_SPRAY` does
**not** enter `MODEL_CONFIG`; it tags the NK subdirectory and the per-file
`lumpspray_tag` (see below).

## Resources

- `ngpus=4`, `ncpus=48` (= 4×12 CPUs/GPU), `mem=1024GB` (= 4×256 GB per rank).
- `walltime=48:00:00` (Gadi max).
- One job per coarsening factor; **no `afterok`/`afterany` chains** — restart
  manually after each 48 h run via `INITIAL_AGE=latest`.

Per the Context-section table (anchored on OM2-01+2×2 = 2.34 TB / 10 h), the
relevant rank-0 envelopes are:

| Coarsening | est. memory | est. factorise time | Verdict on a 1024 GB node |
|---|---|---|---|
| 5×5 | 150–400 GB | 15–40 min | Should fit on a 256 GB rank cleanly; may spill onto a second socket at the high end. |
| 4×4 | 370–600 GB | 40 min – 1.5 h | Needs rank 0 to use the node's full pool (binding levers below). |
| 3×3 | 770 GB – 1 TB | 2–3 h | Tight on the node; viable only with full-pool binding and likely no other large allocations. |
| 6×6 / 7×7 | 90–260 GB | 5–22 min | Comfortable; fallback if 5×5 OOMs or factorises slowly. |
| 2×2 | 2.34 TB | ~10 h | **Infeasible** — exceeds node memory; walltime alone unworkable. |

Two risks of comparable weight here:

- **Preconditioner quality**: 5×5 (or coarser) may not converge in the Newton
  budget — GMRES makes little progress per step.
- **Factorisation cost**: even when memory fits, factorisation is a one-shot
  setup cost; a 3-hour factorisation eats meaningful walltime, and an
  unexpectedly slow one (e.g. ordering pathology) could dominate.

### Rank-0 CPU/memory binding (lever to relax the rank-0 pinch)

Today the launcher uses `mpiexec --bind-to socket --map-by socket -n $NGPUS`
([env_defaults.sh:298 commentary, solve_periodic_NK.sh:41](scripts/solvers/solve_periodic_NK.sh#L41)).
Each rank gets one socket = one GPU = nominally ~256 GB and ~12 CPUs. That's
fine for the symmetric Φ! workers but leaves rank 0 boxed in for the
Pardiso factorisation, which is both CPU- and memory-hungry. Two levers:

- **CPU count for Pardiso**: `nprocs` at [solve_periodic_NK.jl:51](src/solve_periodic_NK.jl#L51)
  defaults to `PBS_NCPUS` (=48 for a 4-GPU job), but with socket binding
  rank 0's MKL threads can only spread across its socket's ~12 CPUs.
  Letting rank 0 use more CPUs would speed up factorisation significantly.
- **Memory headroom on rank 0**: with socket binding, rank 0's NUMA-local
  pool is ~256 GB; spilling into another socket's memory works but is
  NUMA-non-local (slow). To use the full 1024 GB without NUMA penalty
  we'd need to consolidate rank 0 onto a NUMA-friendlier layout.

Options worth trying (cheapest first), each in a one-job experiment:

1. **`--bind-to none --map-by node`** — drop CPU binding entirely. Rank 0
   can spawn MKL threads across all 48 CPUs and grow into the whole 1024 GB
   pool. Risk: workers also lose socket affinity, so Φ! CPU↔GPU traffic may
   cross sockets and slow down. Likely fine because Φ! is mostly GPU-resident.
2. **Asymmetric rankfile**: pin rank 0 to two sockets (≈24 CPUs, ≈512 GB
   NUMA-local) and pack ranks 1–3 onto the remaining two sockets. Preserves
   socket-local CPU↔GPU pairs for workers; grows rank 0's effective budget.
3. **Drop to `1x2` partition** (= 2 ranks, 2 GPUs, 512 GB and 24 CPUs per
   rank). Halves the Φ! parallelism but doubles rank 0's nominal share for
   free. Only attractive if (1) and (2) don't help and Φ! is not the
   bottleneck.

Treat this as an **opt-in** branch of the sweep — pursue only if 5×5 hits
either (a) OOM-during-factorisation, or (b) a Pardiso factorisation slow
enough that the 48 h walltime is dominated by it. Not part of the default
job submissions.

## Code changes

### 1. Parse `LUMP_AND_SPRAY = no | AxB`

Drop the `yes` alias entirely. New accepted values:

- `no` — no coarsening (`LUMP = SPRAY = I`, current `LUMP_AND_SPRAY=no` behaviour).
- `AxB` — `di = A`, `dj = B`, `dk = 1`. Positive integers; no upper bound.

Add a Julia helper in [src/shared_utils/config.jl](src/shared_utils/config.jl) (next
to `parse_config_env` and `parse_load_balance_env`):

```julia
parse_lump_and_spray(s = get(ENV, "LUMP_AND_SPRAY", "no")) -> (; di, dj, dk, on, tag)
# "no"   -> (di=0, dj=0, dk=0, on=false, tag="prec")
# "5x5"  -> (di=5, dj=5, dk=1, on=true,  tag="Q5x5")
# "yes"  -> error("LUMP_AND_SPRAY=yes is no longer supported; use 'no' or 'AxB' (e.g. '2x2').")
# other  -> error("LUMP_AND_SPRAY must be 'no' or '<int>x<int>' (got: ...)").
```

Add a matching shell-side validator in [scripts/env_defaults.sh](scripts/env_defaults.sh)
modelled on the `LOAD_BALANCE` case-statement (lines 66–73). Also derives the
shell-side `Q_TAG` (used only for logging; the directory naming is done in Julia):

```bash
case "$LUMP_AND_SPRAY" in
  no)              Q_TAG="" ;;
  yes)             echo "ERROR: LUMP_AND_SPRAY=yes is no longer supported; use 'no' or 'AxB' (e.g. 2x2)." >&2; exit 1 ;;
  [0-9]*x[0-9]*)   Q_TAG="_Q${LUMP_AND_SPRAY}" ;;
  *)               echo "ERROR: LUMP_AND_SPRAY must be 'no' or '<int>x<int>' (got: $LUMP_AND_SPRAY)" >&2; exit 1 ;;
esac
export Q_TAG
```

### 2. Replace hardcoded `(2, 2, 1)` at both call sites

- [src/solve_periodic_NK.jl:167](src/solve_periodic_NK.jl#L167)
  → `lump_and_spray(wet3D_global, v1D, M; di, dj, dk)` using the parsed values.
- [src/shared_utils/matrix.jl:67](src/shared_utils/matrix.jl#L67) (`compute_and_save_coarsening`)
  → take `di, dj, dk` as keyword args (default to the no-coarsening guard upstream
  via the new helper) and forward them to `lump_and_spray`.

### 3. Tag NK outputs with the coarsening factor

Tag both the directory and the file-level `lumpspray_tag` with `Q{A}x{B}` (the
"Q" prefix matches the math: Q is the preconditioner = `stop_time * M`).

Directory:
- LS=no:  `outputs/{PM}/{EXP}/{TW}/periodic/{MC}/[{px}x{py}/]NK/` (unchanged).
- LS=AxB: `outputs/{PM}/{EXP}/{TW}/periodic/{MC}/[{px}x{py}/]NK_Q{A}x{B}/`.

File-level tag (replaces today's `lumpspray_tag = LSprec | prec`):
- LS=no:  `lumpspray_tag = "prec"` (unchanged).
- LS=AxB: `lumpspray_tag = "Q{A}x{B}"` (replaces `"LSprec"`).

So `age_Pardiso_LSprec.jld2` becomes `age_Pardiso_Q5x5.jld2` etc., living inside
the matching `NK_Q5x5/` dir.

`newton_iterate_NN.jld2` files (written by `periodic_solver_common.jl`) inherit
the dir change automatically — they're written to `solver_output_dir`, which is
the only NK-side path that needs updating in [solve_periodic_NK.jl:108-110](src/solve_periodic_NK.jl#L108-L110).

### 4. Tag coarsened-M files in `matrices_dir`

`compute_and_save_coarsening` currently writes `LUMP.jld2`, `SPRAY.jld2`,
`Mc.jld2` directly to `matrices_dir = outputs/.../TM/{MC}/{TM_SOURCE}/`. These
will collide between factors. Write into a per-factor subdir:

`outputs/.../TM/{MC}/{TM_SOURCE}/Q{A}x{B}/{LUMP,SPRAY,Mc}.jld2`

(Only created when `on == true` — the no-coarsening path doesn't write these
files today and continues not to.)

### 5. Update downstream readers (full wire-through)

Every script that reads/writes via `lumpspray_tag` or `coarse_tag` today needs
to consume the new parsed helper. Files to update:

| File | Today | After |
|---|---|---|
| [src/solve_periodic_NK.jl:60-61,167,280](src/solve_periodic_NK.jl#L60) | `LSprec`/`prec`, hardcoded di=dj=2 | `Q{A}x{B}`/`prec`, parsed di,dj |
| [src/shared_utils/matrix.jl:67,73-75](src/shared_utils/matrix.jl#L67) | hardcoded (2,2,1), files in `matrices_dir/` | parsed di,dj,dk; files in `matrices_dir/Q{A}x{B}/` |
| [src/run_periodic_1year.jl:38-42](src/run_periodic_1year.jl#L38) | `lumpspray_tag` via yes/no | parsed; reads from `NK_Q{A}x{B}/age_..._Q{A}x{B}.jld2` |
| [src/plot_periodic_1year_age.jl:63-66](src/plot_periodic_1year_age.jl#L63) | same | same |
| [src/compute_ventilation_diagnostic.jl:77-91](src/compute_ventilation_diagnostic.jl#L77) | same | same |
| [src/solve_matrix_age.jl:80-81,106,173](src/solve_matrix_age.jl#L80) | `coarse_tag = coarse|full` | drop `coarse_tag`; tag with `Q{A}x{B}` (or unchanged when LS=no) |
| [src/solve_matrix_age_gpu.jl:75-76,99,176](src/solve_matrix_age_gpu.jl#L75) | same | same |
| [src/periodic_solver_common.jl:182-186](src/periodic_solver_common.jl#L182) | `preferred_coarse_tag = coarse\|full` for TMage fallback | use parsed `on` flag; the preferred candidate filename uses `Q{A}x{B}`, fallback uses `full` legacy naming |

Note for `solve_matrix_age{,_gpu}.jl`: today the per-file tag uses
`coarse|full`. Migrate to `Q{A}x{B}|full` to stay consistent with NK's
`Q{A}x{B}|prec`. Legacy OM2-1 `coarse` outputs become unreachable from the
new code — that's accepted (per "drop yes alias" decision).

### 6. Driver wiring

No structural change needed. `LUMP_AND_SPRAY` is already in `NK_VARS`,
`RUNNK_VARS`, `VENT_VARS`, and the TMsolve --vars lists at
[driver.sh:521,546,564,496-513](scripts/driver.sh#L521). It already forwards as
a string; the new validator in `env_defaults.sh` is the only added shell logic.
Driver default `LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-yes}` at
[driver.sh:228](scripts/driver.sh#L228) must change to `no` (since `yes` is no
longer accepted).

## Pre-flight (do first, in order)

1. **Transport matrix `M.jld2`** at the target config:
   `outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/const/M.jld2`
   — currently absent. Run `JOB_CHAIN=TMbuild` on the `hugemem` queue per
   [model_configs/ACCESS-OM2-01.sh](model_configs/ACCESS-OM2-01.sh).

2. **1×4 partition files** for `cgridtransports` velocity FTS with
   `GRID_HX/HY/HZ=7,7,2`. Existing dirs on disk:
   `partitions/{1x4, 1x4_LB, 1x4_LBS}` (not `_LBmix`, not `_LBminmax`).
   Verify parent shapes match `halo=(7,7,2)` (the OM2-1 1×2 test hit a
   halo=13 mismatch — same risk); if stale, rebuild via
   `JOB_CHAIN=partition`. The two missing LB variants must be built before
   the LB sweep below.

3. **Load-balance choice for `wparent`**: brief 1-year `run1yr` sweep at
   `PARTITION=1x4` across `LOAD_BALANCE ∈ {no, surface, cell, mix, minmax}`,
   pick the lowest wall-time. (`wparent` has no w-compute kernel; the prior
   LB choice was made for `wdiagnosed`, so the optimum may differ.)

`diagnose_w` is **not** needed for `wprescribed`+`parent`.

## Coarsening sweep

Start with `5x5` as the central case (already in flight); walk in whichever
direction the failure mode points. Each job is an independent 48 h submission
with `INITIAL_AGE=0`.

| Direction | Factors | When to use |
|---|---|---|
| Centre | `5x5` | **In flight.** Per the estimate table, should be the sweet spot for both memory and factorisation time at OM2-01. |
| Finer (more memory, better preconditioner) | `4x4`, `3x3`, … | If 5×5 converges too slowly: GMRES makes little progress per Newton step, or Newton residual is flat across many iterates. Do **not** attempt `2x2` (2.34 TB, ~10 h — infeasible). |
| Coarser (less memory, weaker preconditioner) | `6x6`, `7x7`, … | If 5×5 OOMs during Pardiso factorisation, or factorisation eats most of the walltime even after binding tweaks. |

After each 48 h job:
- Converged: done. Record (factor, factorise time, peak rank-0 RSS, Newton
  steps, Φ! calls) in the doc.
- Walltime hit + `newton_iterate_NN.jld2` files present: resubmit with
  `INITIAL_AGE=latest` (numbering continues per `g_count_base` in
  `periodic_solver_common.jl`).
- GMRES not making progress / Newton residual flat: step to the next finer
  factor (4×4, then 3×3).
- OOM during Pardiso factorisation, **or factorisation taking >>1 hour**:
  try the binding levers above first; if still bad, step to the next
  coarser factor (6×6, 7×7, …). Record rank-0 peak RSS and factorise time
  in the doc below.

The plan is open-ended on both ends — the AxB parser has no upper bound on
A or B (positive integers only), so any coarsening factor the sweep needs
is supported without further code changes. Hard floor: do not go below `3x3`
(per the estimate table, 2×2 is infeasible at OM2-01).

## Deliverables

- `src/shared_utils/config.jl`: new `parse_lump_and_spray()` helper.
- `src/solve_periodic_NK.jl`, `src/shared_utils/matrix.jl`: parser-driven
  `(di, dj, dk)` and `Q{A}x{B}`-tagged outputs (incl. `solver_output_dir`,
  steady age file, `LUMP.jld2`/`SPRAY.jld2`/`Mc.jld2`).
- Downstream wire-through: `run_periodic_1year.jl`, `plot_periodic_1year_age.jl`,
  `compute_ventilation_diagnostic.jl`, `solve_matrix_age.jl`,
  `solve_matrix_age_gpu.jl`, `periodic_solver_common.jl:182`.
- `scripts/env_defaults.sh`: case-statement validator for `LUMP_AND_SPRAY`;
  reject `yes`.
- `scripts/driver.sh`: change default `LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-yes}`
  → `no` at line 228.
- 1-year LB sweep submission for OM2-01 1×4 `wparent` to pick `LOAD_BALANCE`.
- TMbuild submission for OM2-01 at the target `MODEL_CONFIG`.
- Three NK submission scripts under `test/` (5×5, 4×4, 3×3); each a single
  48 h job, no chain.
- A doc analogous to `docs/NK_restart_tests.md` to log outcomes (per-job
  retcode, iteration count if any, rank-0 peak RSS, walltime spent).

## Verification

- Unit-level: parser tests in a small `test/test_parse_lump_and_spray.jl`
  asserting `"no"`, `"2x2"`, `"5x5"`, error on `"yes"`, error on `"5"` /
  `"5x"` / `"5xfoo"`.
- Integration on OM2-1 (cheap): rerun an existing OM2-1 NK config with
  `LUMP_AND_SPRAY=2x2` and confirm it produces the same numeric result as
  the legacy `LUMP_AND_SPRAY=yes` run (different filename/dir, same age
  field within machine precision). This is the regression check that the
  `yes`→`2x2` rename hasn't silently changed semantics.
- Path collision check: after the 5×5 + 4×4 (or 5×5 + manual 2×2) submissions,
  `ls outputs/.../periodic/{MC}/1x4/` should show two disjoint `NK_Q?x?`
  directories with their own `newton_iterate_NN.jld2` series — confirm via
  `find … -name 'newton_iterate_*.jld2'` that no file is shared.
- End-to-end at OM2-01: pre-flight LB sweep job completes; TMbuild produces
  the expected `M.jld2`; NK 5×5 job either converges or fails with a clean
  diagnosable error (OOM is fine — we log it).

## Out of scope

- Generalising the `dk` factor (we never coarsen in z; helper hardcodes `dk=1`
  when `on=true`).
- Migrating existing OM2-1 `LSprec` / `coarse` outputs to the new naming
  (they're left as-is; downstream code only finds them under the old names).
- Changes to `MODEL_CONFIG` — `LUMP_AND_SPRAY` is preconditioner-only and
  stays out of the model_config tag.
