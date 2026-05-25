# NK solver at OM2-01: settings, code changes, coarsening sweep

## Status

- **Code changes**: ✅ landed in commit `7b581cb`. Parser + downstream
  wire-through + shell validator implemented; parser unit tests pass;
  shell validator accepts `no | AxB` and rejects `yes`/malformed.
  Hardcoded `(2,2,1)` removed from both call sites.
- **5×5 probe (TM age, separate repo)**: ❌ failed at 8 min with NetCDF
  "no such file or directory" at `compute_age_ACCESS-OM2-01_5x5.jl:61`
  (job `169125041`, in `../ACCESS-TMIP/`). Didn't reach Pardiso —
  yields no memory/time data for the sweep. Probe needs an input path
  fix before retry.
- **LB sweep (done, winner = `surface`)**: only `1x4_LBS` had a current
  halo=(7,7,2) partition — `1x4` and `1x4_LB` were stale (halo=(13,13,7)
  and (19,19,7) respectively, from earlier experiments).
  - `169128254` (LB=no): ❌ failed at 11 min — partition halo mismatch.
  - `169128255` (LB=surface): ✅ 52 min, 337 GB.
  - `169128256` (LB=cell): killed before running (partition stale; would
    have failed identically).
  - To compare against `no`/`cell`/`mix`/`minmax`, rebuild the
    corresponding partitions first (`JOB_CHAIN=partition LOAD_BALANCE=…`).
    Not on the critical path — LB=surface is good enough to proceed.
- **TMbuild + NK_5x5**:
  - 1st attempt: TMbuild `169132266` (normal, 192 GB) ❌ **OOM** at 34 min
    (used 186 GB, SIGKILL/exit 137); NK `169132267` purged.
  - 2nd attempt: TMbuild `169134142` (hugemem, 768 GB, 4 h) ❌ **walltime
    exceeded** (exit -29) — completed sparsity (3h 12m) + prep (7m) but
    ran out before computing the Jacobian. Used 692 GB of 768 GB
    (memory fine); NK `169134143` purged.
  - 3rd attempt (`7deb7a3`): TMbuild `169141749` (hugemem 24 CPU / 768 GB / 10 h)
    🗑 killed before completion to bump resources (only ~1 h in).
  - 4th attempt (in flight, `6a60ca9`): TMbuild `169148432` on **hugemem
    48 CPU / 1470 GB / 10 h** (PBS rejected 1536 GB; node max is 1470 GB).
    NK `169148433` held.
  - Per-model TMbuild mem/queue/ncpus overrides + walltime now wired
    through driver and `model_configs/ACCESS-OM2-01.sh`. OM2-1/025
    unchanged.
  - MC = `cgridtransports_wparent_centered2_AB2_mkappaV_LBS_DTx2`.

### TMbuild scaling at OM2-01 vs OM2-025

OM2-025 reference: job `168858362` (totaltransport / wdiagnosed) finished
in 16 min on normal/192 GB. Scaling factors going to OM2-01:

| Stage | OM2-025 | OM2-01 | Ratio |
|---|---|---|---|
| Nidx (wet cells) | 37 M | 352 M | 9.5× |
| Tendency eval | 51 s | 561 s | 11× |
| Detect sparsity | 10.5 min | 192 min | **18×** |
| Prepare Jacobian | 39 s | 7 min | 11× |
| nnz(S) | 255 M | 2.44 B | 9.6× |
| Compute Jacobian | 3.5 min | (didn't finish) | — |

Most steps scale ≈10× (matches the 9.5× Nidx growth). The outlier is
**sparsity detection at 18×** — likely GC pressure (55% GC time at OM2-01
vs 43% at OM2-025; 3.5 TiB allocations).

### run1yr LB sweep follow-on (real failure, NOT blocking for NK)

Both partition reruns completed (`1x4` and `1x4_LB` are now at the
correct halo). The follow-on run1yr jobs both failed during their
JLD2 output writers (not GPU sync — that was the death-cascade):
`InvalidDataException: Invalid Object header signature` in
`JLD2.HeaderMessageIterator` → `JLD2.load_datatypes` inside
`Oceananigans.OutputWriters.jld2_writer.jl:385` (`jld2output!`):

- `169132253` LB=no: failed at sim iter 29592 (0.75 yr, 32 min wall);
  `age_1year_rank0.jld2` ended up 715 bytes (truncated to header). u, v,
  w, eta files had real data (7.2 GB, 7.2 GB, 1.2 GB, 618 MB).
- `169132265` LB=cell: failed at sim iter 0 (first snapshot, 21 min).

Root cause hypothesis: JLD2 0.6.4 `MmapIO` + `with_halos=true` + the
OM2-01 distributed file sizes. The writer's `prewrite` call reads back
the existing committed-datatype catalog and corrupted state trips
`HeaderMessageIterator`. Not yet root-caused — TBD post-NK.

**This failure path is NOT in NK's call chain.** NK uses
`periodic_solver_common.jl` which only includes the bare
`setup_simulation.jl` (`Simulation(model; Δt, stop_time)` + prescribing
callbacks) and explicitly does NOT call `setup_age_simulation`. NK saves
the final age + checkpoint iterates via one-shot `jldsave(...)`, not via
`JLD2Writer`/`MmapIO`, so it doesn't touch the buggy code path.

run1yr fix is a follow-up; it doesn't gate the NK_5x5 evaluation.

**Reproducer in flight**: `169159227` — OM2-1 `run1yr` at PARTITION=1x2,
wparent, MONTHLY_KAPPAV=yes, DTx2, LB=no. Tiny (~8 min compute) but with
the same writer setup. If it reproduces, the bug is JLD2/writer related,
not OM2-01-specific.

**For future LB sweeps**, switch from `run1yr` to `run1yrfast`
(`scripts/standard_runs/run_1year_benchmark.sh` / `src/run_1year_benchmark.jl`)
— same simulation, no output writers, no JLD2 risk. Walltime alone is
all we need for an LB winner pick. The `run1yr` writer outputs are only
useful for diagnostic plotting, which we don't need during the LB sweep.
- **Partition rebuilds + run1yr (in flight)**:
  - LB=no: partition `169132252` (megamem 1.4 TB) → run1yr `169132253`.
  - LB=cell: partition `169132264` (megamem 1.4 TB) → run1yr `169132265`.
  - These complete the LB sweep retroactively; the NK chain doesn't wait
    on them.

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

### Code (done)

- ✅ `src/shared_utils/config.jl`: new `parse_lump_and_spray()` helper
  returning `(; di, dj, dk, on, tag, dir_suffix)`. Accepts `no | AxB`;
  rejects legacy `yes` with a clear redirect.
- ✅ `src/shared_utils/matrix.jl`: `compute_and_save_coarsening` now takes
  `di/dj/dk/on/tag` kwargs and writes `LUMP.jld2`/`SPRAY.jld2`/`Mc.jld2`
  into `{matrices_dir}/{tag}/` (e.g. `…/const/Q5x5/Mc.jld2`).
- ✅ `src/solve_periodic_NK.jl`: parser-driven coarsening, NK output dir
  becomes `…/NK{dir_suffix}` (e.g. `NK_Q5x5/`); steady file unchanged
  format-wise (`age_{LINEAR_SOLVER}_{tag}.jld2`).
- ✅ `src/solve_matrix_age.jl` and `src/solve_matrix_age_gpu.jl`:
  `coarse_tag` migrated to `Q{A}x{B}` (when on) or `"full"` (when off).
- ✅ `src/periodic_solver_common.jl`: TMage warm-start fallback uses the
  parser; tries the preferred `Q{A}x{B}` (or `full`) tag, falls back to
  `full` only.
- ✅ `src/run_periodic_1year.jl`, `src/plot_periodic_1year_age.jl`,
  `src/compute_ventilation_diagnostic.jl`: all consume the parser; NK
  output paths now resolve to `NK{dir_suffix}` (matches the writer).
- ✅ `scripts/env_defaults.sh`: validator added (no | AxB; reject yes);
  derives `Q_TAG` for logging.
- ✅ `scripts/driver.sh`: default flipped `LUMP_AND_SPRAY=${...:-yes}` →
  `no` at line 228.
- ✅ `test/test_parse_lump_and_spray.jl`: unit tests for the parser
  (off/on cases, error cases). Passes on login node.

### Operational (pending)

- ⏳ Wait for 5×5 NK test job to complete (submitted before code landed).
  Compare against post-landing behaviour to confirm output paths align.
- ☐ 1-year LB sweep submission for OM2-01 1×4 `wparent` to pick
  `LOAD_BALANCE`.
- ☐ Submit 4×4 and 3×3 NK jobs after 5×5 result is known (one job per
  factor, manual restart via `INITIAL_AGE=latest`; no chains).
- ☐ Outcome log section below.

## Verification

- ✅ **Unit-level**: `julia --project test/test_parse_lump_and_spray.jl`
  passes (off cases, on cases incl. asymmetric/case-insensitive, error
  cases incl. `yes`, empty, `"5"`, `"5x"`, `"5xfoo"`, `"0x5"`, `"5x0"`).
- ✅ **Shell validator smoke test**:
  `LUMP_AND_SPRAY={no,5x5,2x2,yes,bad,10x10}` produces the expected
  `Q_TAG` (or non-zero exit for `yes`/`bad`).
- ☐ **Integration on OM2-1**: rerun an existing OM2-1 NK config with
  `LUMP_AND_SPRAY=2x2` and compare numerics against the legacy `yes`
  run (different filename/dir, same age within machine precision). This
  is the regression check that the `yes`→`2x2` rename hasn't silently
  changed semantics.
- ☐ **Path collision check**: after 5×5 + 4×4 submissions,
  `ls outputs/.../periodic/{MC}/1x4/` should show `NK_Q5x5/` and
  `NK_Q4x4/` as disjoint dirs; `find … -name 'newton_iterate_*.jld2'`
  confirms no file is shared.
- ☐ **End-to-end at OM2-01**: pre-flight LB sweep completes; NK 5×5 job
  either converges or fails with a clean diagnosable error (OOM is fine
  — we log it).

## Outcome log

| Date | Job ID | Kind | Factor / LB | Result | Notes |
|---|---|---|---|---|---|
| 2026-05-24 | `169124672` | TM age (ACCESS-TMIP) | 5×5 | ❌ failed in 24 s | quick fail; presumed config/path issue (no log content captured here) |
| 2026-05-24 | `169125041` | TM age (ACCESS-TMIP) | 5×5 | ❌ failed at 8 min | NetCDF "no such file or directory" at `compute_age_ACCESS-OM2-01_5x5.jl:61`; never reached Pardiso. Used 16 GB / 24 CPU on hugemem. |
| 2026-05-24 | `169128254` | run1yr LB sweep | LB=no | ❌ 11 min | partition halo mismatch — `1x4` was built at halo=(13,13,7); runtime wants (7,7,2). Needs `JOB_CHAIN=partition` to rebuild. |
| 2026-05-24 | `169128255` | run1yr LB sweep | LB=surface | ✅ 52 min | 337 GB used. Only working LB so far. **Picked for TMbuild + NK.** |
| 2026-05-24 | `169128256` | run1yr LB sweep | LB=cell | 🗑 killed | `1x4_LB` partition halo=(19,19,7), also stale. Killed before run to save compute. |
| 2026-05-24 | `169132252` | partition rebuild | LB=no | ⏳ running | Megamem 1.4 TB, 2 h walltime. Rebuilds `1x4` with halo=(7,7,2). |
| 2026-05-24 | `169132253` | run1yr rerun | LB=no | ⏸ held (afterok 169132252) | |
| 2026-05-24 | `169132264` | partition rebuild | LB=cell | ⏳ running | Megamem 1.4 TB. Rebuilds `1x4_LB`. |
| 2026-05-24 | `169132265` | run1yr rerun | LB=cell | ⏸ held (afterok 169132264) | |
| 2026-05-24 | `169132252` | partition rebuild | LB=no | ✅ 47 min | 1.30 TB used. Rebuilt `1x4` at halo=(7,7,2). |
| 2026-05-24 | `169132264` | partition rebuild | LB=cell | ✅ 42 min | 1.26 TB used. Rebuilt `1x4_LB` at halo=(7,7,2). |
| 2026-05-24 | `169132266` | TMbuild (1st) | LBS | ❌ 34 min OOM | exit 137; used 186 GB of 192 GB on normal queue. Triggered TMBUILD_MEM override (hugemem 768 GB) in `d1cc86d`. |
| 2026-05-24 | `169132267` | NK_5x5 (1st) | LBS / 5×5 | 🗑 purged | afterok dep failed when TMbuild OOM'd. |
| 2026-05-24 | `169132253` | run1yr LB=no | LB=no | ❌ 48 min, exit 1 | CUDA `take!` sync error on rank 3. 372 GB used. Partition was fresh, halo correct — separate failure mode. |
| 2026-05-24 | `169132265` | run1yr LB=cell | LB=cell | ❌ 21 min, exit 1 | Same CUDA sync error. 317 GB used. |
| 2026-05-24 | `169134142` | TMbuild (2nd) | LBS | ❌ 4h01m walltime | exit -29; used 692 GB. Done with sparsity+prep, ran out before Jacobian compute. |
| 2026-05-24 | `169134143` | NK_5x5 (2nd) | LBS / 5×5 | 🗑 purged | afterok 169134142 failed. |
| 2026-05-25 | `169141749` | TMbuild (3rd) | LBS | 🗑 killed at 1 h | resource bump after the fact — see 4th attempt. |
| 2026-05-25 | `169141750` | NK_5x5 (3rd) | LBS / 5×5 | 🗑 killed | qdel together with parent. |
| 2026-05-25 | `169148432` | TMbuild (4th) | LBS | ⏳ running | hugemem **48 CPU / 1470 GB / 10 h** (`6a60ca9`). Max usable hugemem. |
| 2026-05-25 | `169148433` | **NK_5x5 (4th)** | LBS / 5×5 | ⏸ held (afterok 169148432) | gpuhopper 1×4, 24 h. |
| 2026-05-25 | `169159227` | OM2-1 reproducer | LB=no | ⏳ queued | OM2-1 run1yr at 1×2 wparent mkappaV DTx2 LB=no. Tiny test of the JLD2 failure path. |

## Out of scope

- Generalising the `dk` factor (we never coarsen in z; helper hardcodes `dk=1`
  when `on=true`).
- Migrating existing OM2-1 `LSprec` / `coarse` outputs to the new naming
  (they're left as-is; downstream code only finds them under the old names).
- Changes to `MODEL_CONFIG` — `LUMP_AND_SPRAY` is preconditioner-only and
  stays out of the model_config tag.
