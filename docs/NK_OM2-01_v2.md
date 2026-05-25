# NK at OM2-01 — handoff (v2)

A focused continuation of [NK_OM2-01.md](NK_OM2-01.md). That earlier doc captures the full
history (TMbuild OOMs/walltime fights, run1yr JLD2 collision, etc.) and the
`LUMP_AND_SPRAY=AxB` refactor that landed. This file lists only **what's
running, what's left, and what's needed to submit** so a fresh session can
pick up cleanly.

## In flight

| Job | Step | Resources | Notes |
|---|---|---|---|
| `169183483` | **NK_5x5** | gpuhopper 1×4 / 1024 GB / 24 h | First real NK at OM2-01. M.jld2 already on disk; no upstream deps. |

`M.jld2` (39 GB) lives at
`outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/const/M.jld2`
(no `_LBS` in the Julia-side MC path — see "Known caveats" below).

## Remaining work

1. **Wait for `169183483` to land** and record: walltime, peak rank-0 RSS,
   number of Newton iterations, GMRES rate. Update [NK_OM2-01.md](NK_OM2-01.md)
   outcome log.
2. **LB sweep at OM2-01** via short **nsys profiles**, not full 1-year runs.
   Full year is wasteful and full `run1yr` writes JLD2 (which clobbers
   across LB variants at the same Julia MC; see "Known caveats"). The
   profile-based approach gives a clean per-step walltime for each LB
   variant in ~10 min each.
   - Use `JOB_CHAIN=run1yrncu` (driver step) or the standalone scripts:
     - [`scripts/run_nsys_stats.sh`](../scripts/run_nsys_stats.sh) — single nsys run
     - [`scripts/run_nsys_stats_batch.sh`](../scripts/run_nsys_stats_batch.sh) — multiple back-to-back
   - Inspect existing example: `logs/julia/ACCESS-OM2-01/.../*1yearfast_169027665.gadi-pbs_profile_syncGCyes_N5_rank*.nsys-rep`.
   - LB variants to sweep (and partition state):
     - `surface` (LBS) — partition `1x4_LBS` ✓ correct halo
     - `no` — partition `1x4` ✓ rebuilt (`169132252`)
     - `cell` (LB) — partition `1x4_LB` ✓ rebuilt (`169132264`)
     - `mix`, `minmax` — partitions not built yet; build with
       `JOB_CHAIN=partition LOAD_BALANCE=mix` (then `minmax`) before profiling.
3. **Sweep finer/coarser if 5×5 stalls** (per [NK_OM2-01.md](NK_OM2-01.md)
   § Coarsening sweep): walk to `4x4`, `3x3` if preconditioner too weak;
   `6x6`, `7x7` if memory/time pressure (hard floor at `3x3`; `2x2` is
   infeasible).
4. **TMbuild post-save plot crash** — non-fatal, but causes exit-1.
   `src/create_matrix.jl:112` calls `save(..., "M_spy.png", fig)` *after*
   `M.jld2` is already saved; the CairoMakie call OOMs/crashes at OM2-01
   scale. Either wrap that save in `try/catch`, or `OMIT_SPY_PLOT=yes` env
   flag, or just delete the spy plot at OM2-01. Doing so unblocks
   `JOB_CHAIN=TMbuild-NK` chaining (currently afterok bombs the NK job
   because TMbuild "fails" after writing M.jld2).
5. **Include LB tag in Julia output path** to prevent the run1yr concurrent-write
   collision documented in commit `cad68af`/`830f3a1`. Mirror the shell-side
   logic via `parse_load_balance_env()` (in
   [`src/shared_utils/load_balance.jl:257`](../src/shared_utils/load_balance.jl#L257))
   inside [`build_model_config()`](../src/shared_utils/config.jl#L114-L130).
   This is what makes simultaneous LB-sweep submissions safe.

## How to submit (full env-var set)

The submission below is what landed `169183483`. Variables differing from
[scripts/env_defaults.sh](../scripts/env_defaults.sh) defaults are marked
**(non-default)** — these are the levers worth folding into the model
config (next section).

```bash
PARENT_MODEL=ACCESS-OM2-01 \
W_FORMULATION=wprescribed       \   # (non-default; default: wdiagnosed)
PRESCRIBED_W_SOURCE=parent      \   # default; explicit here for clarity
TIMESTEP_MULT=2                 \   # (non-default; default: 1)
MONTHLY_KAPPAV=yes              \   # already default since user edit; explicit here for clarity
PARTITION=1x4                   \   # (non-default; default: 1x1)
LOAD_BALANCE=surface            \   # (non-default; default: no)
LUMP_AND_SPRAY=5x5              \   # (non-default; default: no — varied across sweep)
MATRIX_PROCESSING=symdrop       \   # default
LINEAR_SOLVER=Pardiso           \   # default
TM_SOURCE=const                 \   # (non-default; default: avg)
TRACE_SOLVER_HISTORY=yes        \   # (non-default; default: no — needed for restart)
INITIAL_AGE=0                   \   # default; use `latest` on restart
JOB_CHAIN=NK                    \   # (or TMbuild-NK once spy-plot crash is fixed)
bash scripts/driver.sh
```

`GPU_QUEUE=gpuhopper` is already the OM2-01 default in the model config.

## Model-config updates worth landing

These bake OM2-01-specific choices that survived the iteration into
`model_configs/ACCESS-OM2-01.sh` so they don't have to be set per
submission. Already in (see `git log model_configs/ACCESS-OM2-01.sh`):

- `WALLTIME_TM_BUILD=10:00:00`
- `TMBUILD_QUEUE=hugemem`, `TMBUILD_NCPUS=48`, `TMBUILD_MEM=1470GB`

Suggested additions (require small `env_defaults.sh` / `driver.sh`
plumbing to make them per-model overridable, mirroring `TMBUILD_*`):

| Env var | Suggested OM2-01 default | Justification |
|---|---|---|
| `W_FORMULATION` | `wprescribed` | OM2-01 NK uses the parent w directly; `wdiagnosed` is impractical here. |
| `PRESCRIBED_W_SOURCE` | `parent` | Same. |
| `TIMESTEP_MULT` | `2` | Stable Δt for OM2-01 with this circulation. |
| `PARTITION` | `1x4` | OM2-01 won't fit in fewer GPUs on H200. |
| `LOAD_BALANCE` | `surface` | Only LB variant validated end-to-end so far (LBS). |
| `TM_SOURCE` | `const` | Only TM kind built for OM2-01 to date. |
| `TRACE_SOLVER_HISTORY` | `yes` | Needed for `INITIAL_AGE=latest` restarts; the OM2-01 NK budget is multi-job. |

Once those land, the OM2-01 submission collapses to roughly:

```bash
PARENT_MODEL=ACCESS-OM2-01 LUMP_AND_SPRAY=5x5 JOB_CHAIN=NK bash scripts/driver.sh
```

## Known caveats

- **Julia `build_model_config()` does NOT include the LB tag** in the
  output path; only the shell-side `MODEL_CONFIG` does (and that only
  feeds log filenames). As of today all LB variants share the same
  `outputs/.../standardrun/{MC}/{px}x{py}/` and `outputs/.../TM/{MC}/`.
  This is why `M.jld2` is at the `…_DTx2/` path even though we submitted
  with `LOAD_BALANCE=surface`, and why the LB=no/cell run1yr reruns
  silently clobbered each other on May 25. **Fix this before any
  concurrent LB-sweep submissions** (see "Remaining work" item 5).
- **JLD2 0.6.4 `MmapIO` does not refuse a re-opened-and-truncated file**;
  it silently leaves the existing mmap incoherent. JLD2 isn't doing
  anything wrong, but it does mean a second submission to the same
  output path is *destructive*. The LB-tag fix above is sufficient — no
  JLD2 changes needed.
- **`run1yr` is the wrong tool for an LB sweep.** It writes ~7 GB JLD2 per
  field per rank; nothing in the sweep cares about those outputs. Use a
  short nsys-profiled run (see "Remaining work" item 2).

## Pointers

- Full history + outcome log: [docs/NK_OM2-01.md](NK_OM2-01.md).
- TMbuild scaling note (OM2-025 → OM2-01) is in [NK_OM2-01.md](NK_OM2-01.md)
  § Context. Sparsity detection scales ~18× (vs ~10× for everything else)
  due to GC pressure — keep an eye on this if matrix size grows further.
- Submissions index: `scripts/runs/submissions.tsv` (use
  `scripts/runs/reconcile_submissions.sh` to fill PBS-side columns after
  jobs finish).
