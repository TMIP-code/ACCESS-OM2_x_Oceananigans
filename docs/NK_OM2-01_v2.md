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
`outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/const/M.jld2`.

## Remaining work

1. **Wait for `169183483` to land** and record: walltime, peak rank-0 RSS,
   number of Newton iterations, GMRES rate. Update [NK_OM2-01.md](NK_OM2-01.md)
   outcome log.
2. **LB sweep at OM2-01 via nsys profiles** (not full 1-year runs). See
   [docs/profiling_workflow.md](profiling_workflow.md) for the canonical
   workflow — just add `PROFILE=yes` to a `run1yrfast` driver invocation
   and you get a ~20-step nsys-traced run per LB variant. The driver
   already plumbs everything through; nothing new to wire up.

   ```bash
   PARENT_MODEL=ACCESS-OM2-01 W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=parent \
     TIMESTEP_MULT=2 MONTHLY_KAPPAV=yes PARTITION=1x4 LOAD_BALANCE=<variant> \
     PROFILE=yes JOB_CHAIN=run1yrfast bash scripts/driver.sh
   ```

   - LB variants to sweep (and partition state):
     - `surface` (LBS) — partition `1x4_LBS` ✓ correct halo
     - `no` — partition `1x4` ✓ rebuilt (`169132252`)
     - `cell` (LB) — partition `1x4_LB` ✓ rebuilt (`169132264`)
     - `mix`, `minmax` — partitions not built; build with
       `JOB_CHAIN=partition LOAD_BALANCE=mix` (then `minmax`) before profiling.
   - **Why we re-sweep**: the existing OM2-01 LB results in
     [docs/profiling_results_v2.md § Phase 3](profiling_results_v2.md#phase-3-om2-01--h200-gpuhopper)
     were all measured under `W_FORMULATION=wdiagnosed`, which is dominated
     by the `compute_w_from_continuity!` kernel. With `wparent` that kernel
     is gone — the per-step cost profile is different, so the LB ranking
     may shift. The wparent sweep is what tells us which LB variant to use
     for production NK runs.
3. **Sweep finer/coarser if 5×5 stalls** (per [NK_OM2-01.md](NK_OM2-01.md)
   § Coarsening sweep): walk to `4x4`, `3x3` if preconditioner too weak;
   `6x6`, `7x7` if memory/time pressure (hard floor at `3x3`; `2x2` is
   infeasible).

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
JOB_CHAIN=NK                    \   # M.jld2 already on disk — no TMbuild needed
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
| `LOAD_BALANCE` | `surface` | Only LB variant validated end-to-end so far (LBS); revisit after wparent LB sweep (item 2). |
| `TM_SOURCE` | `const` | Only TM kind built for OM2-01 to date. |
| `TRACE_SOLVER_HISTORY` | `yes` | Needed for `INITIAL_AGE=latest` restarts; the OM2-01 NK budget is multi-job. |

Once those land, the OM2-01 submission collapses to roughly:

```bash
PARENT_MODEL=ACCESS-OM2-01 LUMP_AND_SPRAY=5x5 JOB_CHAIN=NK bash scripts/driver.sh
```

## Pointers

- Full history + outcome log: [docs/NK_OM2-01.md](NK_OM2-01.md).
- Profiling workflow + nsys mechanics: [docs/profiling_workflow.md](profiling_workflow.md).
- Existing LB-sweep results (under `wdiagnosed` — to be re-run for
  `wparent`): [docs/profiling_results_v2.md](profiling_results_v2.md).
- TMbuild scaling note (OM2-025 → OM2-01) is in [NK_OM2-01.md](NK_OM2-01.md)
  § Context. Sparsity detection scales ~18× (vs ~10× for everything else)
  due to GC pressure — keep an eye on this if matrix size grows further.
- Submissions index: `scripts/runs/submissions.tsv` (use
  `scripts/runs/reconcile_submissions.sh` to fill PBS-side columns after
  jobs finish).
