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
2. **LB sweep at OM2-01 (wparent) via nsys profiles** (not full 1-year runs).
   See [docs/profiling_workflow.md](profiling_workflow.md) for the canonical
   workflow — `PROFILE=yes` + `BENCHMARK_STEPS=200` on a `run1yrfast`
   driver invocation produces a 200-step nsys-traced run per LB variant.
   The driver already plumbs everything through.

   ```bash
   PARENT_MODEL=ACCESS-OM2-01 \
     PARTITION=1x4 LOAD_BALANCE=<variant> \
     LUMP_AND_SPRAY=5x5 \
     PROFILE=yes BENCHMARK_STEPS=200 \
     JOB_CHAIN=run1yrfast bash scripts/driver.sh
   ```

   (`W_FORMULATION=wprescribed`, `PRESCRIBED_W_SOURCE=parent`,
   `TIMESTEP_MULT=2`, `MONTHLY_KAPPAV=yes`, `TM_SOURCE=const`,
   `TRACE_SOLVER_HISTORY=yes` are all defaults after the recent
   env/model-config changes — no need to set them.)

   LB variants to run (in priority order):
   - `surface` (LBS) — partition `1x4_LBS` ✓ correct halo
   - `no` — partition `1x4` ✓ rebuilt (`169132252`)
   - `cell` (LB) — partition `1x4_LB` ✓ rebuilt (`169132264`)

   Skip `mix` and `minmax` for now (partitions not built; not worth the
   extra megamem build at this stage).

   **Why re-sweep**: the existing OM2-01 LB results in
   [docs/profiling_results_v2.md § Phase 3](profiling_results_v2.md#phase-3-om2-01--h200-gpuhopper)
   were measured under `W_FORMULATION=wdiagnosed`. The
   `compute_w_from_continuity!` kernel is about **25%** of step cost
   under wdiagnosed — not the dominant term, but not negligible either.
   Switching to `wparent` drops it entirely, so the LB ranking may
   shift; the wparent sweep is what decides the production LB choice.
3. **Sweep finer/coarser if 5×5 stalls** (per [NK_OM2-01.md](NK_OM2-01.md)
   § Coarsening sweep): walk to `4x4`, `3x3` if preconditioner too weak;
   `6x6`, `7x7` if memory/time pressure (hard floor at `3x3`; `2x2` is
   infeasible).

## How to submit (full env-var set)

After the defaults landed in `scripts/env_defaults.sh` and
`model_configs/*.sh` (this session, see "Defaults that landed" below),
the OM2-01 NK submission collapses to:

```bash
PARENT_MODEL=ACCESS-OM2-01 \
  PARTITION=1x4 LOAD_BALANCE=surface \
  LUMP_AND_SPRAY=5x5 \
  JOB_CHAIN=NK \
  bash scripts/driver.sh
```

Every other relevant variable comes from defaults:

| Var | Default source | Value |
|---|---|---|
| `VELOCITY_SOURCE` | env_defaults | `cgridtransports` |
| `W_FORMULATION` | env_defaults | `wprescribed` (was `wdiagnosed`) |
| `PRESCRIBED_W_SOURCE` | env_defaults | `parent` |
| `ADVECTION_SCHEME` | env_defaults | `centered2` |
| `TIMESTEPPER` | env_defaults | `AB2` |
| `TIMESTEP_MULT` | model_config | `2` for OM2-01 (4 for OM2-1, 3 for OM2-025) |
| `MONTHLY_KAPPAV` | env_defaults | `yes` |
| `MATRIX_PROCESSING` | env_defaults | `symdrop` |
| `LINEAR_SOLVER` | env_defaults | `Pardiso` |
| `TM_SOURCE` | env_defaults | `const` (was `avg`) |
| `TRACE_SOLVER_HISTORY` | env_defaults | `yes` (was `no`) |
| `INITIAL_AGE` | env_defaults | `0` (use `latest` on restart) |
| `GPU_QUEUE` | OM2-01 model_config | `gpuhopper` |

## Defaults that landed this session

`scripts/env_defaults.sh`:

- `W_FORMULATION`: `wdiagnosed` → **`wprescribed`** (with `PRESCRIBED_W_SOURCE=parent` unchanged).
- `TM_SOURCE`: `avg` → **`const`**.
- `TRACE_SOLVER_HISTORY`: `no` → **`yes`** (needed for `INITIAL_AGE=latest` restarts).
- Model config is now sourced **before** the variable defaults so per-model
  defaults (e.g. `TIMESTEP_MULT`) win over the cross-model fallbacks. User
  env-var overrides still win over everything.

`model_configs/ACCESS-OM2-1.sh`:

- `TIMESTEP_MULT=${TIMESTEP_MULT:-4}` (was implicit 1).

`model_configs/ACCESS-OM2-025.sh`:

- `TIMESTEP_MULT=${TIMESTEP_MULT:-3}` (was implicit 1).

`model_configs/ACCESS-OM2-01.sh` (in addition to the TMbuild bumps from
earlier in the session):

- `TIMESTEP_MULT=${TIMESTEP_MULT:-2}` (was implicit 1).

`src/create_matrix.jl`:

- Dropped the `M_spy.png` post-save plot (was crashing at OM2-01 scale
  and producing exit-1, breaking `JOB_CHAIN=TMbuild-NK` chains).

## Still left to land (TBD)

- `LOAD_BALANCE` default — leave at `no` until the wparent LB sweep
  (item 2 above) tells us which variant wins. Then bake that into the
  model configs (probably per-model: OM2-1/025/01 may not agree).
- `PARTITION` default — currently `1x1`. OM2-01 effectively needs `1x4`
  on H200; could bake per-model. Less urgent — the driver already
  errors clearly if NGPUS=1 is used at OM2-01.

## Pointers

- Full history + outcome log: [docs/NK_OM2-01.md](NK_OM2-01.md).
- Profiling workflow + nsys mechanics: [docs/profiling_workflow.md](profiling_workflow.md).
- Existing LB-sweep results under `wdiagnosed` (the comparison baseline
  for the upcoming wparent sweep): [docs/profiling_results_v2.md](profiling_results_v2.md).
- TMbuild scaling note (OM2-025 → OM2-01) is in [NK_OM2-01.md](NK_OM2-01.md)
  § Context. Sparsity detection scales ~18× (vs ~10× for everything else)
  due to GC pressure — keep an eye on this if matrix size grows further.
- Submissions index: `scripts/runs/submissions.tsv` (use
  `scripts/runs/reconcile_submissions.sh` to fill PBS-side columns after
  jobs finish).
