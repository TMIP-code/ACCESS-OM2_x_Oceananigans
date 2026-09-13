# Li et al. meltwater — next-session plan (post forward-age)

Quick, executable handoff. Companion to [../Li-etal_simulations.md](../Li-etal_simulations.md)
(full plan + §7 run tracking). Start a fresh session and work top-down.

## State entering this session (✅ done)

**Forward ideal age CONVERGED for both experiments** (`ReturnCode.Success`):

| exp | `EXPERIMENT` | mean steady age | converged file |
|---|---|---:|---|
| wthp  | `01deg_jra55v13_ryf9091_qian_wthp`  | 921.1 yr | `…/periodic/{MC}/1x4/NK_Q4x4/age_Pardiso_Q4x4.jld2` |
| wthmp | `01deg_jra55v13_ryf9091_qian_wthmp` | 990.7 yr | (same path under its EXP) |

`MC = cgridtransports_wparent_upwind3_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS`.
Preprocessing/velocities/avg-matrix all clean (periodicaverage corruption fixed &
verified). **Meltwater signal so far: wthmp − wthp = +69.6 yr** global-mean age.

Shared env for every command below:
```bash
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg
```
`EXPERIMENT ∈ {01deg_jra55v13_ryf9091_qian_wthp, …_qian_wthmp}`.

---

## Phase 1 — Forward post-NK diagnostics + difference plots (ready NOW)

Needs only the converged forward ages (present). Per experiment:

```bash
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
PLOT_NK_QUEUE=hugemem PLOT_NK_MEM=512GB PLOT_NK_NCPUS=18 WALLTIME_PLOT_NK=12:00:00 \
JOB_CHAIN=run1yrNK-combine1yr-ventilation-plotNK-plotventilation \
  bash scripts/driver.sh
```
- **Omit `plotNKtrace`** (broken; aborts the chain under `set -e`).
- `plotNK` is heavy at 0.1° → hugemem/512 GB/long walltime (as set above).
- `run1yrNK`→`combine1yr` produces the 1-yr periodic FTS `age_periodic_1year.jld2`
  that both `ventilation` and the difference plot read.

Then the **`wthmp − wthp` forward-age difference** (after both experiments' run1yrNK):
```bash
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 PARTITION=1x4 \
qsub -v PARENT_MODEL,TIME_WINDOW,GRID_HZ,ADVECTION_SCHEME,PARTITION \
  scripts/plotting/plot_Li_etal_meltwater_diff.sh
```
(A=wthp, B=wthmp → diff panels are B−A = meltwater effect. `MODEL_CONFIG` from the
env selects the forward tree.)

---

## Phase 2 — TRAF adjoint age (Task G) — the untested piece

TRAF at OM2-01 has **never been run** — the Task B avg-matrix enabling is in place
(`solve_periodic_NK.jl` + `create_matrix.jl`, commit `70d97b2`) but unexercised at
0.1°. Expect to debug. The `invVMtV = V⁻¹ Mᵀ V` synthesis reads the (now clean)
`upwind1 avg` forward `M.jld2`.

Per experiment:
```bash
# TMbuild short-circuits to invVMtV synthesis; then NK solves the adjoint.
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=2040-2050 GRID_HZ=4 TRAF=yes TRAF_TM_SOURCE=invVMtV \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg INITIAL_AGE=0 \
JOB_CHAIN=TMbuild-NK bash scripts/driver.sh

# restart on walltime (multi-restart, like the forward solve):
… INITIAL_AGE=latest JOB_CHAIN=NK bash scripts/driver.sh
```
Notes / likely gotchas:
- **`GMRES_RTOL=1e-3`** will probably be needed (as for forward wthp) if an adjoint
  Newton iteration's GMRES exceeds one 48 h walltime — watch for "no new iterate
  saved" and add it.
- TRAF stability: at OM2-025 the adjoint blew up in the fold region until Δt was
  reduced (see `docs/TRAF_simulations.md` §3f). If Φ! NaNs at high latitude, drop
  `TIMESTEP_MULT`.
- Outputs land under the `…_traf` MODEL_CONFIG; adjoint age = "time to re-emergence".

Then TRAF post-NK + difference (same as Phase 1 but add `TRAF=yes`):
```bash
… TRAF=yes JOB_CHAIN=run1yrNK-combine1yr-ventilation-plotNK-plotventilation bash scripts/driver.sh
… TRAF=yes qsub -v …,TRAF scripts/plotting/plot_Li_etal_meltwater_diff.sh   # adjoint diff
```

---

## Deliverables (goal of the whole plan)

1. Equilibrated **ideal age** maps per experiment + **`wthmp − wthp`** diff. *(forward
   solves done; plots = Phase 1.)*
2. **Time to re-emergence** (TRAF adjoint) per experiment + diff. *(Phase 2.)*
3. **Surface ventilation** 𝒱ꜜ (forward × adjoint) per experiment + diff.
   *(needs both; `plot_Li_etal_meltwater_diff.jl` currently does age/adjoint-age only
   — add a ventilation-diff mode, or diff the `ventilation` outputs.)*

## Housekeeping
- Keep §7 of `Li-etal_simulations.md` updated with job IDs / drifts as runs land.
- `bash scripts/runs/reconcile_submissions.sh` while finished jobs are still in
  `qstat -x` (7-day retention) to backfill their PBS-side columns.
- Use `driver.sh` for all submissions; `--color=never` on `ls`/`grep` (CLAUDE.md).
