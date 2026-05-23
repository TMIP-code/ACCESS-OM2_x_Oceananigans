# Newton-Krylov restart tests

Documents the test setup that exercises the checkpoint/restart machinery
added in the "Checkpoint & restart for the Newton-Krylov periodic solver"
work. Two test drivers submit a pair of NK jobs each: A1 runs from
`INITIAL_AGE=0` with `NK_MAXITERS=3` and saves Newton iterates; A2 chains
`afterok` on A1 with `INITIAL_AGE=latest` and `NK_MAXITERS=2`.

## Test drivers

| Script | Layout | Submit mode | Purpose |
|---|---|---|---|
| [test/run_NK_restart_test_serial.sh](../test/run_NK_restart_test_serial.sh) | `PARTITION=1x1` | `--gpu-single` | Smoke-test restart on a single GPU |
| [test/run_NK_restart_test_1x2.sh](../test/run_NK_restart_test_1x2.sh)       | `PARTITION=1x2` | `--gpu`        | Same restart, but partitioned-NK (rank-0-only save) |

Both scripts share the same config:

```
PARENT_MODEL          = ACCESS-OM2-1
TIME_WINDOW           = 1968-1977
VELOCITY_SOURCE       = totaltransport
W_FORMULATION         = wprescribed
PRESCRIBED_W_SOURCE   = parent
TIMESTEPPER           = AB2
TIMESTEP_MULT         = 4
MONTHLY_KAPPAV        = yes
LUMP_AND_SPRAY        = yes
MATRIX_PROCESSING     = symdrop
LINEAR_SOLVER         = Pardiso
GPU_QUEUE             = gpuvolta
TRACE_SOLVER_HISTORY  = yes
TM_SOURCE             = const
TM_MODEL_CONFIG       = totaltransport_wdiagnosed_centered2_AB2_mkappaV_DTx4
```

The NK `model_config` resolves to `totaltransport_wparent_centered2_AB2_mkappaV_DTx4`
(no matrix on disk). `TM_MODEL_CONFIG` redirects the preconditioner load to
the existing `wdiagnosed` matrix at
[outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/TM/totaltransport_wdiagnosed_centered2_AB2_mkappaV_DTx4/const/M.jld2](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/TM/totaltransport_wdiagnosed_centered2_AB2_mkappaV_DTx4/const/M.jld2).
Only the `const` subdirectory contains an `M.jld2` for this experiment, so
both test scripts set `TM_SOURCE=const` explicitly (overriding the
`avg` default in `scripts/env_defaults.sh`).

`MATRIX_PROCESSING=symdrop` ensures the preconditioner is structurally
symmetric (required by Pardiso REAL_SYM).

## Running

```bash
bash test/run_NK_restart_test_serial.sh
bash test/run_NK_restart_test_1x2.sh
```

Each script prints the two PBS job IDs (A1, A2). Monitor with `qstat <A1> <A2>`.

## Expected log paths

Logs land under
`logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/NK/`
(serial) and the same path (1×2 — the partition tag does not appear here
because the log directory tree in `solve_periodic_NK.sh` is partition-flat).
The solver output directory for the iterate files differs:

- Serial: `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/totaltransport_wparent_centered2_AB2_mkappaV_DTx4/NK/`
- 1×2:    `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/totaltransport_wparent_centered2_AB2_mkappaV_DTx4/1x2/NK/`

## Verification checklist

After both jobs in a driver complete, the corresponding log files and
`solver_output_dir` should satisfy:

1. `grep "Saved Newton iterate" <A1_log>` → ≥ 1 line. NK_MAXITERS=3
   produced 5 saves (NN=01..05) in the serial run — NewtonRaphson calls G!
   more than once per "iteration" (likely line-search re-evaluations), so
   the count > MAXITERS−1 is expected.
2. `grep "Saved Newton iterate" <A2_log>` → ≥ 1 line (in the serial run,
   A2 with NK_MAXITERS=2 produced 3 saves, overwriting NN=01..03 from A1).
3. `grep "INITIAL_AGE=latest" <A2_log>` → "resolved to …
   `newton_iterate_NN.jld2`" where NN is the highest written by A1.
4. `grep "Newton-GMRES solve complete" <A2_log>` → `retcode=...
   total_Φ_calls=… total_G_calls=… total_jvp_calls=…`; the line should
   satisfy `total_Φ_calls == total_G_calls + total_jvp_calls`.
5. `grep "NK using TM from a different model_config" <A1_log> <A2_log>` →
   present in both (tm=wdiagnosed, nk=wparent).
6. JLD2 inspection:
   ```julia
   using JLD2
   v = load(".../newton_iterate_01.jld2", "age")
   @show typeof(v) length(v) extrema(v) ./ 3.156e7
   ```
   should report `Vector{Float64}`, `length == Nidx_global`, and a range
   from a few negative tens to a few thousand years (the negative tail is
   normal for under-converged NK iterates).
7. (Optional) Submit a one-off NK job with `INITIAL_AGE=<fixture.jld2>`
   where the fixture has `max ≈ 1e15` s; load errors loudly on the 10000 yr
   threshold (`load_initial_age` in
   [src/periodic_solver_common.jl](../src/periodic_solver_common.jl)).

## Test runs

> _Populate as runs happen. Each entry should record submit time, A1/A2
> job IDs, outcome of each verification step, and any deviation from
> expectations._

| Date | Driver | A1 jobid | A2 jobid | Checks 1–6 | Notes |
|---|---|---|---|---|---|
| 2026-05-22 | serial | 169075809.gadi-pbs | 169075810.gadi-pbs | ❌ failed | A1 exit=1: `No file exists at given path: .../avg/M.jld2`. TM_SOURCE defaulted to `avg`; A2 cancelled (afterok). Submitted from main @ 0469008. |
| 2026-05-22 | 1×2    | 169075833.gadi-pbs | 169075834.gadi-pbs | ❌ failed | Same failure mode as serial. Submitted from main @ 0469008. |
| 2026-05-23 | serial | 169088807.gadi-pbs | 169088808.gadi-pbs | ✅ pass (1, 2, 3, 4, 5, 6) | A1 exit=0 (18:12, 5 saves NN=01..05); A2 exit=0 (08:29, resolved to newton_iterate_05.jld2, 3 saves). JLD2 inspection: Vector{Float64}, length 2 707 869, extrema in years ≈ (−143, 2371). |
| 2026-05-23 | 1×2    | 169088809.gadi-pbs | 169088810.gadi-pbs | ❌ failed (unrelated) | A1 exit=1 (05:23) with "Partition halo-size mismatch": pre-partitioned 1×2 FTS files on disk were built with halo=13 but env_defaults.sh uses GRID_HX/HY=7. Rebuild via `PARTITION=1x2 JOB_CHAIN=partition bash scripts/driver.sh` (or match halos). Not a bug in the restart/units work — serial pair validates the restart machinery; the 1×2 pair would additionally validate that `save_newton_iterate!` is rank-0-only. |
