# Agent Context

## Project directory tree

```
ACCESS-OM2_x_Oceananigans/
├── src/
│   ├── setup_model.jl            # Shared model setup (include'd by run/solve scripts)
│   ├── run_1year.jl              # Standalone 1-year age simulation → outputs/{model}/age/
│   ├── run_10years.jl            # Standalone 10-year age simulation → outputs/{model}/age/
│   ├── run_100years.jl           # Standalone 100-year age simulation → outputs/{model}/age/
│   ├── solve_periodic_NK.jl      # Newton-GMRES periodic steady-state solver
│   ├── periodic_solver_common.jl # Shared solver infrastructure (simulation, wet mask, Φ!, G!)
│   ├── periodicaverage.py         # Python preprocessing: monthly climatologies + yearly averages from ACCESS-OM2 output
│   ├── create_grid.jl            # Build tripolar grid → preprocessed_inputs/{model}/{experiment}/grid.jld2
│   ├── create_velocities.jl      # Preprocess MOM velocities → *_monthly.jld2 + *_yearly.jld2
│   ├── run_1year_benchmark.jl    # Benchmark 1-year run (no output writers, precompile + time)
│   ├── run_diagnostic_steps.jl  # 10-step diagnostic run saving every step (serial/distributed)
│   ├── compare_runs_across_architectures.jl  # Compare serial vs distributed age output
│   ├── test_distributed_halo_fill.jl  # MWE: fill_halo_regions! at all staggered locations
│   ├── create_closures.jl        # (WIP — not yet used in pipeline)
│   ├── create_matrix.jl          # Build transport matrix → outputs/{model}/matrices/
│   ├── solve_matrix_age.jl      # Solve steady-state age from saved matrix M (CPU-only)
│   ├── plot_standardrun_age.jl   # Plot age diagnostics from standard run output (DURATION env var, CPU-only)
│   ├── plot_outputs.jl           # Plot u/v/w/η outputs from simulation (standalone, CPU-only)
│   ├── debug_jacobian_symmetry.jl # Debug script for Jacobian structural symmetry
│   └── shared_functions.jl      # load_project_config(), load_tripolar_grid(), compute_wet_mask(),
│                                  # setup_age_simulation(), validate_age_field(), process_sparse_matrix(),
│                                  # compute_and_save_coarsening(), plot_age_diagnostics(), etc.
├── model_configs/
│   ├── ACCESS-OM2-1.sh              # Model-specific config (walltimes, MODEL_SHORT)
│   └── ACCESS-OM2-025.sh            # Model-specific config (walltimes, MODEL_SHORT)
├── scripts/
│   ├── env_defaults.sh              # Common env var defaults (sourced by all job scripts)
│   ├── driver.sh                    # Unified pipeline driver (PARENT_MODEL, JOB_CHAIN)
│   ├── test_driver.sh               # Test/diagnostic driver (halofill, diag, mpi)
│   ├── prepreprocessing/
│   │   └── periodicaverage.sh       # PBS: Python preprocessing (monthly climatologies + yearly averages)
│   ├── preprocessing/
│   │   ├── build_grid.sh            # PBS: grid build (CPU, express)
│   │   ├── build_velocities.sh      # PBS: velocity preprocessing (CPU/GPU, express)
│   │   ├── build_TMconst.sh         # PBS: Jacobian build from constant fields (CPU, normal)
│   │   └── build_TMavg.sh           # PBS: snapshot + average matrices (CPU, normal)
│   ├── standard_runs/
│   │   ├── run_1year.sh             # PBS: 1-year GPU simulation
│   │   ├── run_10years.sh           # PBS: 10-year GPU simulation
│   │   ├── run_100years.sh          # PBS: 100-year GPU simulation
│   │   ├── run_long.sh              # PBS: long GPU simulation (NYEARS env var)
│   │   └── run_1year_from_periodic_sol.sh  # PBS: 1-year run from periodic solution
│   ├── solvers/
│   │   ├── solve_periodic_NK.sh     # PBS: Newton-GMRES GPU solver
│   │   ├── solve_TM_age_CPU.sh      # PBS: solve age from matrix (CPU, Pardiso/ParU/UMFPACK)
│   │   └── solve_TM_age_GPU.sh      # PBS: solve age from matrix (GPU, CUDSS)
│   ├── plotting/
│   │   ├── plot_standardrun_age.sh  # PBS: plot standard run age diagnostics (CPU, DURATION env var)
│   │   └── plot_1year_from_periodic_sol.sh  # PBS: plot periodic solution diagnostics (CPU)
│   └── tests/                         # Test PBS wrappers (used by test_driver.sh)
│       ├── run_halofill_test.sh       # fill_halo_regions! MWE on distributed GPU
│       ├── run_diagnostic_steps.sh    # 10-step diagnostic run
│       └── run_mpi_test.sh            # MPI smoke test
├── test/                                     # Julia test scripts (matrix regression)
├── archive/scripts/                       # Archived/obsolete PBS scripts
├── preprocessed_inputs/{parentmodel}/  # symlink → /scratch/y99/TMIP/…/preprocessed_inputs/
│   └── {experiment}/
│       ├── grid.jld2                   # tripolar grid (shared across time windows)
│       └── {time_window}/
│           ├── monthly/
│           │   ├── u_interpolated_monthly.jld2    # B-grid → C-grid interpolated, monthly FTS
│           │   ├── u_from_mass_transport_monthly.jld2
│           │   ├── (v_*, w_*, eta_* analogues)
│           │   ├── *_monthly.nc         # NetCDF climatologies from periodicaverage.py
│           │   └── plots/               # diagnostic plots from create_velocities.jl
│           └── yearly/
│               ├── u_interpolated_yearly.jld2     # time-averaged constant Field
│               ├── u_from_mass_transport_yearly.jld2
│               ├── (v_*, w_*, eta_* analogues)
│               └── *_yearly.nc          # NetCDF yearly averages from periodicaverage.py
├── outputs/{parentmodel}/              # symlink → /scratch/y99/TMIP/…/outputs/
│   └── {experiment}/{time_window}/
│       ├── age/{model_config}/            # offline simulation outputs (model_config = VS_WF_AS_TS)
│       └── matrices/{model_config}/       # Jacobian M.jld2, steady_age_*.jld2, plots/
├── logs/                               # symlink → /scratch/y99/TMIP/…/logs/
│   ├── PBS/                            # PBS scheduler stdout/stderr (set via #PBS -o/-e)
│   ├── python/{parentmodel}/{experiment}/{time_window}/  # Python preprocessing logs
│   └── julia/{parentmodel}/{experiment}/
│       ├── preprocess/                 # grid + velocity preprocessing logs
│       └── {time_window}/              # per-time-window Julia logs (runs, solvers, plots, etc.)
├── Project.toml
├── LocalPreferences.toml               # CUDA version pin (local = true, version = "12.9")
└── AGENTS.md
```

## Julia depot
If you need to check Oceananigans code or any package loaded in this project, these are installed in `JULIA_DEPOT_PATH=/g/data/y99/bp3051/.julia/`.

## NetCDF inspection before coding
Use this quick check before changing data-loading, grid, or velocity scripts.

```bash
module load netcdf
ncdump -h xxx.nc
```

Verify variable names, dimension ordering, units, and missing-value conventions from the header output.
For ACCESS-OM2 periodic inputs, this helps avoid index-order mistakes (for example `month, z, y, x` vs `x, y, z`) before implementation.


## Submitting and monitoring PBS jobs

Submit via the unified driver (JOB_CHAIN is required):
```bash
JOB_CHAIN=full bash scripts/driver.sh                                    # full OM2-1 pipeline
PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=preprocessing-run1yr bash scripts/driver.sh
EXPERIMENT=1deg_jra55_ryf9091_gadi TIME_WINDOW=1958-1987 JOB_CHAIN=full bash scripts/driver.sh
JOB_CHAIN=vel..NK bash scripts/driver.sh                                 # range: vel through NK
JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh                          # single run + plot
GPU_RESOURCES=gpuvolta JOB_CHAIN=NK bash scripts/driver.sh               # on Volta GPUs
```

JOB_CHAIN steps: `prep grid vel run1yr run1yrfast run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM plot1yr plot10yr plot100yr`
Shortcuts: `preprocessing` (= `prep-grid-vel`) `standardruns` `TMall` `plotall` `full`
Range notation: `A..B` follows the dependency DAG (e.g., `run1yrNK..plotNK` = `run1yrNK-plotNK`)
TM_SOURCE: `const` (default), `avg`, or `both` — filters TMsolve/NK/run1yrNK branches

DAG dependencies: `prep → vel`, `grid → vel`, `vel → run*`, `vel → TMbuild`, etc.

Tests use a separate driver:
```bash
GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill bash scripts/test_driver.sh
PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diag bash scripts/test_driver.sh
```
Test steps: `halofill` (halo fill MWE), `diag` (10-step diagnostic), `mpi` (MPI smoke test)

### PBS Pro monitoring

This project runs on NCI Gadi, which uses PBS Pro as its job scheduler. Use `/qstat` (custom skill) or the commands below.

Common commands:
```bash
qstat -u bp3051          # list all your running/queued jobs
qstat -f <job_id>        # detailed status for one job
qstat -x <job_id>        # include finished jobs (history)
qdel <job_id>            # cancel a job
```

PBS job states: `Q` = queued, `R` = running, `H` = held, `E` = exiting, `F` = finished.

Job names in this project follow the pattern `{MODEL_SHORT}_{step}` (e.g., `OM21_run1yr`, `OM2025_NK`), set by `driver.sh` via `#PBS -N`.

Log locations after a job completes:
- PBS scheduler logs: `logs/PBS/` (stdout/stderr from `#PBS -o/-e`)
- Python preprocessing logs: `logs/python/{PM}/{EXP}/{TW}/`
- Julia script logs: `logs/julia/{PM}/{EXP}/` (grid/vel in `preprocess/`, others under `{TW}/`)

## Naming conventions

### Preprocessed velocity files (in `preprocessed_inputs/{PM}/{EXP}/{TW}/`)
- Monthly `FieldTimeSeries`: `monthly/*_monthly.jld2` (e.g. `u_from_mass_transport_monthly.jld2`)
- Time-averaged constant `Field`: `yearly/*_yearly.jld2` (e.g. `u_from_mass_transport_yearly.jld2`)
- Grid file: `preprocessed_inputs/{PM}/{EXP}/grid.jld2` (shared across time windows)
- Full file list (under `monthly/` and `yearly/` respectively):
  - `u_interpolated_monthly.jld2` / `u_interpolated_yearly.jld2`
  - `v_interpolated_monthly.jld2` / `v_interpolated_yearly.jld2`
  - `w_monthly.jld2` / `w_yearly.jld2`
  - `u_from_mass_transport_monthly.jld2` / `u_from_mass_transport_yearly.jld2`
  - `v_from_mass_transport_monthly.jld2` / `v_from_mass_transport_yearly.jld2`
  - `w_from_mass_transport_monthly.jld2` / `w_from_mass_transport_yearly.jld2`
  - `eta_monthly.jld2` / `eta_yearly.jld2`

### Output directories (unified 4-part tag: `{VS}_{WF}_{AS}_{TS}` = `model_config`)
- Age simulation outputs: `outputs/{PM}/{EXP}/{TW}/age/{model_config}/`
- Matrix outputs: `outputs/{PM}/{EXP}/{TW}/matrices/{model_config}/`
- Matrix plots: `outputs/{PM}/{EXP}/{TW}/matrices/{model_config}/plots/`

### Log directories
- `MODEL_CONFIG` = `{VELOCITY_SOURCE}_{W_FORMULATION}_{ADVECTION_SCHEME}_{TIMESTEPPER}`
- Python preprocessing logs: `logs/python/{PM}/{EXP}/{TW}/`
- Grid/velocity logs: `logs/julia/{PM}/{EXP}/preprocess/`
- Per-time-window Julia logs: `logs/julia/{PM}/{EXP}/{TW}/...` (runs, solvers, plots, TM, etc.)

## Script overview
| Script | Purpose | Key env vars |
|--------|---------|-------------|
| `src/setup_model.jl` | Shared model setup (include'd by run/solve scripts) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/run_1year.jl` | Standalone 1-year age simulation | (inherits from setup_model.jl) |
| `src/run_10years.jl` | Standalone 10-year age simulation | (inherits from setup_model.jl) |
| `src/run_100years.jl` | Standalone 100-year age simulation | (inherits from setup_model.jl) |
| `src/solve_periodic_NK.jl` | Newton-GMRES periodic steady-state solver | JVP_METHOD, LINEAR_SOLVER, LUMP_AND_SPRAY |
| `src/periodicaverage.py` | Python preprocessing: monthly climatologies + yearly averages from ACCESS-OM2 output | PARENT_MODEL, EXPERIMENT, TIME_WINDOW |
| `src/create_grid.jl` | Build and save the tripolar grid | PARENT_MODEL, EXPERIMENT |
| `src/create_velocities.jl` | Preprocess MOM velocities → monthly FTS + yearly Fields | PARENT_MODEL, EXPERIMENT, TIME_WINDOW |
| `src/plot_outputs.jl` | Plot u/v/w/η outputs from simulation (standalone, CPU-only) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/create_matrix.jl` | Build transport matrix from constant fields | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/create_snapshot_matrices.jl` | Build snapshot Jacobians + inline averages from 1-year velocity snapshots | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/average_snapshot_matrices.jl` | Re-average snapshot matrices from saved files (standalone) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/solve_matrix_age.jl` | Solve steady-state age from saved matrix M (CPU-only) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, LINEAR_SOLVER, LUMP_AND_SPRAY |
| `test/check_snapshot_matrices.jl` | Regression test: compare snapshot/averaged matrices against archive | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |

## PBS scripts

All job scripts are model-agnostic — `PARENT_MODEL` selects the model.
Model-specific config (walltimes, PBS name prefix) lives in `model_configs/{PARENT_MODEL}.sh`.
The unified `scripts/driver.sh` is the single interface for submitting jobs.

- `scripts/env_defaults.sh` — common env var defaults (sourced by all job scripts); sources model config
- `scripts/driver.sh` — unified pipeline driver (PARENT_MODEL, JOB_CHAIN, GPU_RESOURCES)
- `scripts/preprocessing/build_grid.sh` — grid build (CPU, express)
- `scripts/preprocessing/build_velocities.sh` — velocity preprocessing (CPU/GPU, express)
- `scripts/preprocessing/build_TMconst.sh` — Jacobian build from constant fields (CPU, 48 CPU, 192 GB, normal)
- `scripts/preprocessing/build_TMavg.sh` — snapshot + average matrices (CPU, normal)
- `scripts/standard_runs/run_1year.sh` — 1-year GPU simulation
- `scripts/standard_runs/run_10years.sh` — 10-year GPU simulation
- `scripts/standard_runs/run_100years.sh` — 100-year GPU simulation
- `scripts/standard_runs/run_long.sh` — long GPU simulation (NYEARS env var)
- `scripts/standard_runs/run_1year_from_periodic_sol.sh` — 1-year run from periodic solution
- `scripts/solvers/solve_periodic_NK.sh` — Newton-GMRES GPU solver
- `scripts/solvers/solve_TM_age_CPU.sh` — solve age from matrix, CPU (Pardiso/ParU/UMFPACK)
- `scripts/solvers/solve_TM_age_GPU.sh` — solve age from matrix, GPU (CUDSS)
- `scripts/plotting/plot_standardrun_age.sh` — plot standard run age diagnostics (CPU, DURATION env var)
- `scripts/plotting/plot_1year_from_periodic_sol.sh` — plot periodic solution diagnostics (CPU)
- `scripts/benchmarks/submit_all_gpu_job_modes.sh` — batch submit 1yr runs across config combos
- `scripts/benchmarks/submit_all_matrix_jobs.sh` — batch submit matrix build jobs
- `scripts/benchmarks/submit_all_solve_matrix_age.sh` — batch submit TM age solver combos (14 jobs)
- `scripts/benchmarks/submit_all_solver_modes.sh` — batch submit NK solver variants
- `scripts/maintenance/pkg_update_project.sh` — update Julia packages (login node)
- `scripts/maintenance/pkg_instantiate_project_CPU.sh` — precompile on CPU compute node
- `scripts/maintenance/pkg_instantiate_project_GPU.sh` — precompile on GPU compute node
- `scripts/maintenance/setup_mpitrampoline.sh` — one-time MPI setup
- `scripts/maintenance/archive.sh` — copy outputs to archive storage
- `scripts/debugging/check_snapshot_matrices_job.sh` — regression test: check snapshot matrices
- `scripts/debugging/test_mpi.sh` — MPI connectivity test
- `scripts/prepreprocessing/periodicaverage.sh` — PBS: Python preprocessing (monthly climatologies + yearly averages)
- `scripts/prepreprocessing/write_ACCESS-OM2_configs.sh` — write ACCESS-OM2 intake catalog configs (utility)

## Configuration environment variables

### Experiment/time-window variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXPERIMENT` | Intake catalog key for ACCESS-OM2 experiment | `1deg_jra55_iaf_omip2_cycle6` (OM2-1) or `025deg_jra55_iaf_omip2_cycle6` (OM2-025) |
| `TIME_WINDOW` | Year range `YYYY-YYYY` or single year `YYYY` | `1960-1979` |

### Model config variables

The 4 core config env vars are parsed by `parse_config_env()` in `shared_functions.jl`:

| Variable | Valid values | Default |
|----------|-------------|---------|
| `VELOCITY_SOURCE` | `cgridtransports`, `bgridvelocities` | `cgridtransports` |
| `W_FORMULATION` | `wdiagnosed`, `wprescribed` | `wdiagnosed` |
| `ADVECTION_SCHEME` | `centered2`, `weno3`, `weno5` | `centered2` |
| `TIMESTEPPER` | `AB2`, `SRK2`, `SRK3`, `SRK4`, `SRK5` | `AB2` |

- `AB2` = `:QuasiAdamsBashforth2` (Oceananigans default)
- `SRK{N}` = `:SplitRungeKutta{N}` (N = 2..5 stages)

Shell defaults are set in `scripts/env_defaults.sh`, which is sourced by all PBS job scripts.
The combined tag `MODEL_CONFIG = {VS}_{WF}_{AS}_{TS}` determines output directory paths and log filenames.

### GPU queue configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_QUEUE` | `gpuhopper` | GPU queue (`gpuhopper` for H200, `gpuvolta` for V100) |

Memory is auto-set by the driver: 256GB for gpuhopper, 96GB for gpuvolta.

### Newton solver variables (`solve_periodic_NK.jl`, `solve_matrix_age.jl`)

| Variable | Default | Description |
|----------|---------|-------------|
| `JVP_METHOD` | `matrix` | JVP method (`matrix`, `finitediff`, or `exact`); Newton solver only |
| `LINEAR_SOLVER` | `Pardiso` | Direct solver for preconditioner (`Pardiso`, `ParU`, or `UMFPACK`) |
| `LUMP_AND_SPRAY` | `no` | Lump-and-spray coarsening for preconditioner (`yes`/`no`) |

- Output filename tags: `Pardiso`/`ParU`/`UMFPACK` for LINEAR_SOLVER; `LSprec`/`prec` for LUMP_AND_SPRAY
- Example: `age_newton_Pardiso_prec.jld2`, `steady_age_full_ParU_LSprec.jld2`, `steady_age_full_UMFPACK_prec.jld2`

### Anderson solver variables (archived — `solve_periodic_AA.jl` in `archive/`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AA_SOLVER` | `SpeedMapping` | Solver backend: `SpeedMapping`, `NLsolve`, `SIAMFANL`, `FixedPoint` |
| `AA_M` | `40` | Anderson history size (used by NLsolve, SIAMFANL, FixedPoint) |
| `NLSAA_BETA` | `1.0` | Anderson damping parameter (try 0.5 for slow convergence) |
| `SMAA_SIGMA_MIN` | `0.0` | SpeedMapping minimum σ; setting to 1 may avoid stalling |
| `SMAA_STABILIZE` | `no` | Stabilization mapping before extrapolation (`yes`/`no`) |
| `SMAA_CHECK_OBJ` | `no` | Restart at best past iterate on NaN/Inf (`yes`/`no`) |
| `SMAA_ORDERS` | `332` | Alternating order sequence (each digit 1–3) |

## Multi-GPU (MPI) runs

Multi-GPU runs use `mpiexec` with socket binding flags:
```bash
mpiexec --bind-to socket --map-by socket -n $NGPUS julia --project ...
```

**Why `--bind-to socket --map-by socket`:** Gadi's default behaviour assigns MPI ranks to CPU sockets randomly. Since each GPU is physically attached to a specific CPU socket, random assignment means a CPU may be bound to a GPU on a different socket, making CPU-GPU communication cross the inter-socket link and become extremely slow. Socket binding ensures each MPI rank runs on the CPU socket directly connected to its GPU, giving the fastest possible CPU-GPU data path.

Required modules for MPI jobs: `cuda/12.9.0` + `openmpi/5.0.8`.

## Key design decisions
- Model setup is shared via `setup_model.jl` (include'd by downstream scripts)
- `setup_model.jl` creates the model but NOT the simulation — each downstream script creates its own
- Matrix build always on CPU (sparsity detection/coloring incompatible with GPU)
- `periodicaverage.py` (Python prep step) computes monthly climatologies and yearly averages from ACCESS-OM2 output NetCDF files, parameterized by `EXPERIMENT` and `TIME_WINDOW`
- `create_matrix.jl` uses time-averaged yearly Fields (not FieldTimeSeries) → single Jacobian call
- Simulation scripts use monthly FieldTimeSeries (12 monthly snapshots, Cyclical indexing)
- Yearly fields for matrix build use same BCs (`FPivotZipperBoundaryCondition`) as the per-month fields in `create_velocities.jl`
- zstar initialisation before Jacobian: `_update_zstar_scaling!(η_constant, grid)`
- Age solving is factored out into `solve_matrix_age.jl` (runs on saved M.jld2)
- Newton solver loads M from `create_matrix.jl` output; LUMP_AND_SPRAY controls preconditioner coarsening
- LINEAR_SOLVER selects direct solver: Pardiso (MKL), ParU (SuiteSparse parallel LU), or UMFPACK (SuiteSparse serial LU)
- Newton solver uses exact JVP via linear tracer, matrix-based JVP via `stop_time * M`, or finite-diff via `AutoFiniteDiff()`
- GPU arrays preallocated once; `copyto!` used for CPU↔GPU transfer in G!

## Periodic solver

The intent of this projetc is to solve for the equilibrium state of a tracer embedded in a yearly periodic circulation. The circulation is prescribed from monthly climatologies of velocities and other tracers and grid coordinates and metrics from archived outputs of the ACCESS-OM2 model. The goal is to find this equilibrium state for ventilation tracers like the water age. The key is that instead of time-stepping the tracer for ~3000 years, we only time-step one year at a time and wrap this one year simulation into a more efficient solver that accelerates the convergence of our state towards the equilibrium. Hopefully this should only take ~40 to ~400 simulation years.

## The maths

That is, if ϕ is the mapping that advances tracer x by Δt = 1 year

ϕ(x(t)) = x(t + Δt)

then we want to find the solution to

ϕ(x) = x

This is a fixed-point iteration and can be solved, e.g., with Anderson Acceleration. It can also be recast as finding the zero of G where

G(x) = ϕ(x) - x

for which nonlinear solvers can be used, such as Newton's method.

This code base explores different algorithms to solve this problem.

The main problem is that the 1-year simulations take time. But the core idea of this project to solve this problem is that we can run the 1-year simulations "offline" with Oceananigans on GPUs, which should be very fast.

## References

For technical references, see REFERENCES.md
