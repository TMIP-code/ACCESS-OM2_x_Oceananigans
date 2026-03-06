# Agent Context

## Project directory tree

```
ACCESS-OM2_x_Oceananigans/
├── src/
│   ├── setup_model.jl            # Shared model setup (include'd by run/solve scripts)
│   ├── run_1year.jl              # Standalone 1-year age simulation → outputs/{model}/age/
│   ├── run_10years.jl            # Standalone 10-year age simulation → outputs/{model}/age/
│   ├── run_100years.jl           # Standalone 100-year age simulation → outputs/{model}/age/
│   ├── solve_periodic_newton.jl  # Newton-GMRES periodic steady-state solver
│   ├── solve_periodic_anderson.jl # Anderson/SpeedMapping periodic solver
│   ├── create_grid.jl            # Build tripolar grid → preprocessed_inputs/{model}/{model}_grid.jld2
│   ├── create_velocities.jl      # Preprocess MOM velocities → *_periodic.jld2 + *_constant.jld2
│   ├── create_closures.jl        # (WIP — not yet used in pipeline)
│   ├── create_matrix.jl          # Build transport matrix → outputs/{model}/matrices/
│   ├── solve_matrix_age.jl      # Solve steady-state age from saved matrix M (CPU-only)
│   ├── plot_1year_age.jl         # Plot age diagnostics from 1-year output (standalone, CPU-only)
│   ├── plot_10years_age.jl       # Plot age diagnostics from 10-year output (standalone, CPU-only)
│   ├── plot_100years_age.jl      # Plot age diagnostics from 100-year output (standalone, CPU-only)
│   ├── plot_outputs.jl           # Plot u/v/w/η outputs from simulation (standalone, CPU-only)
│   ├── debug_jacobian_symmetry.jl # Debug script for Jacobian structural symmetry
│   └── shared_functions.jl      # load_tripolar_grid(), compute_wet_mask(), plot_age_diagnostics(), etc.
├── scripts/
│   ├── ACCESS-OM2-1_preprocess_job.sh   # PBS: grid + velocities (12 CPU, 47 GB, express)
│   ├── ACCESS-OM2-1_CPU_job.sh          # PBS: offline simulation CPU (12 CPU, 47 GB, express)
│   ├── ACCESS-OM2-1_GPU_job.sh          # PBS: GPU job (1 GPU, 47 GB, gpuvolta) — NONLINEAR_SOLVER selects script
│   ├── ACCESS-OM2-1_plot_1year_age_job.sh  # PBS: CPU plot job for 1-year age diagnostics
│   ├── ACCESS-OM2-1_plot_10years_age_job.sh # PBS: CPU plot job for 10-year age diagnostics
│   ├── ACCESS-OM2-1_plot_100years_age_job.sh # PBS: CPU plot job for 100-year age diagnostics
│   ├── ACCESS-OM2-1_matrix_job.sh       # PBS: matrix build CPU (48 CPU, 190 GB, normal)
│   ├── ACCESS-OM2-1_solve_matrix_age_job.sh # PBS: solve steady-state age from matrix (12 CPU, 47 GB, normal)
│   ├── submit_all_gpu_job_modes.sh      # Submit all VELOCITY_SOURCE × W_FORMULATION combinations
│   ├── pkg_instantiate_project_CPU.sh
│   └── pkg_instantiate_project_GPU.sh
├── preprocessed_inputs/{parentmodel}/  # symlink → /scratch/y99/TMIP/…/preprocessed_inputs/
│   ├── {parentmodel}_grid.jld2
│   ├── u_interpolated_periodic.jld2    # B-grid → C-grid interpolated, monthly FTS
│   ├── u_interpolated_constant.jld2    # time-averaged constant Field
│   ├── u_from_mass_transport_periodic.jld2
│   ├── u_from_mass_transport_constant.jld2
│   ├── (v_*, w_*, eta_* analogues)
│   └── plots/                          # diagnostic plots from create_velocities.jl
├── outputs/{parentmodel}/              # symlink → /scratch/y99/TMIP/…/outputs/
│   ├── age/{model_config}/            # offline simulation outputs (model_config = VS_WF_AS_TS)
│   └── matrices/{model_config}/       # Jacobian M.jld2, steady_age_*.jld2, plots/
├── logs/                               # symlink → /scratch/y99/TMIP/…/logs/
│   ├── PBS/                            # PBS scheduler stdout/stderr (set via #PBS -o/-e)
│   └── julia/                          # Julia script stdout/stderr, organised by script name
│       ├── create_grid/
│       │   └── create_grid_{parentmodel}_{job_id}.{out,err}
│       ├── create_velocities/
│       │   └── create_velocities_{parentmodel}_{job_id}.{out,err}
│       ├── run_ACCESS-OM2/
│       │   └── run_ACCESS-OM2_{MODEL_CONFIG}_{NONLINEAR_SOLVER}_{job_id}.log
│       ├── create_matrix/
│       │   └── create_matrix_{MODEL_CONFIG}_{job_id}.{out,err}
│       └── solve_matrix_age/
│           └── solve_matrix_age_{MODEL_CONFIG}_{LINEAR_SOLVER}_{lumpspray_tag}_{job_id}.log
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

## Code formatting
Use the Runic formatter for Julia code:
Run it after you have finished editing files with `runic --inplace .`

## Submitting and monitoring PBS jobs

Submit a job:
```bash
qsub scripts/ACCESS-OM2-1_matrix_job.sh
qsub scripts/ACCESS-OM2-1_preprocess_job.sh
qsub scripts/ACCESS-OM2-1_GPU_job.sh
# with solver selection:
qsub -v NONLINEAR_SOLVER=newton scripts/ACCESS-OM2-1_GPU_job.sh
qsub -v NONLINEAR_SOLVER=anderson,AA_SOLVER=NLsolve scripts/ACCESS-OM2-1_GPU_job.sh
```

Monitor:
```bash
qstat -u bp3051          # list all your running/queued jobs
qstat -f <job_id>        # detailed status for one job
```

Pass environment variables at submission time:
```bash
qsub -v VELOCITY_SOURCE=bgridvelocities \
    scripts/ACCESS-OM2-1_matrix_job.sh
# solve matrix age with specific solver options:
qsub -v LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=yes \
    scripts/ACCESS-OM2-1_solve_matrix_age_job.sh
```

## Naming conventions

### Preprocessed velocity files (in `preprocessed_inputs/{parentmodel}/`)
- Monthly `FieldTimeSeries`: `*_periodic.jld2` (e.g. `u_from_mass_transport_periodic.jld2`)
- Time-averaged constant `Field`: `*_constant.jld2` (e.g. `u_from_mass_transport_constant.jld2`)
- Full file list:
  - `u_interpolated_periodic.jld2` / `u_interpolated_constant.jld2`
  - `v_interpolated_periodic.jld2` / `v_interpolated_constant.jld2`
  - `w_periodic.jld2` / `w_constant.jld2`
  - `u_from_mass_transport_periodic.jld2` / `u_from_mass_transport_constant.jld2`
  - `v_from_mass_transport_periodic.jld2` / `v_from_mass_transport_constant.jld2`
  - `w_from_mass_transport_periodic.jld2` / `w_from_mass_transport_constant.jld2`
  - `eta_periodic.jld2` / `eta_constant.jld2`
  - `{parentmodel}_grid.jld2` (grid; no periodic/constant distinction)

### Output directories (unified 4-part tag: `{VS}_{WF}_{AS}_{TS}` = `model_config`)
- Age simulation outputs: `outputs/{parentmodel}/age/{model_config}/`
- Matrix outputs: `outputs/{parentmodel}/matrices/{model_config}/`
- Matrix plots: `outputs/{parentmodel}/matrices/{model_config}/plots/`

### Log directories (`logs/julia/`)
- `MODEL_CONFIG` = `{VELOCITY_SOURCE}_{W_FORMULATION}_{ADVECTION_SCHEME}_{TIMESTEPPER}`

## Script overview
| Script | Purpose | Key env vars |
|--------|---------|-------------|
| `src/setup_model.jl` | Shared model setup (include'd by run/solve scripts) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/run_1year.jl` | Standalone 1-year age simulation | (inherits from setup_model.jl) |
| `src/run_10years.jl` | Standalone 10-year age simulation | (inherits from setup_model.jl) |
| `src/run_100years.jl` | Standalone 100-year age simulation | (inherits from setup_model.jl) |
| `src/solve_periodic_newton.jl` | Newton-GMRES periodic steady-state solver | JVP_METHOD, LINEAR_SOLVER, LUMP_AND_SPRAY, PRECONDITIONER_MATRIX_TYPE |
| `src/solve_periodic_anderson.jl` | Anderson/SpeedMapping periodic solver | AA_SOLVER (SpeedMapping/NLsolve/SIAMFANL) |
| `src/create_grid.jl` | Build and save the tripolar grid | PARENT_MODEL |
| `src/create_velocities.jl` | Preprocess MOM velocities → periodic FTS + constant Fields | PARENT_MODEL |
| `src/plot_outputs.jl` | Plot u/v/w/η outputs from simulation (standalone, CPU-only) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/create_matrix.jl` | Build transport matrix from constant fields | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER |
| `src/solve_matrix_age.jl` | Solve steady-state age from saved matrix M (CPU-only) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, LINEAR_SOLVER, LUMP_AND_SPRAY, PRECONDITIONER_MATRIX_TYPE |

## PBS scripts
- `scripts/ACCESS-OM2-1_preprocess_job.sh` — grid + velocities preprocessing (12 CPU, 47 GB)
- `scripts/ACCESS-OM2-1_CPU_job.sh` — offline simulation on CPU (12 CPU, 47 GB)
- `scripts/ACCESS-OM2-1_GPU_job.sh` — GPU job (1 GPU, 12 CPU, 47 GB); `NONLINEAR_SOLVER` selects script (1year/10years/100years/newton/anderson)
- `scripts/ACCESS-OM2-1_plot_1year_age_job.sh` — CPU plot job for 1-year age diagnostics (auto-submitted after 1year GPU job)
- `scripts/ACCESS-OM2-1_plot_10years_age_job.sh` — CPU plot job for 10-year age diagnostics (auto-submitted after 10years GPU job)
- `scripts/ACCESS-OM2-1_plot_100years_age_job.sh` — CPU plot job for 100-year age diagnostics (auto-submitted after 100years GPU job)
- `scripts/ACCESS-OM2-1_matrix_job.sh` — matrix build on CPU (48 CPU, 190 GB, normal queue)
- `scripts/ACCESS-OM2-1_solve_matrix_age_job.sh` — solve steady-state age from saved matrix (12 CPU, 47 GB, normal queue)

## Configuration environment variables

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

### Newton solver variables (`solve_periodic_newton.jl`, `solve_matrix_age.jl`)

| Variable | Default | Description |
|----------|---------|-------------|
| `JVP_METHOD` | `matrix` | JVP method (`matrix` or `finitediff`); Newton solver only |
| `LINEAR_SOLVER` | `Pardiso` | Direct solver for preconditioner (`Pardiso`, `ParU`, or `UMFPACK`) |
| `LUMP_AND_SPRAY` | `no` | Lump-and-spray coarsening for preconditioner (`yes`/`no`) |
| `PRECONDITIONER_MATRIX_TYPE` | `nonsym` | Pardiso matrix type (`nonsym` or `sym_cleaned`); Pardiso only |

- Output filename tags: `Pardiso`/`ParU`/`UMFPACK` for LINEAR_SOLVER; `LSprec`/`prec` for LUMP_AND_SPRAY
- Example: `age_newton_Pardiso_prec.jld2`, `steady_age_full_ParU_LSprec.jld2`, `steady_age_full_UMFPACK_prec.jld2`

### Anderson solver variables (`solve_periodic_anderson.jl`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AA_SOLVER` | `SpeedMapping` | Solver backend: `SpeedMapping` (SpeedMapping.jl), `NLsolve` (NLsolve.jl), `SIAMFANL` (SIAMFANLEquations.jl), `FixedPoint` (FixedPointAcceleration.jl) |
| `AA_M` | `40` | Anderson history size (used by NLsolve, SIAMFANL, FixedPoint) |
| `NLSAA_BETA` | `1.0` | Anderson damping parameter (try 0.5 for slow convergence) |
| `SMAA_SIGMA_MIN` | `0.0` | SpeedMapping minimum σ; setting to 1 may avoid stalling |
| `SMAA_STABILIZE` | `no` | Stabilization mapping before extrapolation (`yes`/`no`) |
| `SMAA_CHECK_OBJ` | `no` | Restart at best past iterate on NaN/Inf (`yes`/`no`) |
| `SMAA_ORDERS` | `332` | Alternating order sequence (each digit 1–3) |

## Key design decisions
- Model setup is shared via `setup_model.jl` (include'd by downstream scripts)
- `setup_model.jl` creates the model but NOT the simulation — each downstream script creates its own
- Matrix build always on CPU (sparsity detection/coloring incompatible with GPU)
- `create_matrix.jl` uses time-averaged constant Fields (not FieldTimeSeries) → single Jacobian call
- Simulation scripts use periodic FieldTimeSeries (12 monthly snapshots, Cyclical indexing)
- Constant fields for matrix build use same BCs (`FPivotZipperBoundaryCondition`) as the per-month fields in `create_velocities.jl`
- zstar initialisation before Jacobian: `_update_zstar_scaling!(η_constant, grid)`
- Age solving is factored out into `solve_matrix_age.jl` (runs on saved M.jld2)
- Newton solver loads M from `create_matrix.jl` output; LUMP_AND_SPRAY controls preconditioner coarsening
- LINEAR_SOLVER selects direct solver: Pardiso (MKL), ParU (SuiteSparse parallel LU), or UMFPACK (SuiteSparse serial LU)
- Newton solver uses approximate JVP via `stop_time * M` (sparse matvec) or finite-diff via `AutoFiniteDiff()`
- Anderson/SpeedMapping solver needs no matrix or preconditioner — pure fixed-point acceleration
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
