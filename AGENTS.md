# Agent Context

## Project directory tree

```
ACCESS-OM2_x_Oceananigans/
├── src/
│   ├── setup_model.jl            # Shared model setup (include'd by run/solve scripts)
│   ├── run_1year.jl              # Standalone 1-year age simulation → outputs/{model}/age/
│   ├── solve_periodic_newton.jl  # Newton-GMRES periodic steady-state solver
│   ├── solve_periodic_anderson.jl # Anderson/SpeedMapping periodic solver
│   ├── create_grid.jl            # Build tripolar grid → preprocessed_inputs/{model}/{model}_grid.jld2
│   ├── create_velocities.jl      # Preprocess MOM velocities → *_periodic.jld2 + *_constant.jld2
│   ├── create_closures.jl        # (WIP — not yet used in pipeline)
│   ├── create_matrix.jl          # Build transport matrix → outputs/{model}/matrices/
│   ├── plot_outputs.jl           # Plot u/v/w/η outputs from simulation (standalone, CPU-only)
│   └── tripolargrid_reader.jl    # load_tripolar_grid(), make_plottable_array()
├── scripts/
│   ├── ACCESS-OM2-1_preprocess_job.sh   # PBS: grid + velocities (12 CPU, 47 GB, express)
│   ├── ACCESS-OM2-1_CPU_job.sh          # PBS: offline simulation CPU (12 CPU, 47 GB, express)
│   ├── ACCESS-OM2-1_GPU_job.sh          # PBS: GPU job (1 GPU, 47 GB, gpuvolta) — SOLVE_METHOD selects script
│   ├── ACCESS-OM2-1_matrix_job.sh       # PBS: matrix build CPU (48 CPU, 190 GB, normal)
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
│   ├── age/{VELOCITY_SOURCE}_{W_FORMULATION}/   # offline simulation outputs
│   └── matrices/{VELOCITY_SOURCE}_constant/     # Jacobian M.jld2, steady_age_*.jld2, plots/
├── logs/                               # symlink → /scratch/y99/TMIP/…/logs/
│   ├── PBS/                            # PBS scheduler stdout/stderr (set via #PBS -o/-e)
│   └── julia/                          # Julia script stdout/stderr, organised by script name
│       ├── create_grid/
│       │   └── create_grid_{parentmodel}_{job_id}.{out,err}
│       ├── create_velocities/
│       │   └── create_velocities_{parentmodel}_{job_id}.{out,err}
│       ├── run_ACCESS-OM2/
│       │   └── run_ACCESS-OM2_{MODEL_CONFIG}_{SOLVE_METHOD}_{job_id}.log
│       └── create_matrix/
│           └── create_matrix_{MODEL_CONFIG}_{job_id}.{out,err}
├── Project.toml
├── LocalPreferences.toml               # CUDA version pin (local = true, version = "12.9")
└── AGENTS.md
```

## Julia depot
Julia packages are installed in `JULIA_DEPOT_PATH=/g/data/y99/bp3051/.julia/`.

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
qsub -v SOLVE_METHOD=newton scripts/ACCESS-OM2-1_GPU_job.sh
qsub -v SOLVE_METHOD=anderson,ACCELERATION_METHOD=anderson scripts/ACCESS-OM2-1_GPU_job.sh
```

Monitor:
```bash
qstat -u bp3051          # list all your running/queued jobs
qstat -f <job_id>        # detailed status for one job
```

Pass environment variables at submission time:
```bash
qsub -v VELOCITY_SOURCE=bgridvelocities,ENABLE_AGE_SOLVE=true \
    scripts/ACCESS-OM2-1_matrix_job.sh
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

### Output directories
- Age simulation outputs: `outputs/{parentmodel}/age/{VELOCITY_SOURCE}_{W_FORMULATION}/`
- Matrix outputs: `outputs/{parentmodel}/matrices/{VELOCITY_SOURCE}_constant/`
- Matrix plots: `outputs/{parentmodel}/matrices/{VELOCITY_SOURCE}_constant/plots/`

### Log directories (`logs/julia/`)
- `MODEL_CONFIG` = `{VELOCITY_SOURCE}_{W_FORMULATION}` for simulation/solver scripts
- `MODEL_CONFIG` = `{VELOCITY_SOURCE}_constant` for `create_matrix`

## Script overview
| Script | Purpose | Key env vars |
|--------|---------|-------------|
| `src/setup_model.jl` | Shared model setup (include'd by run/solve scripts) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION |
| `src/run_1year.jl` | Standalone 1-year age simulation | (inherits from setup_model.jl) |
| `src/solve_periodic_newton.jl` | Newton-GMRES periodic steady-state solver | JVP_METHOD (matrix/finitediff) |
| `src/solve_periodic_anderson.jl` | Anderson/SpeedMapping periodic solver | ACCELERATION_METHOD (speedmapping/anderson) |
| `src/create_grid.jl` | Build and save the tripolar grid | PARENT_MODEL |
| `src/create_velocities.jl` | Preprocess MOM velocities → periodic FTS + constant Fields | PARENT_MODEL |
| `src/plot_outputs.jl` | Plot u/v/w/η outputs from simulation (standalone, CPU-only) | PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION |
| `src/create_matrix.jl` | Build transport matrix from constant fields; optionally solve for steady-state age | PARENT_MODEL, VELOCITY_SOURCE, ENABLE_AGE_SOLVE |

## PBS scripts
- `scripts/ACCESS-OM2-1_preprocess_job.sh` — grid + velocities preprocessing (12 CPU, 47 GB)
- `scripts/ACCESS-OM2-1_CPU_job.sh` — offline simulation on CPU (12 CPU, 47 GB)
- `scripts/ACCESS-OM2-1_GPU_job.sh` — GPU job (1 GPU, 12 CPU, 47 GB); `SOLVE_METHOD` selects script (1year/newton/anderson)
- `scripts/ACCESS-OM2-1_matrix_job.sh` — matrix build on CPU (48 CPU, 190 GB, normal queue)

## Key design decisions
- Model setup is shared via `setup_model.jl` (include'd by downstream scripts)
- `setup_model.jl` creates the model but NOT the simulation — each downstream script creates its own
- Matrix build always on CPU (sparsity detection/coloring incompatible with GPU)
- `create_matrix.jl` uses time-averaged constant Fields (not FieldTimeSeries) → single Jacobian call
- Simulation scripts use periodic FieldTimeSeries (12 monthly snapshots, Cyclical indexing)
- Constant fields for matrix build use same BCs (`FPivotZipperBoundaryCondition`) as the per-month fields in `create_velocities.jl`
- zstar initialisation before Jacobian: `_update_zstar_scaling!(η_constant, grid)`
- ENABLE_AGE_SOLVE (default false) gates the LUMP/SPRAY + linear solves + 3D field save in create_matrix.jl
- Newton solver loads M from `create_matrix.jl` output, builds LUMP/SPRAY preconditioner
- Newton solver uses approximate JVP via `stop_time * M` (sparse matvec) or finite-diff via `AutoFiniteDiff()`
- Anderson/SpeedMapping solver needs no matrix or preconditioner — pure fixed-point acceleration
- GPU arrays preallocated once; `copyto!` used for CPU↔GPU transfer in G!
