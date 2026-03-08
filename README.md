# ACCESS-OM2_x_Oceananigans

Trying to "couple" Oceananigans for time-stepping an offline surrogate of ACCESS-OM2 and then
use transport matrices to solve for periodic state using a Newon–Krylov solver.

🚧 This is exploratory WIP and may be abandonned any time!

## Pipeline

The full pipeline is managed by driver scripts that submit chained PBS jobs with `afterok` dependencies. Run from the login node:

```bash
# Run the full ACCESS-OM2-1 pipeline (11 jobs)
bash scripts/ACCESS-OM2-1_driver.sh

# Run the full ACCESS-OM2-025 pipeline
bash scripts/ACCESS-OM2-025_driver.sh
```

### Dependency DAG

```
grid
 └── vel
      ├── 1year
      │    └── snapM+avgM (TM_SOURCE=avg24)
      │         ├── TMage(avg24) ── Pardiso(CPU) + CUDSS(GPU)
      │         └── NK(avg24)
      └── constM (TM_SOURCE=const)
           ├── TMage(const) ── Pardiso(CPU) + CUDSS(GPU)
           └── NK(const)
```

### Skipping steps with `JOB_CHAIN`

Use the `JOB_CHAIN` env var to run only a subset of the pipeline. Steps not in the string are skipped (their outputs are assumed to already exist):

```bash
# Default (full chain)
JOB_CHAIN=grid-vel-1year-TM-TMage-NK bash scripts/ACCESS-OM2-1_driver.sh

# Skip grid (already built), start from velocities
JOB_CHAIN=vel-1year-TM-TMage-NK bash scripts/ACCESS-OM2-1_driver.sh

# Only run Newton-GMRES solves (matrices must already exist)
JOB_CHAIN=NK bash scripts/ACCESS-OM2-1_driver.sh

# Only build matrices (grid and velocities must already exist)
JOB_CHAIN=1year-TM bash scripts/ACCESS-OM2-1_driver.sh

# Only solve matrix age (matrices must already exist)
JOB_CHAIN=TMage bash scripts/ACCESS-OM2-1_driver.sh

# Rebuild matrices and run all solvers
JOB_CHAIN=1year-TM-TMage-NK bash scripts/ACCESS-OM2-1_driver.sh
```

### GPU preprocessing with `PREPROCESS_ARCH`

Velocity creation can run on GPU for faster processing (grid creation always runs on CPU):

```bash
# Run velocities on GPU
PREPROCESS_ARCH=GPU bash scripts/ACCESS-OM2-1_driver.sh

# Combine with JOB_CHAIN
PREPROCESS_ARCH=GPU JOB_CHAIN=vel bash scripts/ACCESS-OM2-1_driver.sh
```

## Project setup notes

Gadi compute nodes don't have access to the internet, so the project dependencies must be downloaded on the login node. But the default mutli-threaded precompilation could use too much resources and crash during `pkg> up`. Instead, run the dedicated script in `scripts/pkg_udate_project.sh`, which run `pkg> up` on the login node _without_ precompilation, then run precompilation on compute nodes on the CPU and then on the GPU.

## Configuration

Simulations are configured via environment variables. The 4 core config variables determine the model setup and output directory paths:

| Variable | Valid values | Default | Description |
|----------|-------------|---------|-------------|
| `VELOCITY_SOURCE` | `cgridtransports`, `bgridvelocities` | `cgridtransports` | Source of prescribed velocities |
| `W_FORMULATION` | `wdiagnosed`, `wprescribed` | `wdiagnosed` | Vertical velocity treatment |
| `ADVECTION_SCHEME` | `centered2`, `weno3`, `weno5` | `centered2` | Tracer advection scheme |
| `TIMESTEPPER` | `AB2`, `SRK2`, `SRK3`, `SRK4`, `SRK5` | `AB2` | Time-stepping scheme |

Timestepper values map to Oceananigans symbols:
- `AB2` = `:QuasiAdamsBashforth2` (default quasi-Adams-Bashforth 2nd order)
- `SRK{N}` = `:SplitRungeKutta{N}` (split Runge-Kutta with N = 2..5 stages)

The combined tag `MODEL_CONFIG = {VS}_{WF}_{AS}_{TS}` (e.g. `cgridtransports_wdiagnosed_centered2_AB2`) determines output directory paths and log filenames.

### Solver-specific variables

These configure the fixed-point acceleration solvers in `solve_periodic_AA.jl` (archived):

| Variable | Default | Description |
|----------|---------|-------------|
| `AA_M` | `40` | Anderson history size (used by NLsolve, SIAMFANL, FixedPoint) |
| `NLSAA_BETA` | `1.0` | Anderson damping parameter (try 0.5 for slow convergence) |
| `SMAA_SIGMA_MIN` | `0.0` | SpeedMapping minimum σ; setting to 1 may avoid stalling |
| `SMAA_STABILIZE` | `no` | Stabilization mapping before extrapolation (`yes`/`no`) |
| `SMAA_CHECK_OBJ` | `no` | Restart at best past iterate on NaN/Inf (`yes`/`no`) |
| `SMAA_ORDERS` | `332` | Alternating order sequence (each digit 1–3) |

Shell defaults are set in `scripts/env_defaults.sh`, which is sourced by all PBS job scripts. Override at submission time:

```bash
qsub -v TIMESTEPPER=SRK3,ADVECTION_SCHEME=weno5 scripts/ACCESS-OM2-1_run_1year.sh
```

## Preprocessed outputs layout

Preprocessing writes data and images under:

`preprocessed_inputs/<PARENT_MODEL>/`

Data files:

- `u_interpolated.jld2`
- `v_interpolated.jld2`
- `w.jld2`
- `eta.jld2`
- `u_from_mass_transport.jld2`
- `v_from_mass_transport.jld2`
- `w_from_mass_transport.jld2`

Plots are colocated under:

`preprocessed_inputs/<PARENT_MODEL>/plots/`

with subdirectories for each plotted field family:

- `u/` (original B-grid `u`)
- `v/` (original B-grid `v`)
- `u_interpolated/`
- `v_interpolated/`
- `w/`
- `eta/`
- `u_from_mass_transport/`
- `v_from_mass_transport/`
- `w_from_mass_transport/`

Each plot subdirectory is split by vertical level (`k<level>` when applicable). Each image contains one field only.
