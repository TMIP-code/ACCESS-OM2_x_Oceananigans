# ACCESS-OM2_x_Oceananigans

Trying to "couple" Oceananigans for time-stepping an offline surrogate of ACCESS-OM2 and then
use transport matrices to solve for periodic state using a Newon–Krylov solver.

🚧 This is exploratory WIP and may be abandonned any time!

## Pipeline

The full pipeline is managed by a unified `scripts/driver.sh` that submits chained PBS jobs with `afterok` dependencies. All scripts are model-agnostic — `PARENT_MODEL` selects the model. Run from the login node:

```bash
# Run the full ACCESS-OM2-1 pipeline
JOB_CHAIN=full bash scripts/driver.sh

# Run the full ACCESS-OM2-025 pipeline
PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=full bash scripts/driver.sh
```

### Dependency DAG

```
grid
 └── vel
      ├── run1yr
      │    └── TMsnapshot (TM_SOURCE=avg24)
      │         ├── TMsolve(avg24) ── Pardiso(CPU) + CUDSS(GPU)
      │         └── NK(avg24) ── run1yrNK(avg24)
      ├── run10yr ── plot10yr
      ├── run100yr ── plot100yr
      ├── runlong
      └── TMbuild (TM_SOURCE=const)
           ├── TMsolve(const) ── Pardiso(CPU) + CUDSS(GPU)
           └── NK(const) ── run1yrNK(const) ── plotNK
```

### Selecting steps with `JOB_CHAIN`

Use the `JOB_CHAIN` env var to run only a subset of the pipeline. Steps not in the chain are skipped (their outputs are assumed to already exist). `JOB_CHAIN` is required — the driver prints usage help if not set.

**Steps** (topological order):
`grid vel run1yr run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plot1yr plot10yr plot100yr`

**Shortcuts:**
| Shortcut | Expands to |
|---|---|
| `preprocessing` | `grid-vel` |
| `standardruns` | `run1yr-run10yr-run100yr-runlong` |
| `TMall` | `TMbuild-TMsnapshot-TMsolve` |
| `plotall` | `plot1yr-plot10yr-plot100yr-plotNK` |
| `full` | `preprocessing-run1yr-TMall-NK-run1yrNK-plotNK-plot1yr` |

**Range notation:** `A..B` expands to all steps on any path from A to B in the dependency DAG — not a flat list.

```bash
# Only run Newton-GMRES solves (matrices must already exist)
JOB_CHAIN=NK bash scripts/driver.sh

# Run 1-year simulation and plot
JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh

# Build matrices and run all solvers
JOB_CHAIN=run1yr-TMall-NK bash scripts/driver.sh

# Everything from vel to NK (range follows the DAG, excludes run10yr/runlong/TMsolve)
JOB_CHAIN=vel..NK bash scripts/driver.sh

# Re-run + plot from NK solution (range follows NK→run1yrNK→plotNK path only)
JOB_CHAIN=run1yrNK..plotNK bash scripts/driver.sh

# Run both const and avg24 branches
TM_SOURCE=both JOB_CHAIN=NK-run1yrNK-plotNK bash scripts/driver.sh

# Run preprocessing only
JOB_CHAIN=preprocessing bash scripts/driver.sh

# ACCESS-OM2-025 with specific GPU queue
PARENT_MODEL=ACCESS-OM2-025 GPU_RESOURCES=gpuvolta JOB_CHAIN=run1yr bash scripts/driver.sh
```

### TM_SOURCE filtering

`TM_SOURCE` controls which transport matrix branch is used for `TMsolve`, `NK`, and `run1yrNK`:

| Value | Description |
|-------|-------------|
| `const` (default) | Only const-field matrices (from `TMbuild`) |
| `avg24` | Only time-averaged snapshot matrices (from `TMsnapshot`) |
| `both` | Both branches in parallel |

### GPU preprocessing with `PREPROCESS_ARCH`

Velocity creation can run on GPU for faster processing (grid creation always runs on CPU):

```bash
# Run velocities on GPU
PREPROCESS_ARCH=GPU JOB_CHAIN=preprocessing bash scripts/driver.sh
```

### Model configs

Model-specific settings (walltimes, PBS name prefix) live in `model_configs/`:
- `model_configs/ACCESS-OM2-1.sh`
- `model_configs/ACCESS-OM2-025.sh`

### Script organisation

```
scripts/
├── driver.sh                  # Unified pipeline entry point
├── env_defaults.sh            # Common env var defaults
├── preprocessing/             # Grid, velocities, transport matrices
├── standard_runs/             # Age simulations (1yr, 10yr, 100yr, long)
├── solvers/                   # Newton-Krylov + TM age solvers
├── plotting/                  # Diagnostic plots
├── benchmarks/                # Parameter sweep submitters
├── maintenance/               # Package management, MPI setup, archiving
├── debugging/                 # Test/check scripts
└── prepreprocessing/          # CDO periodic averaging (external)
```

## Project setup notes

Gadi compute nodes don't have access to the internet, so the project dependencies must be downloaded on the login node. But the default multi-threaded precompilation could use too much resources and crash during `pkg> up`. Instead, run the dedicated script `scripts/maintenance/pkg_update_project.sh`, which runs `pkg> up` on the login node _without_ precompilation, then submits precompilation on compute nodes on the CPU and then on the GPU.

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
qsub -v TIMESTEPPER=SRK3,ADVECTION_SCHEME=weno5 scripts/standard_runs/run_1year.sh
```

## Tests

Julia test scripts live in `test/`. To run the regression test comparing newly-built snapshot matrices against archived reference matrices, submit a PBS job (these load large matrices and must run on a compute node, not the login node):

```bash
qsub scripts/debugging/check_snapshot_matrices_job.sh
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
