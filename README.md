# ACCESS-OM2_x_Oceananigans

Trying to "couple" Oceananigans for time-stepping an offline surrogate of ACCESS-OM2 and then
use transport matrices to solve for periodic state using a Newon–Krylov solver.

🚧 This is exploratory WIP and may be abandonned any time!

##



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

Shell defaults are set in `scripts/env_defaults.sh`, which is sourced by all PBS job scripts. Override at submission time:

```bash
qsub -v TIMESTEPPER=SRK3,ADVECTION_SCHEME=weno5 scripts/ACCESS-OM2-1_GPU_job.sh
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
