# ACCESS-OM2_x_Oceananigans

Trying to "couple" Oceananigans for time-stepping an offline surrogate of ACCESS-OM2 and then
use transport matrices to solve for periodic state using a Newon–Krylov solver.

🚧 This is exploratory WIP and may be abandonned any time!

##



## Project setup notes

Gadi compute nodes don't have access to the internet, so the project dependencies must be downloaded on the login node. But the default mutli-threaded precompilation could use too much resources and crash during `pkg> up`. A solution may be to first download the deps on a login node, with a flag to prevent precompilation:

```bash
env JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -e 'using Pkg; Pkg.update()'
```

and then precompile on a compute node, essentially running

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

by simply submitting the job (48 CPUs)
```bash
qsub scripts/pkg_instantiate_project_CPU.sh
```

<!-- TODO: Not sure I need a separate script for GPU? -->
<!-- TODO: Maybe it should also use `pkg> update` instead of just `instantiate`? -->
and for GPU (1 GPU, 12 CPUs)
```bash
qsub scripts/pkg_instantiate_project_GPU.sh
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
