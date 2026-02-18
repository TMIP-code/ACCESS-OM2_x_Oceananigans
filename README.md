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
