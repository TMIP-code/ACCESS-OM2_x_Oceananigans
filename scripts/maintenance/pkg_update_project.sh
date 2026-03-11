#!/usr/bin/env bash
set -euo pipefail

# This script updates (downloads) packages on the login node (internet access)
# but does not precompile on the login node. Instead, it submits the precompilation
# as a job on compute nodes (CPU first and GPU second).

# Reference for making the GPU job wait for the CPU job to finish:
# https://opus.nci.org.au/spaces/Help/pages/241927682/Job+Dependencies...

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
source scripts/env_defaults.sh

# 1. Update (and download) packages on the login node but WITHOUT precompilation
env JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -e 'using Pkg; Pkg.update()'

# 2. Download OceanBasins DataDeps data (requires internet, only on login node)
env JULIA_PKG_PRECOMPILE_AUTO=0 DATADEPS_ALWAYS_ACCEPT=true julia --project -e 'using OceanBasins; oceanpolygons()'

# 2. Submit the GPU job first (it will wait for CPU job to finish)
export GPU_JOB_ID=$(qsub -W depend=on:1 scripts/maintenance/pkg_instantiate_project_GPU.sh)

# 3. Submit the CPU job that will precompile on the compute node
qsub -W depend=beforeok:${GPU_JOB_ID} scripts/maintenance/pkg_instantiate_project_CPU.sh
