#!/usr/bin/env bash
set -euo pipefail

# This script updates (downloads) packages on the login node (internet access)
# but does not precompile on the login node. Instead, it submits the precompilation
# as a job on compute nodes (CPU first and GPU second)

# Reference for making the GPU job wait for the CPU job to finish:
# https://opus.nci.org.au/spaces/Help/pages/241927682/Job+Dependencies...

# 1. Update (and download) packages on the login node but WITHOUT precoimpilation
env JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -e 'using Pkg; Pkg.update()'

# 3. Submit the GPU job second, but it will wait for the CPU job to finish.
export GPU_JOB_ID=$(qsub -W depend=on:1 scripts/pkg_instantiate_project_GPU.sh)

# 2. Submit the CPU job that will update packages and precompile on the compute node.
qsub -W depend=beforeok:${GPU_JOB_ID} scripts/pkg_instantiate_project_CPU.sh
