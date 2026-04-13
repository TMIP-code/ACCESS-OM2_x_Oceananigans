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

# 2. Submit the CPU precompile job
CPU_JOB_ID=$(qsub scripts/maintenance/pkg_instantiate_project_CPU.sh)
echo "CPU precompile job: $CPU_JOB_ID"

# 3. Wait for the CPU precompile job to finish before downloading DataDeps data,
#    so that the login-node Julia run in step 4 hits the precompile cache and
#    doesn't trigger a fresh (slow, heavy) precompile on the login node.
echo "Waiting for CPU precompile job ($CPU_JOB_ID) to finish..."
sleep 30
while qstat "$CPU_JOB_ID" &>/dev/null; do
    sleep 30
done
echo "CPU precompile job done."

# 4. Download OceanBasins DataDeps data (requires internet, only on login node)
echo "Downloading OceanBasins data..."
DATADEPS_ALWAYS_ACCEPT=true julia --project -e 'using OceanBasins; oceanpolygons()'
echo "OceanBasins data downloaded successfully"

# 5. Submit the GPU precompile job
GPU_JOB_ID=$(qsub scripts/maintenance/pkg_instantiate_project_GPU.sh)
echo "GPU precompile job: $GPU_JOB_ID"
