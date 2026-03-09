#!/usr/bin/env bash
set -euo pipefail

# One-time MPI setup script. Run on the login node (internet access).
# Adds MPI + MPIPreferences, configures system OpenMPI, then submits
# precompile jobs on compute nodes (CPU first, GPU second).

# 1. Load openmpi so use_system_binary() can find libmpi
module load openmpi/5.0.8

# 2. Add MPI + MPIPreferences WITHOUT precompilation, then configure system MPI
env JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -e '
    using Pkg
    Pkg.add(["MPI", "MPIPreferences"])
    using MPIPreferences
    MPIPreferences.use_system_binary()
'

# 3. Submit precompile jobs (GPU waits for CPU to finish)
export GPU_JOB_ID=$(qsub -W depend=on:1 scripts/setup_mpi_precompile_GPU.sh)
qsub -W depend=beforeok:${GPU_JOB_ID} scripts/setup_mpi_precompile_CPU.sh
