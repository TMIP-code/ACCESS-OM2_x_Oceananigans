#!/bin/bash

#PBS -N OM2-1_TMageGPU
#PBS -P y99
#PBS -l mem=96GB
#PBS -q gpuvolta
#PBS -l walltime=01:00:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Export solver-specific env vars if set
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${MATRIX_PROCESSING:-}" ] && export MATRIX_PROCESSING && echo "MATRIX_PROCESSING=$MATRIX_PROCESSING"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"

# Load CUDA module
echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

# Point CUDSS.jl to the JLL artifact since Gadi's system CUDA lacks libcudss
CUDSS_LIB=$(find "${JULIA_DEPOT_PATH:-$HOME/.julia}/artifacts" -name "libcudss.so" -print -quit 2>/dev/null)
if [ -z "$CUDSS_LIB" ]; then
    # Fallback: search in gdata depot
    CUDSS_LIB=$(find /g/data/y99/bp3051/.julia/artifacts -name "libcudss.so" -print -quit 2>/dev/null)
fi
if [ -z "$CUDSS_LIB" ]; then
    echo "ERROR: Could not find libcudss.so in Julia artifacts" >&2
    exit 1
fi
export JULIA_CUDSS_LIBRARY_PATH=$(dirname "$CUDSS_LIB")
export LD_LIBRARY_PATH="${JULIA_CUDSS_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
echo "JULIA_CUDSS_LIBRARY_PATH=$JULIA_CUDSS_LIBRARY_PATH"

# Determine coarse tag for log file naming
if [ "$LUMP_AND_SPRAY" = "yes" ]; then
    COARSE_TAG="coarse"
else
    COARSE_TAG="full"
fi

run_log_dir=logs/julia/$PARENT_MODEL/TM
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Solving matrix age on GPU (CUDSS) for MODEL_CONFIG=$MODEL_CONFIG, TM_SOURCE=$TM_SOURCE, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, MATRIX_PROCESSING=$MATRIX_PROCESSING"
log_file="$run_log_dir/solve_${MODEL_CONFIG}_${TM_SOURCE}_${COARSE_TAG}_CUDSS_${MATRIX_PROCESSING}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project src/solve_matrix_age_gpu.jl &> "$log_file"
echo "Done solving matrix age on GPU"
echo "logged output in $log_file"
