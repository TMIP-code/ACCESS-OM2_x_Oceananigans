#!/bin/bash

#PBS -N OM2-1_TM_solve
#PBS -P y99
#PBS -l mem=190GB
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=48
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
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${MATRIX_PROCESSING:-}" ] && export MATRIX_PROCESSING && echo "MATRIX_PROCESSING=$MATRIX_PROCESSING"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"

# Determine coarse tag for log file naming
if [ "$LUMP_AND_SPRAY" = "yes" ]; then
    COARSE_TAG="coarse"
else
    COARSE_TAG="full"
fi

run_log_dir=logs/julia/TM
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Solving matrix age for MODEL_CONFIG=$MODEL_CONFIG, TM_SOURCE=$TM_SOURCE, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, MATRIX_PROCESSING=$MATRIX_PROCESSING"
log_file="$run_log_dir/solve_${MODEL_CONFIG}_${TM_SOURCE}_${COARSE_TAG}_${LINEAR_SOLVER}_${MATRIX_PROCESSING}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project src/solve_matrix_age.jl &> "$log_file"
echo "Done solving matrix age"
echo "logged output in $log_file"
