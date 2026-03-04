#!/bin/bash

#PBS -N solve_age_OM2-1
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
[ -n "${PRECONDITIONER_MATRIX_TYPE:-}" ] && export PRECONDITIONER_MATRIX_TYPE && echo "PRECONDITIONER_MATRIX_TYPE=$PRECONDITIONER_MATRIX_TYPE"

# Determine lumpspray tag for log file naming
if [ "$LUMP_AND_SPRAY" = "yes" ]; then
    LUMPSPRAY_TAG="LSprec"
else
    LUMPSPRAY_TAG="prec"
fi

run_log_dir=logs/julia/solve_matrix_age
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Solving matrix age for MODEL_CONFIG=$MODEL_CONFIG, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
julia $JULIA_BOUNDS_FLAG --project src/solve_matrix_age.jl &> "$run_log_dir/solve_matrix_age_${MODEL_CONFIG}_${LINEAR_SOLVER}_${LUMPSPRAY_TAG}_${job_id}.log"
echo "Done solving matrix age"
echo "logged output in $run_log_dir/solve_matrix_age_${MODEL_CONFIG}_${LINEAR_SOLVER}_${LUMPSPRAY_TAG}_${job_id}.log"
