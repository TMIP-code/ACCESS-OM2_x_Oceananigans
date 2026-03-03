#!/bin/bash

#PBS -N matrix_OM2-1_CPU
#PBS -P y99
#PBS -l mem=190GB
#PBS -q normal
#PBS -l walltime=04:00:00
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

ENABLE_AGE_SOLVE=${ENABLE_AGE_SOLVE:-false}
export ENABLE_AGE_SOLVE
echo "ENABLE_AGE_SOLVE=$ENABLE_AGE_SOLVE"

# ulimit -s unlimited

run_log_dir=logs/julia/create_matrix
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Building transport matrix for MODEL_CONFIG=$MODEL_CONFIG, ENABLE_AGE_SOLVE=$ENABLE_AGE_SOLVE"
julia $JULIA_BOUNDS_FLAG --project src/create_matrix.jl &> "$run_log_dir/create_matrix_${MODEL_CONFIG}_${job_id}.log"
echo "Done building transport matrix"
echo "logged output in $run_log_dir/create_matrix_${MODEL_CONFIG}_${job_id}.log"
