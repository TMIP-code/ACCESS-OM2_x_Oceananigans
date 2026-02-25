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

PARENT_MODEL=ACCESS-OM2-1
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
ENABLE_AGE_SOLVE=${ENABLE_AGE_SOLVE:-false}
MODEL_CONFIG="${VELOCITY_SOURCE}_constant"
export PARENT_MODEL VELOCITY_SOURCE ENABLE_AGE_SOLVE

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENT_MODEL=$PARENT_MODEL, VELOCITY_SOURCE=$VELOCITY_SOURCE, ENABLE_AGE_SOLVE=$ENABLE_AGE_SOLVE"
cd "$repo_root"

run_log_dir="$repo_root/logs/julia/create_matrix"
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Building transport matrix for PARENT_MODEL=$PARENT_MODEL, VELOCITY_SOURCE=$VELOCITY_SOURCE, ENABLE_AGE_SOLVE=$ENABLE_AGE_SOLVE"
julia --project "$repo_root/src/create_matrix.jl" \
    1> "$run_log_dir/create_matrix_${MODEL_CONFIG}_${job_id}.out" \
    2> "$run_log_dir/create_matrix_${MODEL_CONFIG}_${job_id}.err"
echo "Done building transport matrix"
echo "logged output in $run_log_dir/create_matrix_${MODEL_CONFIG}_${job_id}.{out,err}"
