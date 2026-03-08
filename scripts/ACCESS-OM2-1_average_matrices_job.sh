#!/bin/bash

#PBS -N OM2-1_avg_M
#PBS -P y99
#PBS -l mem=47GB
#PBS -q normal
#PBS -l walltime=01:00:00
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

run_log_dir=logs/julia/$PARENT_MODEL/TM
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Averaging snapshot matrices for MODEL_CONFIG=$MODEL_CONFIG"
log_file="$run_log_dir/average_matrices_${MODEL_CONFIG}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project src/average_snapshot_matrices.jl &> "$log_file"
echo "Done averaging snapshot matrices"
echo "logged output in $log_file"
