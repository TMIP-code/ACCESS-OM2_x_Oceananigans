#!/bin/bash

#PBS -N OM2-025_long
#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l walltime=48:00:00
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

NYEARS=${NYEARS:-3000}
export NYEARS
echo "NYEARS=$NYEARS"

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_long_${NYEARS}years_${job_id}.log"

echo "Running src/run_long.jl for PARENT_MODEL=$PARENT_MODEL (NYEARS=$NYEARS)"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project src/run_long.jl &> "$log_file"
echo "Done running src/run_long.jl for PARENT_MODEL=$PARENT_MODEL"
