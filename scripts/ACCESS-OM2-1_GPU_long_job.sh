#!/bin/bash

#PBS -N run_OM2-1_long
#PBS -P y99
#PBS -l mem=47GB
#PBS -q gpuvolta
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

echo "Running run_long.jl for PARENT_MODEL=$PARENT_MODEL, NYEARS=$NYEARS"
run_log_dir=logs/julia/run_ACCESS-OM2
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
echo "logging output in $run_log_dir"
julia $JULIA_BOUNDS_FLAG --project src/run_long.jl &> "$run_log_dir/run_ACCESS-OM2_${MODEL_CONFIG}_long_${NYEARS}years_${job_id}.log"
echo "Done running run_long.jl"
