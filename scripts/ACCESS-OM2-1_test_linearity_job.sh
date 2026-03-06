#!/bin/bash

#PBS -N OM2-1_linearity
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

run_log_dir=logs/julia/TM
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Running linearity test for MODEL_CONFIG=$MODEL_CONFIG"
log_file="$run_log_dir/test_linearity_${MODEL_CONFIG}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project src/test_linearity.jl &> "$log_file"
echo "Done running linearity test"
echo "logged output in $log_file"
