#!/bin/bash

#PBS -N TMchk
#PBS -P y99
#PBS -l mem=47GB
#PBS -q normal
#PBS -l walltime=00:30:00
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

log_dir=logs/julia/test
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Checking snapshot matrices for ACCESS-OM2-1"
log_file="$log_dir/check_snapshot_matrices_ACCESS-OM2-1_${MODEL_CONFIG}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project test/check_snapshot_matrices.jl ACCESS-OM2-1 &> "$log_file"
echo "Done checking ACCESS-OM2-1 (logged to $log_file)"

echo "Checking snapshot matrices for ACCESS-OM2-025"
log_file="$log_dir/check_snapshot_matrices_ACCESS-OM2-025_${MODEL_CONFIG}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project test/check_snapshot_matrices.jl ACCESS-OM2-025 &> "$log_file"
echo "Done checking ACCESS-OM2-025 (logged to $log_file)"
