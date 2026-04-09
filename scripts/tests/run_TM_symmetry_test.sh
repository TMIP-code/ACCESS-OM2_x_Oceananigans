#!/bin/bash

#PBS -P y99
#PBS -l mem=192GB
#PBS -q express
#PBS -l ngpus=0
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

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/test
mkdir -p "$log_dir"
log_file="$log_dir/TM_symmetry_${MODEL_CONFIG}_${job_id}.log"

echo "Running test/test_TM_symmetry.jl for MODEL_CONFIG=$MODEL_CONFIG"
echo "logging output in $log_file"
julia --project test/test_TM_symmetry.jl &> "$log_file"
echo "Done running test/test_TM_symmetry.jl"
