#!/bin/bash

#PBS -P y99
#PBS -l walltime=01:00:00
#PBS -l mem=47GB
#PBS -q express
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

echo "Running compare_prescribed_vs_diagnosed_w.jl on CPU"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/plot/compare_w
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project test/compare_prescribed_vs_diagnosed_w.jl \
    &> "$log_dir/${MODEL_CONFIG}_${job_id}.log"
echo "Done running compare_prescribed_vs_diagnosed_w.jl"
