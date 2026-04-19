#!/bin/bash

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

echo "Running plot_grid_metrics.jl on CPU"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/plot/grid
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_grid_metrics.jl &> "$log_dir/plot_grid_metrics_${job_id}.log"
echo "Done running plot_grid_metrics.jl"
echo "logged output in $log_dir/plot_grid_metrics_${job_id}.log"
