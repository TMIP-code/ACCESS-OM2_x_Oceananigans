#!/bin/bash

#PBS -N OM2-025_plt_sim
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
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

# Unlimited stack size
# ulimit -s unlimited

echo "Running plot_100years_age.jl on CPU"
log_dir=logs/julia/plot/standardrun
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_100years_age.jl &> "$log_dir/${MODEL_CONFIG}_100years_${job_id}.log"
echo "Done running plot_100years_age.jl"
