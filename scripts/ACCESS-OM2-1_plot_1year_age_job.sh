#!/bin/bash

#PBS -N plot_1year_age_OM2-1
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

echo "Running plot_1year_age.jl on CPU"
log_dir=logs/julia/plot_1year_age
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_1year_age.jl &> "$log_dir/plot_1year_age_${MODEL_CONFIG}_${job_id}.log"
echo "Done running plot_1year_age.jl"
