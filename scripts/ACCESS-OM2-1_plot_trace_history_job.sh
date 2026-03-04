#!/bin/bash

#PBS -N plot_trace_OM2-1
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=02:00:00
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

echo "Running plot_trace_history.jl on CPU"
log_dir=logs/julia/plot_trace_history
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_trace_history.jl &> "$log_dir/plot_trace_history_${MODEL_CONFIG}_${job_id}.log"
echo "Done running plot_trace_history.jl"
