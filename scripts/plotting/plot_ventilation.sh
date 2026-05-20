#!/bin/bash

#PBS -P y99
#PBS -l mem=24GB
#PBS -q express
#PBS -l ncpus=4
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"

run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/plot/periodic
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Plotting ventilation diagnostic for MODEL_CONFIG=$MODEL_CONFIG"
log_file="$run_log_dir/plot_ventilation_${MODEL_CONFIG}_${job_id}.log"
julia --project src/plot_ventilation.jl &> "$log_file"
echo "Done plotting ventilation diagnostic"
echo "logged output in $log_file"
