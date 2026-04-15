#!/bin/bash

#PBS -P y99
#PBS -q express
#PBS -l mem=47GB
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

echo "Running animate_MOC_rho_timeseries.jl on CPU"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/plot
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/animate_MOC_rho_timeseries.jl &> "$log_dir/animate_MOC_rho_timeseries_${job_id}.log"
echo "Done"
