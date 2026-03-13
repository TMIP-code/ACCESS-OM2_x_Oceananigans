#!/bin/bash

#PBS -P y99
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

TM_LABEL_X="${TM_LABEL_X:-const}"
TM_LABEL_Y="${TM_LABEL_Y:-avg24}"

echo "Running plot_TM_datashader.jl on CPU: $TM_LABEL_Y vs $TM_LABEL_X"
log_dir=logs/julia/$PARENT_MODEL/TM/plot
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_TM_datashader.jl &> "$log_dir/datashader_${TM_LABEL_Y}_vs_${TM_LABEL_X}_${MODEL_CONFIG}_${job_id}.log"
echo "Done running plot_TM_datashader.jl"
