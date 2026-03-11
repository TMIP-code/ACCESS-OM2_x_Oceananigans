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

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/$PARENT_MODEL/preprocess
mkdir -p "$log_dir"

echo "Creating grid for PARENT_MODEL=$PARENT_MODEL"
julia --project src/create_grid.jl &> "$log_dir/create_grid_${job_id}.log"
echo "Done creating grid for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/create_grid_${job_id}.log"
