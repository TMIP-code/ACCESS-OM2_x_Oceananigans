#!/bin/bash

#PBS -N OM2-1_grid
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

PARENT_MODEL=ACCESS-OM2-1
export PARENT_MODEL

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "PARENT_MODEL=$PARENT_MODEL, repo_root=$repo_root"
cd $repo_root

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/preprocess
mkdir -p "$log_dir"

echo "Creating grid for PARENT_MODEL=$PARENT_MODEL"
julia --project src/create_grid.jl &> "$log_dir/create_grid_${PARENT_MODEL}_${job_id}.log"
echo "Done creating grid for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/create_grid_${PARENT_MODEL}_${job_id}.log"
