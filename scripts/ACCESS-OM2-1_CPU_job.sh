#!/bin/bash

#PBS -N OM2-1_preproc
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

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

echo "Creating velocities on CPU for PARENT_MODEL=$PARENT_MODEL"
run_log_dir=logs/julia/$PARENT_MODEL/preprocess
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/create_velocities.jl &> "$run_log_dir/create_velocities_${job_id}.log"
echo "Done creating velocities on CPU for PARENT_MODEL=$PARENT_MODEL"
