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

GPU_TAG=${GPU_TAG:-2x2}
export GPU_TAG

echo "Running compare_1year_runs.jl on CPU (GPU_TAG=$GPU_TAG)"
log_dir=logs/julia/$PARENT_MODEL/plot/compare
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/compare_1year_runs.jl &> "$log_dir/${MODEL_CONFIG}_${GPU_TAG}_${job_id}.log"
echo "Done running compare_1year_runs.jl (GPU_TAG=$GPU_TAG)"
