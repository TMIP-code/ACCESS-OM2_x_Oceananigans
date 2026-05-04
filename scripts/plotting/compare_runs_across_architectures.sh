#!/bin/bash

#PBS -P y99
#PBS -l walltime=01:00:00
#PBS -l mem=47GB
#PBS -q express
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

GPU_TAG=${GPU_TAG:-2x2}
DURATION_TAG=${DURATION_TAG:-1year}
export GPU_TAG DURATION_TAG

echo "Running compare_runs_across_architectures.jl on CPU (GPU_TAG=$GPU_TAG, DURATION_TAG=$DURATION_TAG)"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/plot/compare
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project test/compare_runs_across_architectures.jl &> "$log_dir/${MODEL_CONFIG}_${GPU_TAG}_${DURATION_TAG}_${job_id}.log"
echo "Done running compare_runs_across_architectures.jl (GPU_TAG=$GPU_TAG, DURATION_TAG=$DURATION_TAG)"
