#!/bin/bash

#PBS -N preprocess_OM2-1_CPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o scratch_output/PBS/
#PBS -e scratch_output/PBS/
#PBS -l wd

set -euo pipefail

PARENT_MODEL=ACCESS-OM2-1
export PARENT_MODEL

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENT_MODEL=$PARENT_MODEL, REPO_ROOT=$repo_root"
cd "$repo_root"

echo "Running preprocessing (interpolated + mass-transport velocities) for PARENT_MODEL=$PARENT_MODEL"
run_log_dir="$repo_root/scratch_output/runs/preprocess_${PARENT_MODEL}"
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project "$repo_root/src/create_velocities.jl" &> "$run_log_dir/preprocess.$job_id.out"
echo "Done preprocessing for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $run_log_dir/preprocess.$job_id.out"
