#!/bin/bash

#PBS -N OM2-1_vel
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

# Load CUDA module if running on a GPU node
if [[ "$HOSTNAME" == *gpu* ]]; then
    echo "Loading CUDA module (GPU node detected)"
    module load cuda/12.9.0
    export JULIA_CUDA_USE_COMPAT=false
fi

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/$PARENT_MODEL/preprocess
mkdir -p "$log_dir"

echo "Running create_velocities.jl for PARENT_MODEL=$PARENT_MODEL"
julia --project src/create_velocities.jl &> "$log_dir/create_velocities_${job_id}.log"
echo "Done preprocessing velocities for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/create_velocities_${job_id}.log"
