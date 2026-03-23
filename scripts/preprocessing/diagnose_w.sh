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

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/preprocess
mkdir -p "$log_dir"

echo "Running diagnose_w.jl for PARENT_MODEL=$PARENT_MODEL"
julia --project $JULIA_BOUNDS_FLAG src/diagnose_w.jl &> "$log_dir/diagnose_w_${job_id}.log"
echo "Done diagnosing w for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/diagnose_w_${job_id}.log"
