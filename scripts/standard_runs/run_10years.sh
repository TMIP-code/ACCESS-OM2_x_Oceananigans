#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
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
run_log_dir=logs/julia/$PARENT_MODEL/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_10years_${job_id}.log"

NGPUS="${PBS_NGPUS:-1}"
JULIA_LAUNCHER="julia $JULIA_BOUNDS_FLAG --project"
[ "$NGPUS" -gt 1 ] && JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_LAUNCHER"

echo "Running src/run_10years.jl for PARENT_MODEL=$PARENT_MODEL (NGPUS=$NGPUS)"
echo "logging output in $log_file"
$JULIA_LAUNCHER src/run_10years.jl &> "$log_file"
echo "Done running src/run_10years.jl for PARENT_MODEL=$PARENT_MODEL"
