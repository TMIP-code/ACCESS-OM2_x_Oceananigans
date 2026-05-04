#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=80GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_allocbench_${job_id}.log"

NGPUS="${PBS_NGPUS:-1}"
JULIA_CMD="julia $JULIA_BOUNDS_FLAG --project"

if [ "$NGPUS" -gt 1 ]; then
    echo "Running alloc benchmark (NGPUS=$NGPUS)"
    echo "logging output in $log_file"
    mpiexec --bind-to socket --map-by socket -n $NGPUS --report-bindings $JULIA_CMD \
        src/run_alloc_benchmark.jl &> "$log_file"
else
    echo "Running alloc benchmark (serial)"
    echo "logging output in $log_file"
    $JULIA_CMD src/run_alloc_benchmark.jl &> "$log_file"
fi
echo "Done running src/run_alloc_benchmark.jl for PARENT_MODEL=$PARENT_MODEL"
