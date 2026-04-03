#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=80GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/alloc_profile
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_alloc_profile_${job_id}.log"

export ALLOC_PROFILE_DIR="$run_log_dir"
export ALLOC_SAMPLE_RATE="${ALLOC_SAMPLE_RATE:-0.01}"
export ALLOC_PROFILE_STEPS="${ALLOC_PROFILE_STEPS:-3}"
export NWARMUP_STEPS="${NWARMUP_STEPS:-3}"

NGPUS="${PBS_NGPUS:-1}"
JULIA_CMD="julia $JULIA_BOUNDS_FLAG --project"

echo "Running allocation profiler (ALLOC_PROFILE_STEPS=$ALLOC_PROFILE_STEPS, ALLOC_SAMPLE_RATE=$ALLOC_SAMPLE_RATE, NGPUS=$NGPUS)"
echo "logging output in $log_file"

if [ "$NGPUS" -gt 1 ]; then
    mpiexec --bind-to socket --map-by socket -n "$NGPUS" \
        $JULIA_CMD src/run_1year_alloc_profile.jl &> "$log_file"
else
    $JULIA_CMD src/run_1year_alloc_profile.jl &> "$log_file"
fi

echo "Done — allocation profile saved in $ALLOC_PROFILE_DIR"
