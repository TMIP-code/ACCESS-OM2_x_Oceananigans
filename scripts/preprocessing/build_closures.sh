#!/bin/bash

#PBS -P y99
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
# Closure preprocessing runs serially (a single Julia process); partitioned runs
# are set up later by the `partition` step. Pin PARTITION=1x1 BEFORE sourcing
# env_defaults so this job doesn't inherit the model's run-time partition (e.g.
# OM2-025's 1x2) and have select_architecture.jl try to build a multi-rank
# Distributed architecture under one MPI rank (ArgumentError, 2 ranks vs 1).
export PARTITION=1x1
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/preprocess
mkdir -p "$log_dir"

echo "Running prep_closures.jl for PARENT_MODEL=$PARENT_MODEL"
julia --project src/prep_closures.jl &> "$log_dir/prep_closures_${job_id}.log"
echo "Done preprocessing closures for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/prep_closures_${job_id}.log"
