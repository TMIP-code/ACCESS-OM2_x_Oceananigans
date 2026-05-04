#!/bin/bash

#PBS -P y99
#PBS -l mem=16GB
#PBS -q express
#PBS -l ngpus=0
#PBS -l ncpus=2
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/test
mkdir -p "$log_dir"
log_file="$log_dir/jld2writer_${job_id}.log"

echo "Running test/test_jld2writer_deadlock.jl (PARENT_MODEL=$PARENT_MODEL)"
echo "logging output in $log_file"
mpiexec -n 2 julia $JULIA_BOUNDS_FLAG --project test/test_jld2writer_deadlock.jl &> "$log_file"
echo "Done running test/test_jld2writer_deadlock.jl"
