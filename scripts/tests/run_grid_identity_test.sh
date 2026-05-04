#!/bin/bash

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l ngpus=0
#PBS -l ncpus=4
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
log_file="$log_dir/grid_identity_${job_id}.log"

NCPUS="${PBS_NCPUS:-4}"

echo "Running test/test_grid_identity.jl for PARENT_MODEL=$PARENT_MODEL (NCPUS=$NCPUS)"
echo "logging output in $log_file"
mpiexec -n $NCPUS julia $JULIA_BOUNDS_FLAG --project test/test_grid_identity.jl &> "$log_file"
echo "Done running test/test_grid_identity.jl"
