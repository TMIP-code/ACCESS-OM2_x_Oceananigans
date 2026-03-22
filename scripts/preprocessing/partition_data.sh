#!/bin/bash

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l ncpus=4
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
log_file="$log_dir/partition_data_${PARTITION}_${job_id}.log"

NCPUS="${PBS_NCPUS:-$RANKS}"

echo "Partitioning data for PARENT_MODEL=$PARENT_MODEL, PARTITION=$PARTITION (NCPUS=$NCPUS)"
echo "logging output in $log_file"
mpiexec -n $NCPUS julia $JULIA_BOUNDS_FLAG --project src/partition_data.jl &> "$log_file"
echo "Done partitioning data"
