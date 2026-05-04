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
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/plot/partitions
mkdir -p "$log_dir"
log_file="$log_dir/plot_partitioned_fts_${PARTITION}_${job_id}.log"

echo "Running plot_partitioned_fts.jl (PARTITION=$PARTITION)"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project src/plot_partitioned_fts.jl &> "$log_file"
echo "Done plot_partitioned_fts.jl (PARTITION=$PARTITION)"
