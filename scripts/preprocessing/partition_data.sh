#!/bin/bash

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
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

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/preprocess
mkdir -p "$log_dir"
log_file="$log_dir/partition_data_${PARTITION}_${job_id}.log"

# Use RANKS (not PBS_NCPUS) for MPI: extra CPUs may be requested for memory
# headroom on express/normal queues without spawning extra MPI ranks.
NMPI="$RANKS"

echo "Partitioning data for PARENT_MODEL=$PARENT_MODEL, PARTITION=$PARTITION (RANKS=$NMPI, PBS_NCPUS=${PBS_NCPUS:-?})"
echo "logging output in $log_file"
mpiexec -n $NMPI julia $JULIA_BOUNDS_FLAG --project src/partition_data.jl &> "$log_file"
echo "Done partitioning data"
