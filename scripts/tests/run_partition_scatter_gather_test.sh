#!/bin/bash
# Partitioned scatter/gather test (step 2c of the partitioned-NK plan).
# CPU-only MPI test: compare production 1D Scatterv/Gatherv against an
# Oceananigans-based 3D-field reference round-trip.

#PBS -P y99
#PBS -l mem=47GB
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

# Export partition + LB env so the Julia test reads them
export PARTITION_X PARTITION_Y LOAD_BALANCE

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/test
mkdir -p "$log_dir"
log_file="$log_dir/scattergather_${PARTITION_X}x${PARTITION_Y}_LB${LOAD_BALANCE}_${job_id}.log"

NRANKS="${RANKS:-$((PARTITION_X * PARTITION_Y))}"

echo "Running partitioned scatter/gather test on $NRANKS ranks"
echo "PARTITION=${PARTITION_X}x${PARTITION_Y}, LOAD_BALANCE=$LOAD_BALANCE"
echo "logging output in $log_file"
mpiexec --bind-to socket --map-by socket -n $NRANKS --report-bindings \
    julia $JULIA_BOUNDS_FLAG --project test/test_partition_scatter_gather.jl &> "$log_file"
echo "Done. Log: $log_file"
