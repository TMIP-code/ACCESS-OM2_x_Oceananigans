#!/bin/bash

#PBS -P y99
#PBS -l mem=192GB
#PBS -q gpuvolta
#PBS -l ngpus=2
#PBS -l ncpus=24
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
run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/test
mkdir -p "$run_log_dir"
log_file="$run_log_dir/probe_fts_halo_${PARTITION}_${job_id}.log"

NGPUS="${PBS_NGPUS:-0}"
NCPUS="${PBS_NCPUS:-1}"
JULIA_LAUNCHER="julia $JULIA_BOUNDS_FLAG --project"
if [ "$NGPUS" -gt 1 ]; then
    JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS --report-bindings $JULIA_LAUNCHER"
elif [ "$NGPUS" -eq 0 ] && [ "$NCPUS" -gt 1 ]; then
    JULIA_LAUNCHER="mpiexec -n $NCPUS $JULIA_LAUNCHER"
fi

echo "Running test/probe_fts_halo.jl (PARTITION=$PARTITION NGPUS=$NGPUS NCPUS=$NCPUS)"
echo "logging output in $log_file"
$JULIA_LAUNCHER test/probe_fts_halo.jl &> "$log_file"
echo "Done"
