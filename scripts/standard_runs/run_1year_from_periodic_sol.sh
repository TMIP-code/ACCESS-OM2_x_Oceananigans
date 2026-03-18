#!/bin/bash

#PBS -P y99
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Export solver-specific env vars if set
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"

lumpspray_tag="prec"
[ "${LUMP_AND_SPRAY:-no}" = "yes" ] && lumpspray_tag="LSprec"
solver_tag="${LINEAR_SOLVER:-Pardiso}_${lumpspray_tag}"

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/periodic/1year
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_${solver_tag}_${job_id}.log"

NGPUS="${PBS_NGPUS:-1}"
JULIA_LAUNCHER="julia $JULIA_BOUNDS_FLAG --project"
[ "$NGPUS" -gt 1 ] && JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_LAUNCHER"

echo "Running src/run_periodic_1year.jl for PARENT_MODEL=$PARENT_MODEL (NGPUS=$NGPUS)"
echo "logging output in $log_file"
$JULIA_LAUNCHER src/run_periodic_1year.jl &> "$log_file"
echo "Done running src/run_periodic_1year.jl for PARENT_MODEL=$PARENT_MODEL"
