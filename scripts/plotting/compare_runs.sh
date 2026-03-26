#!/bin/bash

#PBS -P y99
#PBS -l walltime=01:00:00
#PBS -l mem=47GB
#PBS -q express
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
source scripts/env_defaults.sh

: "${SOURCE_A:?SOURCE_A required}" "${SOURCE_B:?SOURCE_B required}"
COMPARE_LABEL=${COMPARE_LABEL:-compare}

log_dir="logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/plot/compare"
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "compare_runs.jl: $COMPARE_LABEL"
echo "  SOURCE_A=$SOURCE_A"
echo "  SOURCE_B=$SOURCE_B"

SOURCE_A="$SOURCE_A" SOURCE_B="$SOURCE_B" COMPARE_LABEL="$COMPARE_LABEL" \
julia --project test/compare_runs.jl \
    &> "$log_dir/${COMPARE_LABEL}_${job_id}.log"

echo "Done — $COMPARE_LABEL"
