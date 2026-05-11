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

# Optional: COMPARE_TARGET=standardrun|nk_steady|nk_periodic (default standardrun).
# LINEAR_SOLVER + LUMP_AND_SPRAY are already exported by env_defaults.sh; they're
# only consulted by the Julia script when COMPARE_TARGET is one of the nk_ modes.
COMPARE_TARGET="${COMPARE_TARGET:-standardrun}"
export COMPARE_TARGET
echo "COMPARE_TARGET=$COMPARE_TARGET"

# Route logs to plot/{standardrun,periodic}/ so the three target modes
# don't share a single log file (matches the script's TSV output split).
case "$COMPARE_TARGET" in
    standardrun) log_subdir="standardrun" ;;
    nk_steady|nk_periodic) log_subdir="periodic" ;;
    *) echo "ERROR: COMPARE_TARGET must be standardrun|nk_steady|nk_periodic (got: $COMPARE_TARGET)" >&2; exit 1 ;;
esac

echo "Running plot_timestep_multiplier_sweep.jl on CPU"
log_dir="logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/plot/$log_subdir"
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_timestep_multiplier_sweep.jl &> "$log_dir/${MODEL_CONFIG}_timestep_multiplier_sweep_${COMPARE_TARGET}_${job_id}.log"
echo "Done running plot_timestep_multiplier_sweep.jl"
