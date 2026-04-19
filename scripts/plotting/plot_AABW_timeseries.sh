#!/bin/bash
#
# PBS wrapper for src/plot_AABW_timeseries.jl — plots AABW transport timeseries
# for all configured models with candidate TIME_WINDOWs bracketed. Reads from
# /scratch/y99/TMIP/data/{model}/{experiment}/{depthspace|rhospace}/ depending
# on SPACE (default depth).
#
# Usage:
#   qsub scripts/plotting/plot_AABW_timeseries.sh                   # depth-space (default)
#   qsub -v "SPACE=rho" scripts/plotting/plot_AABW_timeseries.sh    # density-space
#
# Writes:  outputs/{model}/{experiment}/AABW/*.png and *_windows.txt

#PBS -P y99
#PBS -q express
#PBS -l mem=47GB
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

SPACE=${SPACE:-depth}
log_dir=logs/julia/AABW
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Running plot_AABW_timeseries.jl (SPACE=$SPACE)"
SPACE=$SPACE julia --project src/plot_AABW_timeseries.jl \
    &> "$log_dir/plot_AABW_timeseries_${SPACE}_${job_id}.log"
echo "Done"
