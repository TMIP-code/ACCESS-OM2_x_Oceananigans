#!/bin/bash
#
# PBS wrapper for src/plot_AABW_NADW_timeseries.jl — renders one 2x1 PNG per
# resolution combining the AABW timeseries (global ψ) with 4 NADW timeseries
# (Atlantic ψ). Requires psi_tot_global.nc AND psi_tot_atlantic.nc to exist
# under /scratch/y99/TMIP/data/{model}/{experiment}/rhospace/ for each model.
#
# Usage:
#   qsub scripts/plotting/plot_AABW_NADW_timeseries.sh
#
# Writes:  outputs/{model}/{experiment}/AABW_NADW/AABW_NADW_rhospace_timeseries.png

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

log_dir=logs/julia/AABW_NADW
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Running plot_AABW_NADW_timeseries.jl"
julia --project src/plot_AABW_NADW_timeseries.jl \
    &> "$log_dir/plot_AABW_NADW_timeseries_${job_id}.log"
echo "Done; log: $log_dir/plot_AABW_NADW_timeseries_${job_id}.log"
