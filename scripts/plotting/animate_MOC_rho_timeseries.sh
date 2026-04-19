#!/bin/bash
#
# PBS wrapper for src/animate_MOC_rho_timeseries.jl — records an mp4 of the
# full global density-space MOC timeseries (lat × σ₀, ~720 frames) from the
# NetCDF produced by src/compute_MOC_rho_timeseries.py.
#
# MODE controls which input/output file is used:
#   MODE=monthly     (default) — raw monthly ψ → MOC_rho_global_timeseries.mp4
#   MODE=rollingyear            — 12-month rolling mean → MOC_rho_global_rollingyear.mp4
#
# Usage:
#   qsub -v "PARENT_MODEL=ACCESS-OM2-1,MODE=monthly"     scripts/plotting/animate_MOC_rho_timeseries.sh
#   qsub -v "PARENT_MODEL=ACCESS-OM2-1,MODE=rollingyear" scripts/plotting/animate_MOC_rho_timeseries.sh
#
# Expects: /scratch/y99/TMIP/data/{PARENT_MODEL}/{EXPERIMENT}/rhospace/psi_tot{_rollingyear}_global.nc
# Writes:  outputs/{PARENT_MODEL}/{EXPERIMENT}/MOC_rho_global_{timeseries,rollingyear}.mp4
# Logs to: logs/julia/{PARENT_MODEL}/{EXPERIMENT}/plot/animate_MOC_rho_timeseries_<jobid>.log

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

echo "Running animate_MOC_rho_timeseries.jl on CPU"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/plot
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/animate_MOC_rho_timeseries.jl &> "$log_dir/animate_MOC_rho_timeseries_${job_id}.log"
echo "Done"
