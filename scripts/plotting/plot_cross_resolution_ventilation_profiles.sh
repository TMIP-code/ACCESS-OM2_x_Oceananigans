#!/bin/bash
#
# PBS wrapper for src/plot_cross_resolution_ventilation_profiles.jl — the
# zonal-integral ventilation-profile corner-plot figure (one panel with the 4
# absolute curves + 4 difference panels: Δ decade below, Δ resolution right).
# Reads ventilation.jld2 for BOTH resolutions and BOTH time windows. The 1°
# latitude binning is grid-independent, so no regridding is needed (light job).
# Forward by default; TRAF=yes for the adjoint leg.
#
# Usage:
#   qsub scripts/plotting/plot_cross_resolution_ventilation_profiles.sh
#
# Writes: outputs/cross_resolution/ventilation/calVdown_profiles_{forward,adjoint}.png

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

log_dir=logs/julia/cross_resolution
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Running plot_cross_resolution_ventilation_profiles.jl (TRAF=${TRAF:-no})"
julia --project src/plot_cross_resolution_ventilation_profiles.jl \
    &> "$log_dir/plot_cross_resolution_ventilation_profiles_${job_id}.log"
echo "Done; log: $log_dir/plot_cross_resolution_ventilation_profiles_${job_id}.log"
