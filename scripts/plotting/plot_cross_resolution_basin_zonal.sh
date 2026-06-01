#!/bin/bash
#
# PBS wrapper for src/plot_cross_resolution_basin_zonal.jl — renders three
# 3×3 cross-resolution + cross-decade basin zonal-mean figures (one per basin:
# Atlantic / Pacific / Indian) of the forward (or adjoint, TRAF=yes) periodic
# age — the per-basin analogue of "Figure 1c" of the cross-resolution paper.
#
# Reads the forward age_periodic_1year.jld2 1-year FieldTimeSeries for BOTH
# resolutions and BOTH time windows; the per-model model_config tags differ and
# can be overridden via MODEL_CONFIG_OM21 / MODEL_CONFIG_OM2025 / SOLVER_TAG /
# TW1 / TW2 / TRAF environment variables.
#
# The Δ-resolution column interpolates the OM2-1 zonal-mean profile onto the
# OM2-025 latitude axis (no 3-D regridding — the two grids share the same
# 50-level vertical grid).
#
# Usage:
#   qsub scripts/plotting/plot_cross_resolution_basin_zonal.sh
#
# Writes: outputs/cross_resolution/zonal/zonal_{atlantic,pacific,indian}_{forward,adjoint}_3x3.png

#PBS -P y99
#PBS -q normal
#PBS -l mem=48GB
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

echo "Running plot_cross_resolution_basin_zonal.jl (TRAF=${TRAF:-no})"
julia --project src/plot_cross_resolution_basin_zonal.jl \
    &> "$log_dir/plot_cross_resolution_basin_zonal_${job_id}.log"
echo "Done; log: $log_dir/plot_cross_resolution_basin_zonal_${job_id}.log"
