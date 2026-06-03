#!/bin/bash
#
# PBS wrapper for src/plot_cross_resolution_ventilation.jl — the 3×3
# cross-resolution + cross-decade surface-ventilation MAP figure (no
# zonal-integral side panels). Reads ventilation.jld2 for BOTH resolutions and
# BOTH time windows; regrids OM2-1 → OM2-025 via ConservativeRegridding.jl for
# the Δ-resolution column. Forward by default; TRAF=yes for the adjoint leg.
#
# Usage:
#   qsub scripts/plotting/plot_cross_resolution_ventilation.sh
#
# Writes: outputs/cross_resolution/ventilation/calVdown_{forward,adjoint}_3x3.png

#PBS -P y99
#PBS -q normal
#PBS -l mem=96GB
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

# Conservative regridder build is multithreaded (env_defaults pins to 1).
export JULIA_NUM_THREADS="${PBS_NCPUS:-12}"

log_dir=logs/julia/cross_resolution
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Running plot_cross_resolution_ventilation.jl (TRAF=${TRAF:-no}, threads=$JULIA_NUM_THREADS)"
julia --project src/plot_cross_resolution_ventilation.jl \
    &> "$log_dir/plot_cross_resolution_ventilation_${job_id}.log"
echo "Done; log: $log_dir/plot_cross_resolution_ventilation_${job_id}.log"
