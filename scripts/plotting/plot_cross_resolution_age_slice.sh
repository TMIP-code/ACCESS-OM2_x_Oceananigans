#!/bin/bash
#
# PBS wrapper for src/plot_cross_resolution_age_slice.jl — renders the single
# 3×3 cross-resolution + cross-decade depth-slice comparison of the forward
# (or adjoint, TRAF=yes) periodic age, combining the four panels of "Figure 1a"
# of the cross-resolution ventilation paper plus the resolution/decade diffs.
#
# Reads the forward age_periodic_1year.jld2 1-year FieldTimeSeries for BOTH
# resolutions (OM2-1 and OM2-025) and BOTH time windows, so the per-model
# model_config tags differ — they are configured in the Julia script and can be
# overridden via MODEL_CONFIG_OM21 / MODEL_CONFIG_OM2025 / SOLVER_TAG / TW1 /
# TW2 / DEPTH / TRAF environment variables.
#
# OM2-1 → OM2-025 regridding for the Δ-resolution column uses
# ConservativeRegridding.jl (masked-conservative on the 2-D slices).
#
# Usage:
#   qsub scripts/plotting/plot_cross_resolution_age_slice.sh
#   qsub -v "DEPTH=1000" scripts/plotting/plot_cross_resolution_age_slice.sh
#
# Writes: outputs/cross_resolution/age_slice/age_slice_{depth}m_{forward,adjoint}_3x3.png

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

log_dir=logs/julia/cross_resolution
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

# env_defaults.sh pins JULIA_NUM_THREADS=1 (for MPI jobs); the conservative
# regridder build is multithreaded, so use the allocated CPUs here instead.
export JULIA_NUM_THREADS="${PBS_NCPUS:-12}"

echo "Running plot_cross_resolution_age_slice.jl (DEPTH=${DEPTH:-2000}, TRAF=${TRAF:-no}, threads=$JULIA_NUM_THREADS)"
julia --project src/plot_cross_resolution_age_slice.jl \
    &> "$log_dir/plot_cross_resolution_age_slice_${job_id}.log"
echo "Done; log: $log_dir/plot_cross_resolution_age_slice_${job_id}.log"
