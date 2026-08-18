#!/bin/bash

#PBS -P y99
#PBS -l mem=96GB
#PBS -q normal
#PBS -l ncpus=24
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Read-only advective-CFL probe of the preprocessed monthly u/v/w input
# FieldTimeSeries, using Oceananigans' own cell_advection_timescale. CPU only,
# no model run. One snapshot in memory at a time (OnDisk backend); the extra
# 96 GB headroom is for the materialised CFL field used to locate the worst
# cell (set CFL_LOCATE=no to skip it and halve the memory).
#
# Submitted via the driver's `probeCFL` step, e.g.:
#   PARENT_MODEL=ACCESS-OM2-01 EXPERIMENT=01deg_jra55v140_iaf_cycle4 \
#   TIME_WINDOW=1968-1977 GRID_HZ=4 JOB_CHAIN=probeCFL bash scripts/driver.sh
#
# Optional: CFL_TARGET (default 0.7), CFL_LOCATE (yes|no), WALLTIME_PROBE.

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/probe
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
echo "Probing input-velocity CFL for $EXPERIMENT ($TIME_WINDOW), TIMESTEP_MULT=$TIMESTEP_MULT"
julia --project src/probe_CFL.jl &> "$log_dir/CFL_${job_id}.log"
echo "Done. Log: $log_dir/CFL_${job_id}.log"
