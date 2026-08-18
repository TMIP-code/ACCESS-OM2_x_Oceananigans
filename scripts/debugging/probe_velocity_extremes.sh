#!/bin/bash

#PBS -P y99
#PBS -l mem=48GB
#PBS -q normal
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Read-only scan of the preprocessed monthly input u/v/w/eta FieldTimeSeries for
# NaN/Inf and extreme magnitudes — hunting the source of the Qian OM2-01 NK
# forward-map NaN blow-up. CPU only, no model run. One snapshot in memory at a
# time (OnDisk), so 48 GB is ample despite ~80 GB per-field files.
#
# Submit per experiment, e.g.:
#   EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp PARENT_MODEL=ACCESS-OM2-01 \
#   TIME_WINDOW=2040-2050 GRID_HZ=4 \
#   qsub -v EXPERIMENT,PARENT_MODEL,TIME_WINDOW,GRID_HZ \
#     scripts/debugging/probe_velocity_extremes.sh

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/probe
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
echo "Probing velocity extremes for $EXPERIMENT ($TIME_WINDOW)"
julia --project src/probe_velocity_extremes.jl &> "$log_dir/velocity_extremes_${job_id}.log"
echo "Done. Log: $log_dir/velocity_extremes_${job_id}.log"
