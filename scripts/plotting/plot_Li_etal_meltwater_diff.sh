#!/bin/bash

#PBS -P y99
#PBS -l mem=512GB
#PBS -q hugemem
#PBS -l ncpus=18
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Meltwater-difference plots (wthmp − wthp) for the Li et al. (2023) OM2-01
# experiments. Runs on CPU. MODEL_CONFIG (built by env_defaults.sh from the
# config env vars; add TRAF=yes for the adjoint _traf tree) selects forward
# ideal age vs adjoint time-to-re-emergence. Resources match plotNK at 0.1°
# (hugemem, full-res figures from two 25-snapshot FTS).
#
# Example (forward age):
#   PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=2040-2050 GRID_HZ=4 \
#   ADVECTION_SCHEME=upwind3 PARTITION=1x4 \
#   qsub -v PARENT_MODEL,TIME_WINDOW,GRID_HZ,ADVECTION_SCHEME,PARTITION \
#     scripts/plotting/plot_Li_etal_meltwater_diff.sh
# For the adjoint (time to re-emergence) add TRAF=yes.

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Baseline (wind+thermal only) and perturbed (with meltwater) experiments.
# Diff panels render B − A = wthmp − wthp = the meltwater contribution.
export EXPERIMENT_A=${EXPERIMENT_A:-01deg_jra55v13_ryf9091_qian_wthp}
export EXPERIMENT_B=${EXPERIMENT_B:-01deg_jra55v13_ryf9091_qian_wthmp}

echo "Meltwater diff: A=$EXPERIMENT_A  B=$EXPERIMENT_B  MC=$MODEL_CONFIG  TRAF=$TRAF"
log_dir=logs/julia/$PARENT_MODEL/comparisons/Li_etal_meltwater/$LOG_TW_TAG
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/plot_Li_etal_meltwater_diff.jl &> "$log_dir/${MODEL_CONFIG}_${job_id}.log"
echo "Done running plot_Li_etal_meltwater_diff.jl"
