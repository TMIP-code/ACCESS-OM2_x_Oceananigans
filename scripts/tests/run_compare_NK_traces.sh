#!/bin/bash
# Compare per-Φ!-call trace files between two NK runs (typically serial vs
# partitioned). CPU-only; reads JLD2 traces from the periodic/.../NK directories
# of each run and writes a divergence scan plot + spatial diff plots.

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l ngpus=0
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

# Solver/job IDs and partitioning come in via qsub -v
[ -n "${REF_JOB_ID:-}" ] && export REF_JOB_ID && echo "REF_JOB_ID=$REF_JOB_ID"
[ -n "${CMP_JOB_ID:-}" ] && export CMP_JOB_ID && echo "CMP_JOB_ID=$CMP_JOB_ID"
[ -n "${GPU_TAG:-}" ] && export GPU_TAG && echo "GPU_TAG=$GPU_TAG"
[ -n "${REF_GPU_TAG:-}" ] && export REF_GPU_TAG && echo "REF_GPU_TAG=$REF_GPU_TAG"
[ -n "${DIVERGE_TOL_YR:-}" ] && export DIVERGE_TOL_YR && echo "DIVERGE_TOL_YR=$DIVERGE_TOL_YR"

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/test
mkdir -p "$log_dir"
log_file="$log_dir/compare_NK_traces_${REF_JOB_ID:-noref}_vs_${CMP_JOB_ID:-nocmp}_${job_id}.log"

echo "Running test/compare_NK_traces.jl"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project test/compare_NK_traces.jl &> "$log_file"
echo "Done. Log: $log_file"
