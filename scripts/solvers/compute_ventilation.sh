#!/bin/bash

#PBS -P y99
#PBS -l mem=24GB
#PBS -q express
#PBS -l ncpus=4
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Export solver-specific env vars if set (resolves the converged NK file name)
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"

lumpspray_tag="prec"
[ "${LUMP_AND_SPRAY:-no}" = "yes" ] && lumpspray_tag="LSprec"
solver_tag="${LINEAR_SOLVER:-Pardiso}_${lumpspray_tag}"

run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/periodic/ventilation
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Computing ventilation diagnostic for MODEL_CONFIG=$MODEL_CONFIG (solver=$solver_tag)"
log_file="$run_log_dir/ventilation_${MODEL_CONFIG}_${solver_tag}_${job_id}.log"
julia --project src/compute_ventilation_diagnostic.jl &> "$log_file"
echo "Done computing ventilation diagnostic"
echo "logged output in $log_file"
