#!/bin/bash

#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
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

# Export solver-specific env vars if set
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${OMEGA:-}" ] && export OMEGA && echo "OMEGA=$OMEGA"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"

lumpspray_tag="prec"
[ "${LUMP_AND_SPRAY:-no}" = "yes" ] && lumpspray_tag="LSprec"
solver_tag="${LINEAR_SOLVER:-Pardiso}_${lumpspray_tag}"

echo "Running combine_periodic_1year.jl on CPU (stitch per-rank 1-year age into combined file)"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/postprocess
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/combine_periodic_1year.jl &> "$log_dir/combine1yr_${MODEL_CONFIG}_${solver_tag}_${job_id}.log"
echo "Done running combine_periodic_1year.jl"
