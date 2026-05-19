#!/bin/bash

#PBS -P y99
#PBS -l mem=96GB
#PBS -q normal
#PBS -l ncpus=24
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Phase toggles — defaults run all of Phase 1/2/3 (Phase 3b skipped)
[ -n "${RUN_PHASE1:-}"  ] && export RUN_PHASE1  && echo "RUN_PHASE1=$RUN_PHASE1"
[ -n "${RUN_PHASE2:-}"  ] && export RUN_PHASE2  && echo "RUN_PHASE2=$RUN_PHASE2"
[ -n "${RUN_PHASE3:-}"  ] && export RUN_PHASE3  && echo "RUN_PHASE3=$RUN_PHASE3"
[ -n "${RUN_PHASE3B:-}" ] && export RUN_PHASE3B && echo "RUN_PHASE3B=$RUN_PHASE3B"
[ -n "${REGRID_DIRECTION:-}" ] && export REGRID_DIRECTION && echo "REGRID_DIRECTION=$REGRID_DIRECTION"
[ -n "${MC_OM2_1:-}"   ] && export MC_OM2_1   && echo "MC_OM2_1=$MC_OM2_1"
[ -n "${MC_OM2_025:-}" ] && export MC_OM2_025 && echo "MC_OM2_025=$MC_OM2_025"

echo "Running compare_NK_ages.jl on CPU"
log_dir=logs/julia/comparisons/NK_age
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project src/compare_NK_ages.jl &> "$log_dir/compare_NK_ages_${job_id}.log"
echo "Done running compare_NK_ages.jl"
