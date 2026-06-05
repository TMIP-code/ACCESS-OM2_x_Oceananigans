#!/bin/bash

#PBS -P y99
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Offline-factorization toy (Phase 1, docs/offline_factorization.md).
# Proves the save -> load -> reuse cycle for the NK preconditioner matrix Q on the
# small OM2-1 LUMP_AND_SPRAY=2x2 case. Runs the `factor` and `reuse` phases as two
# separate julia processes so PBS / Sys.maxrss attribute the factorization peak and
# the load RAM cleanly. CPU only. See src/benchmark_offline_factor.jl.
#
# One job per OFFLINE_SOLVER (UMFPACK | PureUMFPACK | MUMPS) so memory is per-solver.

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

# Toy defaults: OM2-1, 2x2 coarsening, unsymmetric LU needs no structural symmetry.
# PARENT_MODEL must be set BEFORE sourcing env_defaults.sh (it resolves MODEL_CONFIG
# from it and errors if unset).
export PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
export LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-2x2}
export MATRIX_PROCESSING=${MATRIX_PROCESSING:-raw}
export TM_SOURCE=${TM_SOURCE:-const}
export OFFLINE_SOLVER=${OFFLINE_SOLVER:-UMFPACK}

source scripts/env_defaults.sh

echo "PARENT_MODEL=$PARENT_MODEL LUMP_AND_SPRAY=$LUMP_AND_SPRAY MATRIX_PROCESSING=$MATRIX_PROCESSING TM_SOURCE=$TM_SOURCE OFFLINE_SOLVER=$OFFLINE_SOLVER"

run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/TM/benchmarks
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

for ph in factor reuse; do
    export OFFLINE_PHASE=$ph
    log_file="$run_log_dir/offline_factor_${MODEL_CONFIG}_${TM_SOURCE}_${LUMP_AND_SPRAY}_${OFFLINE_SOLVER}_${ph}_${job_id}.log"
    echo "=== phase=$ph  ->  $log_file ==="
    julia $JULIA_BOUNDS_FLAG --project src/benchmark_offline_factor.jl &> "$log_file"
    echo "phase=$ph done"
done

echo "Done offline-factorization toy (OFFLINE_SOLVER=$OFFLINE_SOLVER)"
