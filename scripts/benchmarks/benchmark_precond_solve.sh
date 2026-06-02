#!/bin/bash

#PBS -P y99
#PBS -q hugemem
#PBS -l ncpus=48
#PBS -l mem=1470GB
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Benchmark the NK preconditioner's coarsened factorization+solve for one
# LUMP_AND_SPRAY value. One job per coarsening factor so PBS resources_used.mem
# cleanly attributes peak memory to each. See src/benchmark_precond_solve.jl.

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Export benchmark-specific env vars if set
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${MATRIX_PROCESSING:-}" ] && export MATRIX_PROCESSING && echo "MATRIX_PROCESSING=$MATRIX_PROCESSING"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"

# Pardiso threads: default to the full node's CPUs
export PARDISO_NPROCS=${PARDISO_NPROCS:-${PBS_NCPUS:-48}}
echo "PARDISO_NPROCS=$PARDISO_NPROCS"

run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/TM/benchmarks
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Benchmarking preconditioner solve for MODEL_CONFIG=$MODEL_CONFIG, TM_SOURCE=$TM_SOURCE, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, MATRIX_PROCESSING=$MATRIX_PROCESSING"
log_file="$run_log_dir/precond_solve_${MODEL_CONFIG}_${TM_SOURCE}_${LUMP_AND_SPRAY}_${LINEAR_SOLVER}_${MATRIX_PROCESSING}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project src/benchmark_precond_solve.jl &> "$log_file"
echo "Done benchmarking preconditioner solve"
echo "logged output in $log_file"
