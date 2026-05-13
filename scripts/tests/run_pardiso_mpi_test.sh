#!/bin/bash
# Pardiso-under-MPI test (step 1 of the partitioned-NK plan).
# Runs julia first as a serial baseline (no mpiexec), then under
# mpiexec -n $NGPUS with the same socket binding that production NK uses.

#PBS -P y99
#PBS -l mem=96GB
#PBS -q gpuvolta
#PBS -l ngpus=2
#PBS -l ncpus=24
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=10GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"
[ -n "${PARDISO_NPROCS_SWEEP:-}" ] && export PARDISO_NPROCS_SWEEP && echo "PARDISO_NPROCS_SWEEP=$PARDISO_NPROCS_SWEEP"

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/test
mkdir -p "$log_dir"

NGPUS="${PBS_NGPUS:-2}"

# ── Baseline: single process, no mpiexec (known-good reference) ──
baseline_log="$log_dir/pardisompi_baseline_${MODEL_CONFIG}_${job_id}.log"
echo "[baseline] Running serial Pardiso test (no mpiexec)"
echo "logging to $baseline_log"
julia $JULIA_BOUNDS_FLAG --project test/test_pardiso_mpi.jl &> "$baseline_log" \
    || echo "[baseline] FAILED (see $baseline_log)"

# ── MPI sweep: same launch flags production NK uses ──
mpi_log="$log_dir/pardisompi_mpi_${MODEL_CONFIG}_${job_id}.log"
echo "[mpi] Running Pardiso under mpiexec -n $NGPUS (socket binding)"
echo "logging to $mpi_log"
mpiexec --bind-to socket --map-by socket -n $NGPUS --report-bindings \
    julia $JULIA_BOUNDS_FLAG --project test/test_pardiso_mpi.jl &> "$mpi_log"

echo "Done. Logs:"
echo "  baseline: $baseline_log"
echo "  mpi:      $mpi_log"
