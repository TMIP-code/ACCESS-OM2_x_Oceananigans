#!/bin/bash
#
# MPI ping-pong bandwidth probe (intra-node vs inter-node, CPU + GPU buffers).
#
# Diagnoses whether CUDA-aware MPI is using GPUDirect RDMA across the IB
# fabric, or silently host-staging GPU buffers. See test/test_mpi_bandwidth.jl
# for the decision rule.
#
# We run on gpuvolta (4× V100 per node) because that queue is much less
# busy than gpuhopper. The diagnosis transfers: if CUDA-aware MPI is
# host-staging on V100, it'll do the same on H200. If it works on V100,
# at least the Julia/MPI/UCX stack is correct and we can move on to
# gpuhopper-specific issues (e.g. fabric driver on the H200 nodes).
#
# Submit:  qsub scripts/tests/run_mpi_bandwidth_test.sh

#PBS -N mpi_bw_test
#PBS -P y99
#PBS -q gpuvolta
#PBS -l ngpus=8
#PBS -l ncpus=96
#PBS -l mem=768GB
#PBS -l walltime=00:20:00
#PBS -l jobfs=10GB
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l wd
#PBS -o logs/PBS/
#PBS -e logs/PBS/

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

# Force a 1×8 partition just so env_defaults.sh / compute_resources.sh see
# 8 ranks (the test itself doesn't use Oceananigans, but env_defaults.sh
# expects PARTITION-derived vars to be consistent).
export PARTITION=1x8
export GPU_QUEUE=gpuvolta
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/test/mpi_bandwidth
mkdir -p "$log_dir"
log_file="$log_dir/mpi_bandwidth_${job_id}.log"

NGPUS="${PBS_NGPUS:-8}"

echo "Running test/test_mpi_bandwidth.jl on $NGPUS GPUs (gpuvolta, 2 nodes × 4 V100)"
echo "logging output in $log_file"

mpiexec --bind-to socket --map-by socket -n "$NGPUS" --report-bindings \
    julia $JULIA_BOUNDS_FLAG --project test/test_mpi_bandwidth.jl \
    2>&1 | tee "$log_file"

echo "Done — see $log_file"
