#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}.log"

NGPUS="${PBS_NGPUS:-1}"
JULIA_CMD="julia $JULIA_BOUNDS_FLAG --project"

# Nsight Systems profiling: set PROFILE=yes to wrap the run with nsys profile.
# Produces a .nsys-rep file in the log directory for analysis with Nsight Systems GUI.
# Following Oceananigans' distributed_scaling pattern: nsys wraps the entire MPI launcher
# (not each rank individually), producing a single profile covering all ranks.
# Also sets JULIA_NVTX_CALLBACKS=gc to trace Julia GC events.
PROFILE="${PROFILE:-no}"
NSYS_PREFIX=""
if [ "$PROFILE" = "yes" ]; then
    export JULIA_NVTX_CALLBACKS=gc
    # Profile to local SSD ($PBS_JOBFS) then copy back — avoids network FS distortion.
    # Following carstenbauer/JuliaHLRS22 pattern: nsys wraps the entire MPI launcher.
    # Use --mpi-impl=openmpi for MPI call tracing, --wait=primary to handle re-parented
    # processes, --stats=true for inline summary.
    PROFILE_DIR="${PBS_JOBFS:-/tmp}/nsys_profiles"
    mkdir -p "$PROFILE_DIR"
    profile_final="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}_profile"
    NSYS_PREFIX="nsys profile --trace=nvtx,cuda,mpi --mpi-impl=openmpi --cuda-memory-usage=true --wait=primary --stats=true --force-overwrite=true --output=${PROFILE_DIR}/profile"
    echo "PROFILE=yes: nsys profiling to ${PROFILE_DIR}, will copy to ${profile_final}.nsys-rep"
fi

JULIA_LAUNCHER="$JULIA_CMD"
[ "$NGPUS" -gt 1 ] && JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_CMD"
[ -n "$NSYS_PREFIX" ] && JULIA_LAUNCHER="$NSYS_PREFIX $JULIA_LAUNCHER"

echo "Running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL (NGPUS=$NGPUS)"
echo "logging output in $log_file"
$JULIA_LAUNCHER src/run_1year_benchmark.jl &> "$log_file"
echo "Done running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL"

# Copy nsys profiles from local SSD to persistent storage
if [ "$PROFILE" = "yes" ]; then
    echo "Copying nsys profiles from $PROFILE_DIR to $run_log_dir"
    cp "${PROFILE_DIR}"/*.nsys-rep "$run_log_dir/" 2>/dev/null && \
        echo "Profiles copied: $(ls ${run_log_dir}/*nsys-rep 2>/dev/null | wc -l) files" || \
        echo "WARNING: No .nsys-rep files found in $PROFILE_DIR"
    ls -lh "${PROFILE_DIR}"/ 2>/dev/null
fi
