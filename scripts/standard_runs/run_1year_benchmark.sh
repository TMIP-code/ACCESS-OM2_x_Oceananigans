#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=80GB
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
# Produces .nsys-rep files in the log directory for analysis with Nsight Systems GUI.
# Uses --capture-range=cudaProfilerApi so nsys only records data during the
# benchmark section (CUDA.@profile external=true in run_1year_benchmark.jl).
# Serial: nsys wraps julia directly (--trace=nvtx,cuda).
# Distributed: bash -c wrapper profiles each rank with MPI tracing (--trace=nvtx,cuda,mpi).
# Also sets JULIA_NVTX_CALLBACKS=gc to trace Julia GC events.
PROFILE="${PROFILE:-no}"
if [ "$PROFILE" = "yes" ]; then
    export JULIA_NVTX_CALLBACKS=gc
    export BENCHMARK_STEPS="${BENCHMARK_STEPS:-20}"
    echo "PROFILE=yes: limiting simulation to BENCHMARK_STEPS=$BENCHMARK_STEPS"
    profile_base="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}_profile"
    if [ "$NGPUS" -gt 1 ]; then
        echo "PROFILE=yes: profiling all $NGPUS ranks with MPI tracing"
    else
        echo "PROFILE=yes: profiling serial run → ${profile_base}.nsys-rep"
    fi
fi

if [ "$PROFILE" = "yes" ] && [ "$NGPUS" -gt 1 ]; then
    echo "Running with MPI profiling (NGPUS=$NGPUS, all ranks)"
    echo "logging output in $log_file"
    mpiexec --bind-to socket --map-by socket -n "$NGPUS" bash -c "
        nsys profile \
            --trace=nvtx,cuda,mpi \
            --cuda-memory-usage=true \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --force-overwrite=true \
            --output=${profile_base}_rank\${OMPI_COMM_WORLD_RANK} \
            $JULIA_CMD src/run_1year_benchmark.jl
    " &> "$log_file"
elif [ "$PROFILE" = "yes" ]; then
    echo "Running with serial profiling"
    echo "logging output in $log_file"
    nsys profile --trace=nvtx,cuda --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true --output="${profile_base}" \
        $JULIA_CMD src/run_1year_benchmark.jl &> "$log_file"
elif [ "$NGPUS" -gt 1 ]; then
    echo "Running (NGPUS=$NGPUS)"
    echo "logging output in $log_file"
    mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_CMD \
        src/run_1year_benchmark.jl &> "$log_file"
else
    echo "Running (serial)"
    echo "logging output in $log_file"
    $JULIA_CMD src/run_1year_benchmark.jl &> "$log_file"
fi
echo "Done running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL"
