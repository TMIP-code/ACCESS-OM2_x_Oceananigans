#!/bin/bash

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=80GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Make benchmark-only env vars visible to the Julia process for both
# profile and non-profile runs. Empty SYNC_GC_NSTEPS = "disabled" on the
# Julia side (parse to 0); the PROFILE block below overrides empty → 5.
export SYNC_GC_NSTEPS="${SYNC_GC_NSTEPS:-}"
export LOAD_BALANCE="${LOAD_BALANCE:-no}"

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
    # Default 240 = 20 batches × K=12 (= max divisor in {6, 9, 12, 18, 36};
    # also covers non-TB profiles cleanly). Override at submission with
    # BENCHMARK_STEPS=<n> if you need a shorter run.
    export BENCHMARK_STEPS="${BENCHMARK_STEPS:-240}"
    echo "PROFILE=yes: limiting simulation to BENCHMARK_STEPS=$BENCHMARK_STEPS"
    # Synchronized GC for distributed profiling (see docs/DISTRIBUTED_GC.md).
    # Default N=5 → ~48 fires across the 240-step benchmark; SYNC_GC_NSTEPS=0
    # disables. Under TBLOCKING the unit becomes batches (= N·K raw steps).
    export SYNC_GC_NSTEPS="${SYNC_GC_NSTEPS:-5}"
    if [ "$SYNC_GC_NSTEPS" -gt 0 ]; then
        sync_gc_tag="syncGCyes_N${SYNC_GC_NSTEPS}"
        echo "PROFILE=yes: synchronized GC every $SYNC_GC_NSTEPS iterations"
    else
        sync_gc_tag="syncGCno"
        echo "PROFILE=yes: synchronized GC disabled (baseline)"
    fi
    profile_base="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}_profile_${sync_gc_tag}"
    if [ "$NGPUS" -gt 1 ]; then
        echo "PROFILE=yes: profiling all $NGPUS ranks with MPI tracing"
    else
        echo "PROFILE=yes: profiling serial run → ${profile_base}.nsys-rep"
    fi
fi

if [ "$PROFILE" = "yes" ] && [ "$NGPUS" -gt 1 ]; then
    echo "Running with MPI profiling (NGPUS=$NGPUS, all ranks)"
    echo "logging output in $log_file"
    mpiexec --bind-to socket --map-by socket -n "$NGPUS" --report-bindings bash -c "
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
    mpiexec --bind-to socket --map-by socket -n $NGPUS --report-bindings $JULIA_CMD \
        src/run_1year_benchmark.jl &> "$log_file"
else
    echo "Running (serial)"
    echo "logging output in $log_file"
    $JULIA_CMD src/run_1year_benchmark.jl &> "$log_file"
fi
echo "Done running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL"
