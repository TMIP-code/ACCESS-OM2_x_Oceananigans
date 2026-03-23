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
# Produces .nsys-rep file(s) in the log directory for analysis with Nsight Systems GUI.
# For MPI runs, nsys wraps each rank individually (using %q{OMPI_COMM_WORLD_RANK} for
# per-rank output files), so each rank gets its own profile.
PROFILE="${PROFILE:-no}"
if [ "$PROFILE" = "yes" ]; then
    profile_output="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}_profile"
    if [ "$NGPUS" -gt 1 ]; then
        # MPI: nsys inside mpiexec, per-rank output files
        JULIA_CMD="nsys profile --trace=cuda,mpi,nvtx --cuda-memory-usage=true --output=${profile_output}_rank%q{OMPI_COMM_WORLD_RANK} $JULIA_CMD"
    else
        JULIA_CMD="nsys profile --trace=cuda,mpi,nvtx --cuda-memory-usage=true --output=$profile_output $JULIA_CMD"
    fi
    echo "PROFILE=yes: nsys output → ${profile_output}*.nsys-rep"
fi

JULIA_LAUNCHER="$JULIA_CMD"
[ "$NGPUS" -gt 1 ] && JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_CMD"

echo "Running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL (NGPUS=$NGPUS)"
echo "logging output in $log_file"
$JULIA_LAUNCHER src/run_1year_benchmark.jl &> "$log_file"
echo "Done running src/run_1year_benchmark.jl for PARENT_MODEL=$PARENT_MODEL"
