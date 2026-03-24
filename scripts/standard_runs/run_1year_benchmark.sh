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
# Produces a .nsys-rep file in the log directory for analysis with Nsight Systems GUI.
# Following Oceananigans' distributed_scaling pattern: nsys wraps the entire MPI launcher
# (not each rank individually), producing a single profile covering all ranks.
# Also sets JULIA_NVTX_CALLBACKS=gc to trace Julia GC events.
PROFILE="${PROFILE:-no}"
if [ "$PROFILE" = "yes" ]; then
    export JULIA_NVTX_CALLBACKS=gc
    # Profile to local SSD ($PBS_JOBFS) then copy back — avoids network FS distortion.
    PROFILE_DIR="${PBS_JOBFS:-/tmp}/nsys_profiles"
    mkdir -p "$PROFILE_DIR"
    profile_final="$run_log_dir/${MODEL_CONFIG}_1yearfast_${job_id}_profile"
    if [ "$NGPUS" -gt 1 ]; then
        # MPI: profile only rank 0 via a wrapper script. Other ranks run plain julia.
        # Wrapping the entire mpiexec with nsys causes SIGTERM on Gadi (PBS + OpenMPI).
        # Per-rank nsys inside mpiexec also crashes. So we profile rank 0 only.
        PROFILE_WRAPPER="$PROFILE_DIR/nsys_wrapper.sh"
        cat > "$PROFILE_WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
    exec nsys profile --trace=nvtx,cuda --cuda-memory-usage=true --force-overwrite=true --output=PROFILE_OUTPUT_PLACEHOLDER "$@"
else
    exec "$@"
fi
WRAPPER_EOF
        sed -i "s|PROFILE_OUTPUT_PLACEHOLDER|${PROFILE_DIR}/profile_rank0|" "$PROFILE_WRAPPER"
        chmod +x "$PROFILE_WRAPPER"
        echo "PROFILE=yes: profiling rank 0 only → ${profile_final}_rank0.nsys-rep"
    else
        echo "PROFILE=yes: profiling serial run → ${profile_final}.nsys-rep"
    fi
fi

if [ "$PROFILE" = "yes" ] && [ "$NGPUS" -gt 1 ]; then
    JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $PROFILE_WRAPPER $JULIA_CMD"
elif [ "$PROFILE" = "yes" ]; then
    JULIA_LAUNCHER="nsys profile --trace=nvtx,cuda --cuda-memory-usage=true --force-overwrite=true --output=${PROFILE_DIR}/profile $JULIA_CMD"
elif [ "$NGPUS" -gt 1 ]; then
    JULIA_LAUNCHER="mpiexec --bind-to socket --map-by socket -n $NGPUS $JULIA_CMD"
else
    JULIA_LAUNCHER="$JULIA_CMD"
fi

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
