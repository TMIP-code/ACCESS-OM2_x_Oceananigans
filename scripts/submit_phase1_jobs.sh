#!/bin/bash
# Submit Phase 1 (OM2-1) simulation jobs with clean job ID capture
# Usage: bash scripts/submit_phase1_jobs.sh

set -e

MODEL=ACCESS-OM2-1
GPU_Q=gpuvolta
GRID=13

declare -A jobs
job_count=0

# Helper function to submit a job and capture ID cleanly
submit_job() {
    local label=$1
    shift
    local output
    output=$(bash scripts/driver.sh "$@" 2>&1)
    local job_id=$(echo "$output" | grep -E '^ +run1yrfast +[0-9]+\.gadi-pbs' | awk '{print $NF}')
    if [ -z "$job_id" ]; then
        echo "ERROR: Failed to extract job ID for $label"
        return 1
    fi
    jobs[$job_count]="$label: $job_id"
    echo "$label: $job_id"
    ((job_count++))
}

echo "=== Submitting Phase 1 (OM2-1) Simulation Jobs ==="
echo

# 1x1 baseline
submit_job "1x1 baseline bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID JOB_CHAIN=run1yrfast
submit_job "1x1 baseline nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x2 baseline
submit_job "1x2 baseline bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 JOB_CHAIN=run1yrfast
submit_job "1x2 baseline nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x2 +GC
submit_job "1x2 +GC bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 SYNC_GC_NSTEPS=5 JOB_CHAIN=run1yrfast
submit_job "1x2 +GC nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 SYNC_GC_NSTEPS=5 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x2 +TB
submit_job "1x2 +TB bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 TBLOCKING=12 JOB_CHAIN=run1yrfast
submit_job "1x2 +TB nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 TBLOCKING=12 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x2 +LB
submit_job "1x2 +LB bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 LOAD_BALANCE=cell JOB_CHAIN=run1yrfast
submit_job "1x2 +LB nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x2 LOAD_BALANCE=cell PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x4 baseline
submit_job "1x4 baseline bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x4 JOB_CHAIN=run1yrfast
submit_job "1x4 baseline nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x4 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

# 1x8 baseline
submit_job "1x8 baseline bench" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x8 JOB_CHAIN=run1yrfast
submit_job "1x8 baseline nsys" PARENT_MODEL=$MODEL GPU_QUEUE=$GPU_Q GRID_HX=$GRID GRID_HY=$GRID PARTITION=1x8 PROFILE=yes BENCHMARK_STEPS=240 JOB_CHAIN=run1yrfast

echo
echo "=== Summary: $job_count jobs submitted ==="
for ((i=0; i<job_count; i++)); do
    echo "${jobs[$i]}"
done
