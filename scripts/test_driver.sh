#!/bin/bash
# Unified test driver for ACCESS-OM2_x_Oceananigans.
#
# Usage (from login node):
#   GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill bash scripts/test_driver.sh
#   PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diag bash scripts/test_driver.sh
#   GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill-diag-mpi bash scripts/test_driver.sh
#
# Available test steps (dash-separated in JOB_CHAIN):
#   halofill  — fill_halo_regions! MWE at all staggered locations (distributed GPU)
#   halofillcpu — same MWE on 4 CPU ranks (no GPUs, express queue)
#   jld2      — JLD2Writer deadlock MWE on 2 CPU ranks (CliMA/Oceananigans.jl#5410)
#   diag      — 10-step diagnostic run saving every step (serial or distributed)
#   mpi       — MPI smoke test (rank/device info, 10-iteration simulation)

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

# Require clean git status before submitting jobs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: Commit before you submit a job. Working tree is not clean:" >&2
    git status --short >&2
    exit 1
fi
GIT_COMMIT=$(git rev-parse HEAD)

source scripts/env_defaults.sh

# --- Validate JOB_CHAIN ---
JOB_CHAIN=${JOB_CHAIN:-}
if [[ -z "$JOB_CHAIN" ]]; then
    echo "Usage: JOB_CHAIN=<step[-step...]> [GPU_RESOURCES=...] [PARENT_MODEL=...] bash scripts/test_driver.sh"
    echo ""
    echo "Available test steps: halofill halofillcpu jld2 diag mpi"
    echo ""
    echo "Examples:"
    echo "  GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill bash scripts/test_driver.sh"
    echo "  PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diag bash scripts/test_driver.sh"
    echo "  GPU_RESOURCES=gpuvolta-2x2 JOB_CHAIN=halofill-diag-mpi bash scripts/test_driver.sh"
    exit 1
fi

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

# --- GPU queue configuration (same as driver.sh) ---
GPU_RESOURCES=${GPU_RESOURCES:-gpuvolta}

GPU_BASE="${GPU_RESOURCES%%-*}"
GPU_SUFFIX="${GPU_RESOURCES#*-}"
if [[ "$GPU_BASE" != "$GPU_SUFFIX" ]] && [[ "$GPU_SUFFIX" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    GPU_PARTITION_X="${BASH_REMATCH[1]}"
    GPU_PARTITION_Y="${BASH_REMATCH[2]}"
    GPU_NGPUS=$(( GPU_PARTITION_X * GPU_PARTITION_Y ))
else
    GPU_PARTITION_X=1
    GPU_PARTITION_Y=1
    GPU_NGPUS=1
fi

case "$GPU_BASE" in
    gpuvolta)  GPU_MEM_PER_GPU=96;  GPU_QUEUE=gpuvolta ;;
    gpuhopper) GPU_MEM_PER_GPU=256; GPU_QUEUE=gpuhopper ;;
    *) echo "Unknown GPU_RESOURCES base: $GPU_BASE (must be gpuvolta or gpuhopper)"; exit 1 ;;
esac
GPU_MEM="$(( GPU_NGPUS * GPU_MEM_PER_GPU ))GB"
GPU_NCPUS=$(( GPU_NGPUS * 12 ))

export GPU_PARTITION_X GPU_PARTITION_Y

COMMON_VARS="PARENT_MODEL=${PARENT_MODEL},GIT_COMMIT=${GIT_COMMIT}"
WALLTIME=00:30:00

echo "=== ${PARENT_MODEL} test driver ==="
echo "MODEL_SHORT=$MODEL_SHORT"
echo "JOB_CHAIN=$JOB_CHAIN"
echo "GIT_COMMIT=$GIT_COMMIT"
echo "GPU_RESOURCES=$GPU_RESOURCES (queue=$GPU_QUEUE, partition=${GPU_PARTITION_X}x${GPU_PARTITION_Y}, ngpus=$GPU_NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"
echo ""

STEP=0
HALOFILL_JOB="" HALOFILLCPU_JOB="" JLD2_JOB="" DIAG_JOB="" MPI_JOB=""

# --- halofill: fill_halo_regions! MWE (GPU) ---
if has_step halofill; then
    STEP=$((STEP + 1))
    HALOFILL_JOB=$(qsub \
        -N "${MODEL_SHORT}_halofill" -l walltime=$WALLTIME \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},GPU_RESOURCES=${GPU_RESOURCES},GPU_PARTITION_X=${GPU_PARTITION_X},GPU_PARTITION_Y=${GPU_PARTITION_Y} \
        scripts/tests/run_halofill_test.sh)
    echo "[$STEP] halofill (GPU): $HALOFILL_JOB"
fi

# --- halofillcpu: fill_halo_regions! MWE (CPU, 4 ranks, no GPUs) ---
if has_step halofillcpu; then
    STEP=$((STEP + 1))
    HALOFILLCPU_JOB=$(qsub \
        -N "${MODEL_SHORT}_halofillcpu" -l walltime=$WALLTIME \
        -q express -l ngpus=0 -l ncpus=4 -l mem=16GB \
        -v ${COMMON_VARS} \
        scripts/tests/run_halofill_test.sh)
    echo "[$STEP] halofill (CPU): $HALOFILLCPU_JOB"
fi

# --- jld2: JLD2Writer deadlock MWE (2 CPU ranks, express queue) ---
if has_step jld2; then
    STEP=$((STEP + 1))
    JLD2_JOB=$(qsub \
        -N "${MODEL_SHORT}_jld2" -l walltime=$WALLTIME \
        -q express -l ngpus=0 -l ncpus=2 -l mem=16GB \
        -v ${COMMON_VARS} \
        scripts/tests/run_jld2writer_test.sh)
    echo "[$STEP] jld2 (CPU): $JLD2_JOB"
fi

# --- diag: 10-step diagnostic run ---
if has_step diag; then
    STEP=$((STEP + 1))
    DIAG_JOB=$(qsub \
        -N "${MODEL_SHORT}_diag" -l walltime=$WALLTIME \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},GPU_RESOURCES=${GPU_RESOURCES},GPU_PARTITION_X=${GPU_PARTITION_X},GPU_PARTITION_Y=${GPU_PARTITION_Y} \
        scripts/tests/run_diagnostic_steps.sh)
    echo "[$STEP] diag: $DIAG_JOB"
fi

# --- mpi: MPI smoke test ---
if has_step mpi; then
    STEP=$((STEP + 1))
    MPI_JOB=$(qsub \
        -N "${MODEL_SHORT}_mpi" -l walltime=$WALLTIME \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},GPU_RESOURCES=${GPU_RESOURCES},GPU_PARTITION_X=${GPU_PARTITION_X},GPU_PARTITION_Y=${GPU_PARTITION_Y} \
        scripts/tests/run_mpi_test.sh)
    echo "[$STEP] mpi: $MPI_JOB"
fi

echo ""
echo "=== $STEP jobs submitted for ${PARENT_MODEL} ==="
echo ""
for label_job in \
    "halofill:$HALOFILL_JOB" \
    "halofillcpu:$HALOFILLCPU_JOB" \
    "jld2:$JLD2_JOB" \
    "diag:$DIAG_JOB" \
    "mpi:$MPI_JOB" \
; do
    label="${label_job%%:*}"
    job="${label_job#*:}"
    if [ -n "$job" ]; then
        printf "  %-24s %s\n" "$label" "$job"
    fi
done
