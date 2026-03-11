#!/usr/bin/env bash
set -euo pipefail

# Submit the periodic 1-year simulation (GPU) followed by diagnostic plotting (CPU).
# Run from the login node (not as a PBS job).
#
# Usage:
#   bash scripts/ACCESS-OM2-1_submit_periodic_1year.sh                          # default (gpuvolta)
#   GPU_RESOURCES=gpuhopper bash scripts/ACCESS-OM2-1_submit_periodic_1year.sh  # on H200
#   LINEAR_SOLVER=ParU LUMP_AND_SPRAY=yes bash scripts/ACCESS-OM2-1_submit_periodic_1year.sh

PREFIX=ACCESS-OM2-1

# Solver configuration
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-no}

# GPU queue configuration (same case block as driver)
GPU_RESOURCES=${GPU_RESOURCES:-gpuvolta}
case "$GPU_RESOURCES" in
    gpuvolta)  GPU_MEM=96GB;  GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuvolta ;;
    gpuvolta2) GPU_MEM=192GB; GPU_NGPUS=2; GPU_NCPUS=24; GPU_QUEUE=gpuvolta ;;
    gpuhopper) GPU_MEM=256GB; GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuhopper ;;
    *)         echo "Unknown GPU_RESOURCES=$GPU_RESOURCES (must be: gpuvolta, gpuvolta2, gpuhopper)"; exit 1 ;;
esac

echo "=== ${PREFIX} periodic 1-year submission ==="
echo "GPU_RESOURCES=$GPU_RESOURCES (queue=$GPU_QUEUE, ngpus=$GPU_NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"
echo "LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY"

# 1. Submit GPU run job
RUN_JOB=$(qsub \
    -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
    -v LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
    scripts/${PREFIX}_run_periodic_1year.sh)
echo "[1] Run: $RUN_JOB"

# 2. Submit CPU plot job (waits for run to finish)
PLOT_JOB=$(qsub \
    -W depend=afterok:${RUN_JOB} \
    -v LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
    scripts/${PREFIX}_plot_periodic_1year_age_job.sh)
echo "[2] Plot: $PLOT_JOB (afterok $RUN_JOB)"

echo ""
echo "=== 2 jobs submitted ==="
echo "  run  ($RUN_JOB) → plot ($PLOT_JOB)"
