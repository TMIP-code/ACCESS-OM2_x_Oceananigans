#!/usr/bin/env bash
set -euo pipefail

# Submit 1-year GPU simulations for all config combinations.
#
# Optional env vars:
#   PARENT_MODEL  (default: ACCESS-OM2-1)
#   GPU_RESOURCES (default: gpuhopper; options: gpuvolta, gpuvolta2, gpuhopper)

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
GPU_RESOURCES=${GPU_RESOURCES:-gpuhopper}

# Source model config for MODEL_SHORT and walltimes
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2; exit 1
fi
source "$MODEL_CONF"

# Auto-set GPU resources based on selection
case "$GPU_RESOURCES" in
    gpuvolta)  GPU_MEM=96GB;  GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuvolta ;;
    gpuvolta2) GPU_MEM=192GB; GPU_NGPUS=2; GPU_NCPUS=24; GPU_QUEUE=gpuvolta ;;
    gpuhopper) GPU_MEM=256GB; GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuhopper ;;
    *)         echo "Unknown GPU_RESOURCES=$GPU_RESOURCES (must be: gpuvolta, gpuvolta2, gpuhopper)"; exit 1 ;;
esac

echo "PARENT_MODEL=$PARENT_MODEL (MODEL_SHORT=$MODEL_SHORT)"
echo "GPU_RESOURCES=$GPU_RESOURCES (queue=$GPU_QUEUE, ngpus=$GPU_NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"

# velocity_sources=(bgridvelocities cgridtransports)
velocity_sources=(cgridtransports)
# w_formulations=(wdiagnosed wprescribed)
w_formulations=(wdiagnosed)
timesteppers=(AB2 SRK2 SRK3 SRK4 SRK5)

count=0
expected=$(( ${#velocity_sources[@]} * ${#w_formulations[@]} * ${#timesteppers[@]} ))

for velocity_source in "${velocity_sources[@]}"; do
    for w_formulation in "${w_formulations[@]}"; do
        for timestepper in "${timesteppers[@]}"; do
        echo "Submitting ${PARENT_MODEL} 1-year run:"
        echo "  VELOCITY_SOURCE=${velocity_source}"
        echo "  W_FORMULATION=${w_formulation}"
        echo "  TIMESTEPPER=${timestepper}"
        extra_vars=""
        [ -n "${TRACE_SOLVER_HISTORY:-}" ] && extra_vars="${extra_vars},TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY}"
        qsub -N "${MODEL_SHORT}_run1yr" -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
            -l walltime=${WALLTIME_RUN_1YEAR} \
            -v PARENT_MODEL=${PARENT_MODEL},VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",TIMESTEPPER="${timestepper}"${extra_vars} \
            scripts/standard_runs/run_1year.sh
        count=$((count + 1))
        done
    done
done

echo "Submitted ${count} jobs (expected ${expected})."
