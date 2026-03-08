#!/usr/bin/env bash
set -euo pipefail

# Submit 1-year GPU simulations for all config combinations.
#
# GPU queue selection (default: gpuhopper):
#   GPU_QUEUE=gpuvolta bash scripts/submit_all_gpu_job_modes.sh

GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# Auto-set GPU memory based on queue
case "$GPU_QUEUE" in
    gpuvolta)  GPU_MEM=96GB ;;
    gpuhopper) GPU_MEM=256GB ;;
    *)         echo "Unknown GPU_QUEUE=$GPU_QUEUE"; exit 1 ;;
esac

echo "GPU_QUEUE=$GPU_QUEUE (mem=$GPU_MEM)"

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
        echo "Submitting ACCESS-OM2-1 1-year run:"
        echo "  VELOCITY_SOURCE=${velocity_source}"
        echo "  W_FORMULATION=${w_formulation}"
        echo "  TIMESTEPPER=${timestepper}"
        extra_vars=""
        [ -n "${TRACE_SOLVER_HISTORY:-}" ] && extra_vars="${extra_vars},TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY}"
        qsub -q $GPU_QUEUE -l mem=$GPU_MEM \
            -v PARENT_MODEL=ACCESS-OM2-1,VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",TIMESTEPPER="${timestepper}"${extra_vars} \
            scripts/ACCESS-OM2-1_run_1year.sh
        count=$((count + 1))
        done
    done
done

echo "Submitted ${count} jobs (expected ${expected})."
