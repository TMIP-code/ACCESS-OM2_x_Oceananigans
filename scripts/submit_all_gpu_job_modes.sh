#!/usr/bin/env bash
set -euo pipefail

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
        echo "Submitting ACCESS-OM2-1 GPU jobs:"
        echo "  VELOCITY_SOURCE=${velocity_source}"
        echo "  W_FORMULATION=${w_formulation}"
        echo "  TIMESTEPPER=${timestepper}"
        echo "  ENABLE_PLOTTING=true"
        extra_vars=""
        [ -n "${TRACE_SOLVER_HISTORY:-}" ] && extra_vars="${extra_vars},TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY}"
        [ -n "${NONLINEAR_SOLVER:-}" ] && extra_vars="${extra_vars},NONLINEAR_SOLVER=${NONLINEAR_SOLVER}"
        qsub -v PARENT_MODEL=ACCESS-OM2-1,VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",TIMESTEPPER="${timestepper}",ENABLE_PLOTTING=true${extra_vars} scripts/ACCESS-OM2-1_GPU_job.sh
        count=$((count + 1))
        done
    done
done

echo "Submitted ${count} jobs (expected ${expected})."
