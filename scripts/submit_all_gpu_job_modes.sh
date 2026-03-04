#!/usr/bin/env bash
set -euo pipefail

velocity_sources=(bgridvelocities cgridtransports)
w_formulations=(wdiagnosed wprescribed)

count=0
expected=$(( ${#velocity_sources[@]} * ${#w_formulations[@]} ))

for velocity_source in "${velocity_sources[@]}"; do
  for w_formulation in "${w_formulations[@]}"; do
    echo "Submitting ACCESS-OM2-1 GPU jobs:"
    echo "  VELOCITY_SOURCE=${velocity_source}"
    echo "  W_FORMULATION=${w_formulation}"
    echo "  ENABLE_PLOTTING=true"
    qsub -v PARENT_MODEL=ACCESS-OM2-1,VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",TIMESTEPPER="${TIMESTEPPER:-AB2}",ENABLE_PLOTTING=true scripts/ACCESS-OM2-1_GPU_job.sh
    count=$((count + 1))
  done
done

echo "Submitted ${count} jobs (expected ${expected})."
