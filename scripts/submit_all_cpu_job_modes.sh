#!/usr/bin/env bash
set -euo pipefail

velocity_sources=(bgridvelocities cgridtransports)
w_formulations=(wdiagnosed wprescribed)
free_surfaces=(etaprescribed etazero etanothing)

count=0
expected=$(( ${#velocity_sources[@]} * ${#w_formulations[@]} * ${#free_surfaces[@]} ))

for velocity_source in "${velocity_sources[@]}"; do
  for w_formulation in "${w_formulations[@]}"; do
    for free_surface in "${free_surfaces[@]}"; do
      echo "Submitting ACCESS-OM2-1 CPU jobs:"
      echo "  VELOCITY_SOURCE=${velocity_source}"
      echo "  W_FORMULATION=${w_formulation}"
      echo "  FREE_SURFACE=${free_surface}"
      echo "  ENABLE_PLOTTING=true"
      qsub -v PARENT_MODEL=ACCESS-OM2-1,VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",FREE_SURFACE="${free_surface}",ENABLE_PLOTTING=true scripts/ACCESS-OM2-1_CPU_job.sh
      count=$((count + 1))
    done
  done
done

echo "Submitted ${count} jobs (expected ${expected})."
