#!/usr/bin/env bash
set -euo pipefail

# velocity_sources=(bgridvelocities cgridtransports)
velocity_sources=(cgridtransports)
# w_formulations=(wdiagnosed wprescribed)
w_formulations=(wdiagnosed)
# advection_schemes=(centered2 weno3 weno5)
advection_schemes=(centered2)

count=0
expected=$(( ${#velocity_sources[@]} * ${#w_formulations[@]} * ${#advection_schemes[@]} ))

echo "Submitting ACCESS-OM2-1 matrix jobs:"
for velocity_source in "${velocity_sources[@]}"; do
    echo "  VELOCITY_SOURCE=${velocity_source}"
    for w_formulation in "${w_formulations[@]}"; do
        echo "  W_FORMULATION=${w_formulation}"
        for advection_scheme in "${advection_schemes[@]}"; do
            echo "  ADVECTION_SCHEME=${advection_scheme}"
            echo "  ENABLE_AGE_SOLVE=true"
            qsub -v PARENT_MODEL=ACCESS-OM2-1,VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",ADVECTION_SCHEME="${advection_scheme}",TIMESTEPPER="${TIMESTEPPER:-AB2}",ENABLE_AGE_SOLVE=true scripts/ACCESS-OM2-1_matrix_job.sh
            count=$((count + 1))
        done
    done
done

echo "Submitted ${count} jobs (expected ${expected})."
