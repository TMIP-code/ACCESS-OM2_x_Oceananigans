#!/usr/bin/env bash
set -euo pipefail

# velocity_sources=(cgridtransports totaltransports)
velocity_sources=(cgridtransports)
# w_formulations=(wdiagnosed wprescribed)
w_formulations=(wdiagnosed)
# advection_schemes=(centered2 weno3 weno5)
advection_schemes=(centered2)

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}

# Source model config for MODEL_SHORT and walltimes
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2; exit 1
fi
source "$MODEL_CONF"

count=0
expected=$(( ${#velocity_sources[@]} * ${#w_formulations[@]} * ${#advection_schemes[@]} ))

echo "Submitting ${PARENT_MODEL} matrix jobs:"
for velocity_source in "${velocity_sources[@]}"; do
    echo "  VELOCITY_SOURCE=${velocity_source}"
    for w_formulation in "${w_formulations[@]}"; do
        echo "  W_FORMULATION=${w_formulation}"
        for advection_scheme in "${advection_schemes[@]}"; do
            echo "  ADVECTION_SCHEME=${advection_scheme}"
            qsub -N "${MODEL_SHORT}_TMbuild" -l walltime=${WALLTIME_TM_BUILD} \
                -v PARENT_MODEL=${PARENT_MODEL},VELOCITY_SOURCE="${velocity_source}",W_FORMULATION="${w_formulation}",ADVECTION_SCHEME="${advection_scheme}",TIMESTEPPER="${TIMESTEPPER:-AB2}" \
                scripts/preprocessing/build_TMconst.sh
            count=$((count + 1))
        done
    done
done

echo "Submitted ${count} jobs (expected ${expected})."
