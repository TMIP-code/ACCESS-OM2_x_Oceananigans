#!/usr/bin/env bash

# Usage: source scripts/run_model.sh [PARENTMODEL]

# parent model (falls back to existing env or sensible default)
parent=${1:-${PARENTMODEL:-ACCESS-OM2-1}}
export PARENTMODEL="$parent"

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENTMODEL=$PARENTMODEL, REPO_ROOT=$repo_root"

# Run the script

echo "Running offline ACCESS-OM2 for PARENTMODEL=$PARENTMODEL"
julia --project $repo_root/src/offline_ACCESS-OM2.jl &> scratch_output/offline_ACCESS-OM2.$PBS_JOBID.out
echo "Done running offline ACCESS-OM2 for PARENTMODEL=$PARENTMODEL"
