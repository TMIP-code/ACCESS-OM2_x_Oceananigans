#!/usr/bin/env bash

# Usage: source scripts/run_model.sh [PARENTMODEL]

# parent model (falls back to existing env or sensible default)
parent=${1:-${PARENTMODEL:-ACCESS-OM2-1}}
export PARENTMODEL="$parent"

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
echo "Sourced: PARENTMODEL=$PARENTMODEL, REPO_ROOT=$repo_root"

# Run the scripts

echo "Creating grid on CPU for PARENTMODEL=$PARENTMODEL"
julia --project $repo_root/src/create_grid.jl

echo "Creating velocities on CPU for PARENTMODEL=$PARENTMODEL"
julia --project $repo_root/src/create_velocities.jl

# echo "Creating closures on CPU for PARENTMODEL=$PARENTMODEL"
# julia --project $repo_root/src/create_closures.jl

echo "Running offline ACCESS-OM2 on CPU for PARENTMODEL=$PARENTMODEL"
julia --project $repo_root/src/offline_ACCESS-OM2.jl
