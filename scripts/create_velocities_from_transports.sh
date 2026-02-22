#!/usr/bin/env bash

parent=${1:-${PARENTMODEL:-ACCESS-OM2-1}}
export PARENTMODEL="$parent"

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENTMODEL=$PARENTMODEL, REPO_ROOT=$repo_root"

echo "Creating transport-derived velocities for PARENTMODEL=$PARENTMODEL"
julia --project $repo_root/src/create_velocities_from_transports.jl
echo "Done creating transport-derived velocities for PARENTMODEL=$PARENTMODEL"
