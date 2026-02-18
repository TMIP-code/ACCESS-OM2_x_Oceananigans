#!/bin/bash

#PBS -N run_OM2-1_CPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o scratch_output/PBS/
#PBS -e scratch_output/PBS/
#PBS -l wd

# parent model (falls back to existing env or sensible default)
PARENTMODEL=ACCESS-OM2-1

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENTMODEL=$PARENTMODEL, REPO_ROOT=$repo_root"

# echo "Create grid on CPU with PARENTMODEL=$PARENTMODEL"
# source $repo_root/scripts/create_grid.sh $PARENTMODEL
# echo "Done creating grid on CPU with PARENTMODEL=$PARENTMODEL"

echo "Create velocities on CPU with PARENTMODEL=$PARENTMODEL"
source $repo_root/scripts/create_velocities.sh $PARENTMODEL
echo "Done creating velocities on CPU with PARENTMODEL=$PARENTMODEL"

# echo "Create closures on CPU with PARENTMODEL=$PARENTMODEL"
# source $repo_root/scripts/create_closures.sh $PARENTMODEL
# echo "Done creating closures on CPU with PARENTMODEL=$PARENTMODEL"
