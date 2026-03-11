#!/bin/bash
# This script is for writing the locations of ACCESS-OM2 config files into
# ACCESS-OM2_configs.yaml. To run it, simply use
#     scripts/write_ACCESS-OM2_configs.sh

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "repo_root=$repo_root"
cd $repo_root

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3
python3 src/write_ACCESS-OM2_configs.py
