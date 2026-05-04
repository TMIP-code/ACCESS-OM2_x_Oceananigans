#!/bin/bash

#PBS -P y99
#PBS -l ncpus=48
#PBS -l mem=192GB
#PBS -l jobfs=4GB
#PBS -l storage=gdata/xv83+gdata/oi10+gdata/dk92+gdata/hh5+gdata/rr3+gdata/al33+gdata/fs38+gdata/xp65+gdata/p73+gdata/cj50+gdata/ik11
#PBS -l wd
#PBS -o logs/PBS/
#PBS -j oe

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

# Defaults (can be overridden via qsub -v)
PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
if [ -z "${EXPERIMENT:-}" ]; then
    case "$PARENT_MODEL" in
        ACCESS-OM2-1)   EXPERIMENT="1deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-025) EXPERIMENT="025deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-01)  EXPERIMENT="01deg_jra55v140_iaf_cycle4" ;;
        *)              echo "ERROR: No default EXPERIMENT for $PARENT_MODEL" >&2; exit 1 ;;
    esac
fi
TIME_WINDOW=${TIME_WINDOW:-1968-1977}

# Mirror env_defaults.sh logic for LOG_TW_TAG (this script doesn't source it).
if [ -n "${MLD_TIME_WINDOW:-}" ]; then
    LOG_TW_TAG="test/TR${TIME_WINDOW}_MLD${MLD_TIME_WINDOW}"
else
    MLD_TIME_WINDOW="$TIME_WINDOW"
    LOG_TW_TAG="$TIME_WINDOW"
fi
export PARENT_MODEL EXPERIMENT TIME_WINDOW MLD_TIME_WINDOW LOG_TW_TAG

echo "PARENT_MODEL=$PARENT_MODEL"
echo "EXPERIMENT=$EXPERIMENT"
echo "TIME_WINDOW=$TIME_WINDOW"
echo "MLD_TIME_WINDOW=$MLD_TIME_WINDOW"
echo "LOG_TW_TAG=$LOG_TW_TAG"

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/python/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG
mkdir -p "$log_dir"

echo "Loading conda/analysis3 module"
module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

# Disable HDF5 file locking — /home (NFS) breaks dask+netCDF4 writes with
# "Unable to lock file" at any nontrivial output size. Safe for single-writer
# workloads (we have exactly one process per output file).
export HDF5_USE_FILE_LOCKING=FALSE

echo "Running periodicaverage.py"
python3 src/periodicaverage.py &> "$log_dir/periodicaverage_${job_id}.log"
echo "Done preprocessing for $PARENT_MODEL/$EXPERIMENT/$TIME_WINDOW"
echo "logged output in $log_dir/periodicaverage_${job_id}.log"
