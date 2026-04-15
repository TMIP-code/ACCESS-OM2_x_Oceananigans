#!/bin/bash
#
# PBS wrapper for src/compute_MOC_rho_timeseries.py — computes the full
# monthly timeseries of the global density-space MOC (ty_trans_rho +
# ty_trans_rho_gm, zonal sum + cumsum over potrho → Sv) and writes a single
# NetCDF per model. Reads raw MOM output via the intake catalog, so runs
# independently of TIME_WINDOW — hence not wired into driver.sh.
#
# Usage:
#   qsub -v "PARENT_MODEL=ACCESS-OM2-1" scripts/prepreprocessing/compute_MOC_rho_timeseries.sh
#
# Writes:  /scratch/y99/TMIP/data/{PARENT_MODEL}/{EXPERIMENT}/rhospace/psi_tot_global.nc
# Logs to: logs/python/{PARENT_MODEL}/{EXPERIMENT}/compute_MOC_rho_timeseries_<jobid>.log

#PBS -P y99
#PBS -q express
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

# Defaults (overridable via qsub -v)
PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
if [ -z "${EXPERIMENT:-}" ]; then
    case "$PARENT_MODEL" in
        ACCESS-OM2-1)   EXPERIMENT="1deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-025) EXPERIMENT="025deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-01)  EXPERIMENT="01deg_jra55v140_iaf_cycle4" ;;
        *)              echo "ERROR: No default EXPERIMENT for $PARENT_MODEL" >&2; exit 1 ;;
    esac
fi

echo "PARENT_MODEL=$PARENT_MODEL"
echo "EXPERIMENT=$EXPERIMENT"

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/python/$PARENT_MODEL/$EXPERIMENT
mkdir -p "$log_dir"

echo "Loading conda/analysis3 module"
module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

echo "Running compute_MOC_rho_timeseries.py"
python3 src/compute_MOC_rho_timeseries.py "$PARENT_MODEL" "$EXPERIMENT" \
    &> "$log_dir/compute_MOC_rho_timeseries_${job_id}.log"
echo "Done; logs in $log_dir/compute_MOC_rho_timeseries_${job_id}.log"
