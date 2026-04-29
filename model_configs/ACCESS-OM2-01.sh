# Model-specific config for ACCESS-OM2-01 (0.1°, eddy-resolving)
# Sourced by env_defaults.sh — provides MODEL_SHORT and walltimes.
#
# NOTE: OM2-01 is eddy-resolving and does NOT use GM parameterization.
# The intake catalog has no ty_trans_rho_gm (or analogues). Scripts that
# read GM variables must treat them as optional for OM2-01 — see
# README § Model notes and src/compute_MOC_rho_timeseries.py.
#
# Walltimes below are placeholders; most of the OM2-* pipeline has not
# been exercised at 0.1°. Adjust when you run.

MODEL_SHORT=OM2-01

# --- GPU queue (H200 preferred for 0.1° — much larger memory) ---
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# --- Preprocessing ---
WALLTIME_PREP=12:00:00
PREP_NCPUS=32
PREP_MEM=2048GB
PREP_QUEUE=megamem
WALLTIME_GRID=02:00:00
WALLTIME_VEL=06:00:00
VEL_NCPUS=16
VEL_MEM=512GB
VEL_QUEUE=hugemem
# clo: same envelope as vel for now (OM2-01 default 47GB express OOMed)
WALLTIME_CLO=06:00:00
CLO_NCPUS=16
CLO_MEM=512GB
CLO_QUEUE=hugemem
PARTITION_WALLTIME=02:00:00

# --- Standard runs ---
WALLTIME_RUN_1YEAR=${WALLTIME_RUN_1YEAR:-16:00:00}
WALLTIME_RUN_10YEARS=24:00:00
WALLTIME_RUN_100YEARS=48:00:00
WALLTIME_RUN_LONG=48:00:00

# --- Newton-Krylov solver ---
WALLTIME_NK=24:00:00

# --- Transport matrix ---
WALLTIME_TM_BUILD=04:00:00
WALLTIME_TM_SNAPSHOT=08:00:00
WALLTIME_TM_SOLVE=04:00:00

# --- Plotting ---
WALLTIME_PLOT=00:30:00
WALLTIME_PLOT_NK=02:00:00
