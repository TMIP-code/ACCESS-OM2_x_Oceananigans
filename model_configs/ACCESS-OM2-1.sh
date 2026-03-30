# Model-specific config for ACCESS-OM2-1
# Sourced by env_defaults.sh — provides MODEL_SHORT and walltimes.

MODEL_SHORT=OM2-1

# --- GPU queue (V100 sufficient for 1° resolution) ---
GPU_QUEUE=${GPU_QUEUE:-gpuvolta}

# --- Preprocessing ---
WALLTIME_PREP=01:00:00            # actual ~16min (full), ~2min (cached)
PREP_NCPUS=10
PREP_MEM=40GB                     # actual ~20-45GB
WALLTIME_GRID=00:30:00
WALLTIME_VEL=00:30:00

# --- Standard runs ---
WALLTIME_RUN_1YEAR=${WALLTIME_RUN_1YEAR:-00:30:00}       # actual ~8min
WALLTIME_RUN_10YEARS=02:00:00     # estimated ~80min
WALLTIME_RUN_100YEARS=15:00:00    # estimated ~13hr
WALLTIME_RUN_LONG=48:00:00

# --- Newton-Krylov solver ---
WALLTIME_NK=03:00:00              # actual ~2h45m

# --- Transport matrix: Jacobian build (create_matrix.jl) ---
WALLTIME_TM_BUILD=00:30:00       # actual ~10min

# --- Transport matrix: snapshot + averaging (create_snapshot_matrices.jl) ---
WALLTIME_TM_SNAPSHOT=01:00:00    # actual ~14min

# --- Transport matrix: age solver (solve_matrix_age.jl) ---
WALLTIME_TM_SOLVE=00:30:00       # actual ~5min

# --- Plotting ---
WALLTIME_PLOT=00:30:00
WALLTIME_PLOT_NK=01:00:00
