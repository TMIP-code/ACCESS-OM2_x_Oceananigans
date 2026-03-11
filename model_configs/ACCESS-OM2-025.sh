# Model-specific config for ACCESS-OM2-025
# Sourced by env_defaults.sh — provides MODEL_SHORT and walltimes.

MODEL_SHORT=OM2-025

# --- GPU queue (H200 needed for 0.25° resolution) ---
GPU_RESOURCES=${GPU_RESOURCES:-gpuhopper}

# --- Preprocessing ---
WALLTIME_GRID=00:30:00
WALLTIME_VEL=00:30:00

# --- Standard runs ---
WALLTIME_RUN_1YEAR=00:30:00
WALLTIME_RUN_10YEARS=48:00:00
WALLTIME_RUN_100YEARS=48:00:00
WALLTIME_RUN_LONG=48:00:00

# --- Newton-Krylov solver ---
WALLTIME_NK=48:00:00

# --- Transport matrix: Jacobian build (create_matrix.jl) ---
WALLTIME_TM_BUILD=00:30:00

# --- Transport matrix: snapshot + averaging (create_snapshot_matrices.jl) ---
WALLTIME_TM_SNAPSHOT=01:00:00

# --- Transport matrix: age solver (solve_matrix_age.jl) ---
WALLTIME_TM_SOLVE=00:30:00

# --- Plotting ---
WALLTIME_PLOT=00:30:00
WALLTIME_PLOT_NK=01:00:00
