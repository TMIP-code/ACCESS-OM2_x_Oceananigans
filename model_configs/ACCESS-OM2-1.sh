# Model-specific config for ACCESS-OM2-1
# Sourced by env_defaults.sh — provides MODEL_SHORT and walltimes.

MODEL_SHORT=OM2-1

# --- GPU queue (V100 sufficient for 1° resolution) ---
GPU_QUEUE=${GPU_QUEUE:-gpuvolta}

# --- Partition (1x1 — serial run on a single V100 fits OM2-1 easily) ---
PARTITION=${PARTITION:-1x1}

# --- Velocity source (mass transports preferred over interpolated C-grid) ---
VELOCITY_SOURCE=${VELOCITY_SOURCE:-totaltransport}

# --- Tracer timestep multiplier (Δt = M·Δt_base) ---
TIMESTEP_MULT=${TIMESTEP_MULT:-4}

# --- NK preconditioner coarsening (small matrix → 2x2 is plenty) ---
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-2x2}

# --- Diffusivities (m²/s) — baseline reference values for 1° resolution ---
# κH (horizontal) scales with √(cell area); κV with √(level-ratio × dx-ratio).
# OM2-1 is the reference: κH=300, κVML=0.1, κVBG=3.0e-5.
KAPPA_H=${KAPPA_H:-300}
KAPPA_V_ML=${KAPPA_V_ML:-1e-1}
KAPPA_V_BG=${KAPPA_V_BG:-3e-5}

# --- Preprocessing ---
WALLTIME_PREP=02:00:00            # actual ~16min (20yr), longer for 30yr windows
PREP_NCPUS=24                     # 4GB/CPU minimum charge; periodicaverage.py uses n_workers=ncpus
PREP_MEM=96GB                     # 24 CPUs × 4GB
WALLTIME_GRID=00:30:00
WALLTIME_VEL=00:30:00

# Partition build memory: each rank loads the full serial FTS into memory,
# so memory scales linearly with RANKS. Observed peak ≈ 4-6 GB/rank
# (1x8 used 30/32 GB for halos=13). 12 GB/rank gives ~2× headroom.
PARTITION_MEM_PER_RANK=12

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
