# Model-specific config for ACCESS-OM2-025
# Sourced by env_defaults.sh — provides MODEL_SHORT and walltimes.

MODEL_SHORT=OM2-025

# --- GPU queue (H200 needed for 0.25° resolution) ---
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# --- Partition (1x2 — two H200 ranks for OM2-025; LBS wins) ---
PARTITION=${PARTITION:-1x2}

# --- Tracer timestep multiplier (Δt = M·Δt_base) ---
TIMESTEP_MULT=${TIMESTEP_MULT:-3}

# --- NK preconditioner coarsening (medium matrix → 2x2) ---
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-2x2}

# --- CPU queue (hugemem for 0.25° — partition needs ~48 GB per rank) ---
CPU_QUEUE=${CPU_QUEUE:-hugemem}

# --- Preprocessing ---
WALLTIME_PREP=03:00:00            # actual ~25min (full)
PREP_NCPUS=48
PREP_MEM=192GB                    # actual ~160GB
WALLTIME_GRID=00:30:00
WALLTIME_VEL=01:00:00
# vel: peak 47GB observed across 24 historical jobs (cap hit on 8/24 with default 47GB);
# 78GB peak when given 96GB. 96GB on express gives ~22% headroom.
VEL_MEM=96GB

# Partition build: with halos=13, observed peaks 195GB (1x4) and 268GB (1x8)
# both hit caps with old defaults. 64GB/rank covers all sizes; hugemem queue
# min 192GB still kicks in for 1x2.
PARTITION_MEM_PER_RANK=64

# --- Standard runs ---
WALLTIME_RUN_1YEAR=${WALLTIME_RUN_1YEAR:-00:30:00}
WALLTIME_RUN_10YEARS=48:00:00
WALLTIME_RUN_100YEARS=48:00:00
WALLTIME_RUN_LONG=48:00:00

# --- Newton-Krylov solver ---
WALLTIME_NK=48:00:00

# --- Transport matrix: Jacobian build (create_matrix.jl) ---
WALLTIME_TM_BUILD=02:00:00       # actual ~1h13m, 178GB mem

# --- Transport matrix: snapshot + averaging (create_snapshot_matrices.jl) ---
WALLTIME_TM_SNAPSHOT=01:00:00

# --- Transport matrix: age solver (solve_matrix_age.jl) ---
WALLTIME_TM_SOLVE=00:30:00

# --- Plotting ---
WALLTIME_PLOT=${WALLTIME_PLOT:-00:30:00}
# plotNK on OM2-025 ran out of the 01:00:00 wall (used 01:00:15 / 01:00:51
# under TRAF; killed with -29). OM2-1 plotNK takes ~6–19 min, and OM2-025 is
# ~4× the cell count for the same per-cell plotting work, so 02:00:00 gives
# comfortable margin. Resources also doubled (24 CPU / 94 GB), keeping the
# per-CPU memory constant at ~3.9 GB so the plot scripts that load multiple
# fields into memory have headroom at this resolution.
WALLTIME_PLOT_NK=${WALLTIME_PLOT_NK:-02:00:00}
PLOT_NK_NCPUS=${PLOT_NK_NCPUS:-24}
PLOT_NK_MEM=${PLOT_NK_MEM:-96GB}      # 4 GB/CPU on express (NCI charges max(ncpus, ⌈mem/4⌉))
PLOT_TM_NCPUS=24
PLOT_TM_MEM=96GB
