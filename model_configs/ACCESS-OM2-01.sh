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

# --- GPU queue (H200 needed for 0.1° — 256 GB per H200) ---
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# --- Partition (1x4 — four H200 ranks needed for OM2-01 memory; LBS wins) ---
PARTITION=${PARTITION:-1x4}

# --- Velocity source (cgridtransports — OM2-01 has no separate mass-transport intake) ---
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}

# --- Tracer timestep multiplier (Δt = M·Δt_base) ---
TIMESTEP_MULT=${TIMESTEP_MULT:-2}

# --- NK preconditioner coarsening (large matrix → 5x5 to fit Pardiso budget) ---
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-5x5}

# --- CPU queue (megamem for 0.1° — partition + TMbuild need >1 TB) ---
CPU_QUEUE=${CPU_QUEUE:-megamem}

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
# Partition build: peak ~370GB/rank observed for 1x4 (1.47TB total) and ~256GB/rank
# for 1x8 (2.05TB). 350GB/rank covers all sizes with headroom; megamem queue
# (3TB max) needed for 1x8 (2.8TB).
PARTITION_QUEUE=megamem
PARTITION_MEM_PER_RANK=350

# --- Standard runs ---
WALLTIME_RUN_1YEAR=${WALLTIME_RUN_1YEAR:-04:00:00}
WALLTIME_RUN_10YEARS=24:00:00
WALLTIME_RUN_100YEARS=48:00:00
WALLTIME_RUN_LONG=48:00:00

# --- Newton-Krylov solver ---
WALLTIME_NK=24:00:00

# --- Transport matrix ---
# OM2-01 TMbuild dominates by sparsity detection (3h 12m at 351M wet cells;
# job 169134142). Extrapolated total ≈5.5 h; budget 10 h for margin.
WALLTIME_TM_BUILD=10:00:00
WALLTIME_TM_SNAPSHOT=08:00:00
WALLTIME_TM_SOLVE=04:00:00
# TMbuild at OM2-01 ran OOM at the default 192 GB (job 169132266 used 186 GB
# before SIGKILL). Push to hugemem max usable: 48 CPU / 1470 GB (PBS rejects
# 1536 GB as exceeding hugemem per-node mem). Refine downward only after we
# see a successful run; safety > efficiency here.
TMBUILD_QUEUE=hugemem
TMBUILD_NCPUS=48
TMBUILD_MEM=1470GB

# --- Plotting ---
WALLTIME_PLOT=00:30:00
WALLTIME_PLOT_NK=02:00:00
