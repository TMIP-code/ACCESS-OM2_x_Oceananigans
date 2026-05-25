# Common env var defaults for all job scripts.
# Sourced (not executed), so variables are set in the caller's scope.

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}

# Experiment and time window (parent model forcing)
if [ -z "${EXPERIMENT:-}" ]; then
    case "$PARENT_MODEL" in
        ACCESS-OM2-1)   EXPERIMENT="1deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-025) EXPERIMENT="025deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-01)  EXPERIMENT="01deg_jra55v140_iaf_cycle4" ;;
        *)              echo "ERROR: No default EXPERIMENT for $PARENT_MODEL" >&2; exit 1 ;;
    esac
fi
TIME_WINDOW=${TIME_WINDOW:-1968-1977}

# MLD time window (decoupled from TIME_WINDOW). When unset/empty, MLD inputs
# come from TIME_WINDOW and outputs/logs land in the production tree. When
# explicitly set, MLD inputs come from MLD_TIME_WINDOW and outputs/logs are
# routed under test/TR{TIME_WINDOW}_MLD{MLD_TIME_WINDOW}/.
MLD_EXPLICIT="no"
if [ -n "${MLD_TIME_WINDOW:-}" ]; then
    MLD_EXPLICIT="yes"
else
    MLD_TIME_WINDOW="$TIME_WINDOW"
fi
if [ "$MLD_EXPLICIT" = "yes" ]; then
    LOG_TW_TAG="test/TR${TIME_WINDOW}_MLD${MLD_TIME_WINDOW}"
else
    LOG_TW_TAG="$TIME_WINDOW"
fi
# Julia's load_project_config uses haskey(ENV, "MLD_TIME_WINDOW") as the
# "explicit" signal. Only export MLD_TIME_WINDOW when explicitly set, so the
# default code path doesn't masquerade as explicit and reroute to test/.
export EXPERIMENT TIME_WINDOW MLD_EXPLICIT LOG_TW_TAG
[ "$MLD_EXPLICIT" = "yes" ] && export MLD_TIME_WINDOW

# Source model-specific config FIRST so it can establish per-model defaults
# (e.g. TIMESTEP_MULT) before the cross-model fallback defaults below run.
# The model config uses `${VAR:-default}` so user-set env vars still win.
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2
    exit 1
fi
source "$MODEL_CONF"

VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}     # cgridtransports | totaltransport
W_FORMULATION=${W_FORMULATION:-wprescribed}             # wdiagnosed | wprescribed
PRESCRIBED_W_SOURCE=${PRESCRIBED_W_SOURCE:-parent}      # diagnosed | parent (only when W_FORMULATION=wprescribed)
ADVECTION_SCHEME=${ADVECTION_SCHEME:-centered2}         # centered2 | weno3 | weno5
TIMESTEPPER=${TIMESTEPPER:-AB2}                         # AB2 | SRK2 | SRK3 | SRK4 | SRK5
TIMESTEP_MULT=${TIMESTEP_MULT:-1}                       # integer ≥ 1; Δt = TIMESTEP_MULT · Δt_base. Per-model defaults in model_configs/*.sh (4/3/2 for OM2-1/025/01).
PLOT_TS=${PLOT_TS:-no}                                  # yes | no — opt-in T/S surface animations in plot_standardrun_age.jl
TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY:-yes}       # yes | no — save Newton iterates xₙ as newton_iterate_NN.jld2 (use INITIAL_AGE=latest to restart)
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}                 # Pardiso | ParU | UMFPACK
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-no}                    # no | AxB (e.g. 5x5); legacy `yes` is rejected
MATRIX_PROCESSING=${MATRIX_PROCESSING:-symdrop}         # raw | symfill | dropzeros | symdrop
INITIAL_AGE=${INITIAL_AGE:-0}                           # 0 | TMage | latest | <path to .jld2>
TM_SOURCE=${TM_SOURCE:-const}                           # const | avg
TM_MODEL_CONFIG=${TM_MODEL_CONFIG:-}                    # override MODEL_CONFIG used to locate NK's preconditioner TM (empty = use MODEL_CONFIG)
GM_REDI=${GM_REDI:-no}                                  # no | diff | adv (legacy: yes = diff)
MONTHLY_KAPPAV=${MONTHLY_KAPPAV:-yes}                   # yes | no — derive 3D κV on the fly from 2D monthly MLD (tags MODEL_CONFIG with _mkappaV); default yes
IMPLICIT_KAPPAV=${IMPLICIT_KAPPAV:-yes}                 # yes | no — when "no", drop implicit vertical-diffusion closure (Probe B); tags MODEL_CONFIG with _noKV
TBLOCKING=${TBLOCKING:-no}                              # no | integer K ≥ 2 (temporal blocking: K sub-steps per MPI exchange)
GRID_HX=${GRID_HX:-7}                                   # grid halo in x (≥ K+1 when TBLOCKING=K)
GRID_HY=${GRID_HY:-7}                                   # grid halo in y (≥ K+1 when TBLOCKING=K)
GRID_HZ=${GRID_HZ:-2}                                   # grid halo in z (2 sufficient; larger is harmless)
LOAD_BALANCE=${LOAD_BALANCE:-no}                        # no | surface | cell | mix | minmax | yes(=surface; back-compat) — only valid when PARTITION_X=1
ACTIVE_CELLS_MAP=${ACTIVE_CELLS_MAP:-yes}               # yes | no — when "no", build IBG with active_cells_map=false and tag output files with _noACM
TRAF=${TRAF:-no}                                        # yes | no — Time-Reversed Adjoint Flow (adjoint age via reversed monthly FTS + sign-flipped u, v)
case "$TRAF" in yes|no) ;; *) echo "ERROR: TRAF must be yes or no (got: $TRAF)" >&2; exit 1 ;; esac
TRAF_TM_SOURCE=${TRAF_TM_SOURCE:-invVMtV}               # invVMtV | M_traf — matrix to use for TMsolve/NK when TRAF=yes (ignored when TRAF=no)
case "$TRAF_TM_SOURCE" in invVMtV|M_traf) ;; *) echo "ERROR: TRAF_TM_SOURCE must be invVMtV or M_traf (got: $TRAF_TM_SOURCE)" >&2; exit 1 ;; esac
OMEGA=${OMEGA:-all}                                     # all | z<depth> — restrict the age source to where z_center < -<depth> m (filename suffix only)
case "$OMEGA" in
    all) ;;
    z[0-9]*) ;;
    *) echo "ERROR: OMEGA must be 'all' or 'z<depth>' (e.g. z500, z1500) (got: $OMEGA)" >&2; exit 1 ;;
esac
# Validate LUMP_AND_SPRAY and derive Q_TAG (used only for logging; the
# NK subdir/file naming is composed Julia-side from parse_lump_and_spray).
case "$LUMP_AND_SPRAY" in
    no)            Q_TAG="" ;;
    yes)           echo "ERROR: LUMP_AND_SPRAY=yes is no longer supported; use 'no' or 'AxB' (e.g. 2x2)." >&2; exit 1 ;;
    [0-9]*x[0-9]*) Q_TAG="_Q${LUMP_AND_SPRAY}" ;;
    *) echo "ERROR: LUMP_AND_SPRAY must be 'no' or '<int>x<int>' (got: $LUMP_AND_SPRAY)" >&2; exit 1 ;;
esac
export Q_TAG

# Normalise + validate LOAD_BALANCE and derive MODEL_CONFIG tag suffix.
case "$LOAD_BALANCE" in
    no)             LB_TAG="" ;;
    surface|yes)    LB_TAG="_LBS" ; LOAD_BALANCE="surface" ;;
    cell)           LB_TAG="_LB" ;;
    mix)            LB_TAG="_LBmix" ;;
    minmax)         LB_TAG="_LBminmax" ;;
    *) echo "ERROR: LOAD_BALANCE must be no | surface | cell | mix | minmax (got: $LOAD_BALANCE)" >&2; exit 1 ;;
esac
MODEL_CONFIG="${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}_${TIMESTEPPER}"
if [ "$W_FORMULATION" = "wprescribed" ]; then
    if [ "$PRESCRIBED_W_SOURCE" = "diagnosed" ]; then
        MODEL_CONFIG="${VELOCITY_SOURCE}_wprediag_${ADVECTION_SCHEME}_${TIMESTEPPER}"
    else
        MODEL_CONFIG="${VELOCITY_SOURCE}_wparent_${ADVECTION_SCHEME}_${TIMESTEPPER}"
    fi
fi
case "$GM_REDI" in
    diff|yes)  MODEL_CONFIG="${MODEL_CONFIG}_GMREDI" ;;
    adv)       MODEL_CONFIG="${MODEL_CONFIG}_GMREDIadv" ;;
esac
if [ "$MONTHLY_KAPPAV" = "yes" ]; then
    MODEL_CONFIG="${MODEL_CONFIG}_mkappaV"
fi
case "$IMPLICIT_KAPPAV" in
    yes) ;;
    no)  MODEL_CONFIG="${MODEL_CONFIG}_noKV" ;;
    *) echo "ERROR: IMPLICIT_KAPPAV must be yes or no (got: $IMPLICIT_KAPPAV)" >&2; exit 1 ;;
esac
if [ "$TBLOCKING" != "no" ]; then
    MODEL_CONFIG="${MODEL_CONFIG}_TB${TBLOCKING}"
fi
MODEL_CONFIG="${MODEL_CONFIG}${LB_TAG}"
if [ "$TIMESTEP_MULT" != "1" ]; then
    MODEL_CONFIG="${MODEL_CONFIG}_DTx${TIMESTEP_MULT}"
fi
if [ "$TRAF" = "yes" ]; then
    MODEL_CONFIG="${MODEL_CONFIG}_traf"
fi
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION PRESCRIBED_W_SOURCE ADVECTION_SCHEME TIMESTEPPER TIMESTEP_MULT PLOT_TS TRACE_SOLVER_HISTORY
# export AA_M NLSAA_BETA SMAA_SIGMA_MIN SMAA_STABILIZE SMAA_CHECK_OBJ SMAA_ORDERS
export LINEAR_SOLVER LUMP_AND_SPRAY MATRIX_PROCESSING INITIAL_AGE TM_SOURCE TM_MODEL_CONFIG
export GM_REDI MONTHLY_KAPPAV IMPLICIT_KAPPAV TBLOCKING GRID_HX GRID_HY GRID_HZ LOAD_BALANCE ACTIVE_CELLS_MAP
export TRAF TRAF_TM_SOURCE OMEGA

echo "PARENT_MODEL=$PARENT_MODEL"
echo "EXPERIMENT=$EXPERIMENT"
echo "TIME_WINDOW=$TIME_WINDOW"
echo "MLD_TIME_WINDOW=$MLD_TIME_WINDOW (explicit=$MLD_EXPLICIT)"
echo "LOG_TW_TAG=$LOG_TW_TAG"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
echo "PRESCRIBED_W_SOURCE=$PRESCRIBED_W_SOURCE"
echo "ADVECTION_SCHEME=$ADVECTION_SCHEME"
echo "TIMESTEPPER=$TIMESTEPPER"
echo "TIMESTEP_MULT=$TIMESTEP_MULT"
echo "PLOT_TS=$PLOT_TS"
echo "TRACE_SOLVER_HISTORY=$TRACE_SOLVER_HISTORY"
# echo "AA_M=$AA_M"
# echo "NLSAA_BETA=$NLSAA_BETA"
# echo "SMAA_SIGMA_MIN=$SMAA_SIGMA_MIN"
# echo "SMAA_STABILIZE=$SMAA_STABILIZE"
# echo "SMAA_CHECK_OBJ=$SMAA_CHECK_OBJ"
# echo "SMAA_ORDERS=$SMAA_ORDERS"
echo "LINEAR_SOLVER=$LINEAR_SOLVER"
echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
echo "MATRIX_PROCESSING=$MATRIX_PROCESSING"
echo "INITIAL_AGE=$INITIAL_AGE"
echo "TM_SOURCE=$TM_SOURCE"
echo "GM_REDI=$GM_REDI"
echo "MONTHLY_KAPPAV=$MONTHLY_KAPPAV"
echo "IMPLICIT_KAPPAV=$IMPLICIT_KAPPAV"
echo "TBLOCKING=$TBLOCKING"
echo "GRID_HX=$GRID_HX, GRID_HY=$GRID_HY, GRID_HZ=$GRID_HZ"
echo "LOAD_BALANCE=$LOAD_BALANCE"
echo "ACTIVE_CELLS_MAP=$ACTIVE_CELLS_MAP"
echo "TRAF=$TRAF"
echo "TRAF_TM_SOURCE=$TRAF_TM_SOURCE"
echo "OMEGA=$OMEGA"
echo "MODEL_CONFIG=$MODEL_CONFIG"

# Bounds checking: set CHECK_BOUNDS=yes to run julia with --check-bounds=yes
CHECK_BOUNDS=${CHECK_BOUNDS:-no}
JULIA_BOUNDS_FLAG=""
if [ "$CHECK_BOUNDS" = "yes" ]; then
    JULIA_BOUNDS_FLAG="--check-bounds=yes"
    echo "CHECK_BOUNDS=yes (running julia with --check-bounds=yes)"
fi

# Module loading and environment — required for all jobs.
# MPItrampoline delegates to system OpenMPI via mpiwrapper;
# openmpi module must be loaded so libmpiwrapper.so can find libmpi.
echo "Loading modules (cuda, openmpi)"
module load cuda/12.9.0
module load openmpi/5.0.8
export JULIA_CUDA_USE_COMPAT=false
# Prepend JLL Libmount artifact to LD_LIBRARY_PATH so Glib_jll's libgio-2.0.so
# finds the JLL's libmount.so.1 (MOUNT_2_40) instead of the system one (RHEL 8
# only has up to MOUNT_2_37). Required for Makie/OceananigansMakieExt.
LIBMOUNT_DIR=$(dirname "$(find "${JULIA_DEPOT_PATH:-$HOME/.julia}/artifacts" -name "libmount.so.1" -print -quit 2>/dev/null)" 2>/dev/null)
if [ -n "$LIBMOUNT_DIR" ]; then
    export LD_LIBRARY_PATH="${LIBMOUNT_DIR}:/apps/openmpi/5.0.8/lib"
    echo "LIBMOUNT_DIR=$LIBMOUNT_DIR (prepended to LD_LIBRARY_PATH)"
else
    export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib
    echo "Warning: Libmount_jll artifact not found, Makie may fail to precompile"
fi
export JULIA_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export UCX_WARN_UNUSED_ENV_VARS=n
# MPItrampoline: point to mpiwrapper built against system OpenMPI
export MPITRAMPOLINE_LIB=$HOME/mpiwrapper/lib64/libmpiwrapper.so

# Model-specific config was sourced earlier (before variable defaults) so
# it can establish per-model defaults like TIMESTEP_MULT.

# --- Partition + queue configuration ---
CPU_QUEUE=${CPU_QUEUE:-express}
source "$(dirname "${BASH_SOURCE[0]}")/compute_resources.sh"

export MODEL_SHORT GPU_QUEUE CPU_QUEUE
export PARTITION PARTITION_X PARTITION_Y RANKS
export NGPUS GPU_NCPUS GPU_MEM GPU_MEM_SINGLE MEM_PER_GPU
export CPU_NCPUS CPU_MEM MEM_PER_CPU

echo "MODEL_SHORT=$MODEL_SHORT"
echo "GPU_QUEUE=$GPU_QUEUE"
echo "PARTITION=$PARTITION (${PARTITION_X}x${PARTITION_Y}, RANKS=$RANKS)"

# Git commit tracking (passed from driver via qsub -v)
if [ -n "${GIT_COMMIT:-}" ]; then
    export GIT_COMMIT
fi
echo "GIT_COMMIT=${GIT_COMMIT:-unknown}"
