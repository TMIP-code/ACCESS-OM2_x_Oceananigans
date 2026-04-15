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
TIME_WINDOW=${TIME_WINDOW:-1960-1979}
export EXPERIMENT TIME_WINDOW

VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}    # bgridvelocities | cgridtransports | totaltransport
W_FORMULATION=${W_FORMULATION:-wdiagnosed}              # wdiagnosed | wprescribed
PRESCRIBED_W_SOURCE=${PRESCRIBED_W_SOURCE:-parent}      # diagnosed | parent (only when W_FORMULATION=wprescribed)
ADVECTION_SCHEME=${ADVECTION_SCHEME:-centered2}         # centered2 | weno3 | weno5
TIMESTEPPER=${TIMESTEPPER:-AB2}                         # AB2 | SRK2 | SRK3 | SRK4 | SRK5
TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY:-no}        # yes | no
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}                 # Pardiso | ParU | UMFPACK
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-no}                    # yes | no
MATRIX_PROCESSING=${MATRIX_PROCESSING:-raw}             # raw | symfill | dropzeros | symdrop
INITIAL_AGE=${INITIAL_AGE:-TMage}                       # TMage | 0 | <path to .jld2>
TM_SOURCE=${TM_SOURCE:-avg}                             # const | avg
GM_REDI=${GM_REDI:-no}                                  # no | diff | adv (legacy: yes = diff)
MONTHLY_KAPPAV=${MONTHLY_KAPPAV:-no}                    # yes | no
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
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION PRESCRIBED_W_SOURCE ADVECTION_SCHEME TIMESTEPPER TRACE_SOLVER_HISTORY
# export AA_M NLSAA_BETA SMAA_SIGMA_MIN SMAA_STABILIZE SMAA_CHECK_OBJ SMAA_ORDERS
export LINEAR_SOLVER LUMP_AND_SPRAY MATRIX_PROCESSING INITIAL_AGE TM_SOURCE
export GM_REDI MONTHLY_KAPPAV

echo "PARENT_MODEL=$PARENT_MODEL"
echo "EXPERIMENT=$EXPERIMENT"
echo "TIME_WINDOW=$TIME_WINDOW"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
echo "PRESCRIBED_W_SOURCE=$PRESCRIBED_W_SOURCE"
echo "ADVECTION_SCHEME=$ADVECTION_SCHEME"
echo "TIMESTEPPER=$TIMESTEPPER"
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

# Source model-specific config (MODEL_SHORT, GPU_QUEUE, walltimes)
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2
    exit 1
fi
source "$MODEL_CONF"

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
