# Common env var defaults for all job scripts.
# Sourced (not executed), so variables are set in the caller's scope.

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
ADVECTION_SCHEME=${ADVECTION_SCHEME:-centered2}
TIMESTEPPER=${TIMESTEPPER:-AB2}
TRACE_SOLVER_HISTORY=${TRACE_SOLVER_HISTORY:-no}
# AA solver variables — no longer needed, will be removed in the near future.
# AA_M=${AA_M:-40}
# NLSAA_BETA=${NLSAA_BETA:-1.0}
# SMAA_SIGMA_MIN=${SMAA_SIGMA_MIN:-0.0}
# SMAA_STABILIZE=${SMAA_STABILIZE:-no}
# SMAA_CHECK_OBJ=${SMAA_CHECK_OBJ:-no}
# SMAA_ORDERS=${SMAA_ORDERS:-332}
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-no}
MATRIX_PROCESSING=${MATRIX_PROCESSING:-raw}
INITIAL_AGE=${INITIAL_AGE:-TMage}
TM_SOURCE=${TM_SOURCE:-avg24}
MODEL_CONFIG="${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}_${TIMESTEPPER}"
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER TRACE_SOLVER_HISTORY
# export AA_M NLSAA_BETA SMAA_SIGMA_MIN SMAA_STABILIZE SMAA_CHECK_OBJ SMAA_ORDERS
export LINEAR_SOLVER LUMP_AND_SPRAY MATRIX_PROCESSING INITIAL_AGE TM_SOURCE

echo "PARENT_MODEL=$PARENT_MODEL"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
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

# Source model-specific config (MODEL_SHORT, GPU_RESOURCES, walltimes)
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2
    exit 1
fi
source "$MODEL_CONF"
export MODEL_SHORT GPU_RESOURCES
echo "MODEL_SHORT=$MODEL_SHORT"
echo "GPU_RESOURCES=$GPU_RESOURCES"
