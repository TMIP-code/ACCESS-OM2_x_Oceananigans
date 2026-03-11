#!/usr/bin/env bash
set -euo pipefail

# Submit all LINEAR_SOLVER × LUMP_AND_SPRAY × MATRIX_PROCESSING combinations
# for solve_matrix_age.
#
# MATRIX_PROCESSING variants are only submitted for Pardiso (the only solver
# that uses matrix_type); ParU and UMFPACK use raw only.
#
# Total: Pardiso (1 × 2 × 4 = 8) + ParU (1 × 2 × 1 = 2) + UMFPACK (1 × 2 × 1 = 2) + CUDSS (1 × 2 × 1 = 2) = 14 jobs
#
# 1s delay between submissions to avoid concurrent precompilation OOM.
#
# Optional env vars forwarded to all jobs:
#   VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, CHECK_BOUNDS
#
# GPU queue selection (default: gpuhopper):
#   GPU_QUEUE=gpuvolta bash scripts/submit_all_solve_matrix_age.sh

DELAY=1  # seconds between submissions
PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# Source model config for MODEL_SHORT
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2; exit 1
fi
source "$MODEL_CONF"

# Auto-set GPU memory based on queue
case "$GPU_QUEUE" in
    gpuvolta)  GPU_MEM=96GB ;;
    gpuhopper) GPU_MEM=256GB ;;
    *)         echo "Unknown GPU_QUEUE=$GPU_QUEUE"; exit 1 ;;
esac

echo "PARENT_MODEL=$PARENT_MODEL (MODEL_SHORT=$MODEL_SHORT)"
echo "GPU_QUEUE=$GPU_QUEUE (mem=$GPU_MEM)"

# Forward optional env vars to all jobs (always include PARENT_MODEL)
EXTRA_VARS=",PARENT_MODEL=${PARENT_MODEL}"
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS TM_SOURCE; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
[ -n "$EXTRA_VARS" ] && echo "Forwarding:$EXTRA_VARS"

TOTAL_JOBS=14
count=0

submit() {
    local ls="$1" las="$2" mp="$3"
    # Abbreviate for PBS job name: LS=Pa/PU/UM/CU, coarse=c/f, MP=r/sf/dz/sd
    local ls_abbr mp_abbr c_abbr
    case "$ls" in Pardiso) ls_abbr=Pa;; ParU) ls_abbr=PU;; UMFPACK) ls_abbr=UM;; CUDSS) ls_abbr=CU;; esac
    case "$mp" in raw) mp_abbr=r;; symfill) mp_abbr=sf;; dropzeros) mp_abbr=dz;; symdrop) mp_abbr=sd;; esac
    [ "$las" = "yes" ] && c_abbr=c || c_abbr=f
    # CUDSS uses the GPU job script; all others use the CPU job script
    if [ "$ls" = "CUDSS" ]; then
        local job_script=scripts/solvers/solve_TM_age_GPU.sh
        qsub -N "${MODEL_SHORT}_TM_${c_abbr}_${ls_abbr}_${mp_abbr}" \
            -l walltime=${WALLTIME_TM_SOLVE} \
            -q $GPU_QUEUE -l mem=$GPU_MEM \
            -v LINEAR_SOLVER=${ls},LUMP_AND_SPRAY=${las},MATRIX_PROCESSING=${mp}${EXTRA_VARS} \
            "$job_script"
    else
        local job_script=scripts/solvers/solve_TM_age_CPU.sh
        qsub -N "${MODEL_SHORT}_TM_${c_abbr}_${ls_abbr}_${mp_abbr}" \
            -l walltime=${WALLTIME_TM_SOLVE} \
            -v LINEAR_SOLVER=${ls},LUMP_AND_SPRAY=${las},MATRIX_PROCESSING=${mp}${EXTRA_VARS} \
            "$job_script"
    fi
    count=$((count + 1))
    local tag="${ls}/${las}/${mp}"
    echo "[$count] Submitted $tag."
    if [ "$count" -lt "$TOTAL_JOBS" ]; then
        echo "Waiting ${DELAY}s..."
        sleep $DELAY
    fi
}

# Pardiso — all MATRIX_PROCESSING variants
for las in no yes; do
    for mp in raw symfill dropzeros symdrop; do
        submit Pardiso "$las" "$mp"
    done
done

# ParU — raw only
for las in no yes; do
    submit ParU "$las" raw
done

# UMFPACK — raw only
for las in no yes; do
    submit UMFPACK "$las" raw
done

# CUDSS (GPU) — raw only
for las in no yes; do
    submit CUDSS "$las" raw
done

echo "Submitted ${count} jobs."
