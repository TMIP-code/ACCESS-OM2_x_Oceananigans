#!/usr/bin/env bash
set -euo pipefail

# Submit all NONLINEAR_SOLVER × preconditioner/acceleration combinations.
# 10s delay between submissions to avoid concurrent precompilation OOM.
#
# Optional env vars forwarded to all jobs:
#   VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, CHECK_BOUNDS
#
# GPU queue selection (default: gpuhopper):
#   GPU_QUEUE=gpuvolta bash scripts/submit_all_solver_modes.sh

DELAY=10  # seconds between submissions
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# Auto-set GPU memory based on queue
case "$GPU_QUEUE" in
    gpuvolta)  GPU_MEM=96GB ;;
    gpuhopper) GPU_MEM=256GB ;;
    *)         echo "Unknown GPU_QUEUE=$GPU_QUEUE"; exit 1 ;;
esac

echo "GPU_QUEUE=$GPU_QUEUE (mem=$GPU_MEM)"

# Forward optional env vars to all jobs
EXTRA_VARS=""
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS TRACE_SOLVER_HISTORY; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
[ -n "$EXTRA_VARS" ] && echo "Forwarding:$EXTRA_VARS"

count=0

# # 1year
# qsub -N OM2-1_1yr -q $GPU_QUEUE -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/ACCESS-OM2-1_run_1year.sh
# count=$((count + 1))
# echo "[$count] Submitted 1year. Waiting ${DELAY}s..."
# sleep $DELAY

# # 10years
# qsub -N OM2-1_10yr -q $GPU_QUEUE -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/ACCESS-OM2-1_run_10years.sh
# count=$((count + 1))
# echo "[$count] Submitted 10years. Waiting ${DELAY}s..."
# sleep $DELAY

# # 100years
# qsub -N OM2-1_100yr -q $GPU_QUEUE -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/ACCESS-OM2-1_run_100years.sh
# count=$((count + 1))
# echo "[$count] Submitted 100years. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — SpeedMapping (archived)
# qsub -N OM2-1_AA_SM -q $GPU_QUEUE -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=SpeedMapping,INITIAL_AGE=TMage${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/SpeedMapping. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — NLsolve (archived)
# qsub -N OM2-1_AA_NLs -q $GPU_QUEUE -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=NLsolve,INITIAL_AGE=TMage${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/NLsolve. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — SIAMFANL (archived)
# qsub -N OM2-1_AA_SIA -q $GPU_QUEUE -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=SIAMFANL,INITIAL_AGE=TMage${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/SIAMFANL. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — FixedPoint (archived)
# qsub -N OM2-1_AA_FP -q $GPU_QUEUE -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=FixedPoint,INITIAL_AGE=TMage${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/FixedPoint. Waiting ${DELAY}s..."
# sleep $DELAY

# # Picard — plain fixed-point iteration (archived)
# qsub -N OM2-1_AA_Pi -q $GPU_QUEUE -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=Picard${EXTRA_VARS:+,$EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/Picard. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), finitediff JVP
# qsub -N OM2-1_NK_Pa -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), matrix JVP
# qsub -N OM2-1_NK_Pa -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=matrix,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/matrix/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), exact JVP
# qsub -N OM2-1_NK_Pa -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=exact,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no,INITIAL_AGE=0${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/exact/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# Newton — Pardiso, lump-and-spray, exact JVP
qsub -N OM2-1_NK_Pa -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=exact,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=yes,INITIAL_AGE=0${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
count=$((count + 1))
echo "[$count] Submitted newton/exact/Pardiso/LSprec. Waiting ${DELAY}s..."
sleep $DELAY

# # Newton — Pardiso, lump-and-spray, finitediff JVP
# qsub -N OM2-1_NK_Pa -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/LSprec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, no lump-and-spray, finitediff JVP
# qsub -N OM2-1_NK_PU -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, lump-and-spray, finitediff JVP
# qsub -N OM2-1_NK_PU -q $GPU_QUEUE -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/ACCESS-OM2-1_solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/LSprec. Waiting ${DELAY}s..."

echo "Submitted ${count} jobs."
