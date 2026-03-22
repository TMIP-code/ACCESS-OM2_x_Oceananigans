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
PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# Source model config for MODEL_SHORT and walltimes
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2; exit 1
fi
source "$MODEL_CONF"

# Auto-set GPU resources (single GPU benchmarks)
case "$GPU_QUEUE" in
    gpuvolta)  MEM_PER_GPU=96 ;;
    gpuhopper) MEM_PER_GPU=256 ;;
    *)         echo "Unknown GPU_QUEUE=$GPU_QUEUE (must be: gpuvolta or gpuhopper)"; exit 1 ;;
esac
NGPUS=1; GPU_NCPUS=12; GPU_MEM="${MEM_PER_GPU}GB"

echo "PARENT_MODEL=$PARENT_MODEL (MODEL_SHORT=$MODEL_SHORT)"
echo "GPU_QUEUE=$GPU_QUEUE (ngpus=$NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"

# Forward optional env vars to all jobs (always include PARENT_MODEL)
EXTRA_VARS=",PARENT_MODEL=${PARENT_MODEL}"
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS TRACE_SOLVER_HISTORY; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
echo "Forwarding:$EXTRA_VARS"

count=0

# # 1year
# qsub -N OM2-1_1yr -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/standard_runs/run_1year.sh
# count=$((count + 1))
# echo "[$count] Submitted 1year. Waiting ${DELAY}s..."
# sleep $DELAY

# # 10years
# qsub -N OM2-1_10yr -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/standard_runs/run_10years.sh
# count=$((count + 1))
# echo "[$count] Submitted 10years. Waiting ${DELAY}s..."
# sleep $DELAY

# # 100years
# qsub -N OM2-1_100yr -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM ${EXTRA_VARS:+-v ${EXTRA_VARS#,}} scripts/standard_runs/run_100years.sh
# count=$((count + 1))
# echo "[$count] Submitted 100years. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — SpeedMapping (archived)
# qsub -N OM2-1_AA_SM -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=SpeedMapping,INITIAL_AGE=TMage${EXTRA_VARS} scripts/solvers/solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/SpeedMapping. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — NLsolve (archived)
# qsub -N OM2-1_AA_NLs -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=NLsolve,INITIAL_AGE=TMage${EXTRA_VARS} scripts/solvers/solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/NLsolve. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — SIAMFANL (archived)
# qsub -N OM2-1_AA_SIA -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=SIAMFANL,INITIAL_AGE=TMage${EXTRA_VARS} scripts/solvers/solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/SIAMFANL. Waiting ${DELAY}s..."
# sleep $DELAY

# # Anderson — FixedPoint (archived)
# qsub -N OM2-1_AA_FP -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=FixedPoint,INITIAL_AGE=TMage${EXTRA_VARS} scripts/solvers/solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/FixedPoint. Waiting ${DELAY}s..."
# sleep $DELAY

# # Picard — plain fixed-point iteration (archived)
# qsub -N OM2-1_AA_Pi -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -l walltime=06:00:00 -v AA_SOLVER=Picard${EXTRA_VARS:+,$EXTRA_VARS} scripts/solvers/solve_periodic_AA.sh
# count=$((count + 1))
# echo "[$count] Submitted anderson/Picard. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), finitediff JVP
# qsub -N ${MODEL_SHORT}_NK_Pa -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), matrix JVP
# qsub -N ${MODEL_SHORT}_NK_Pa -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=matrix,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/matrix/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), exact JVP
# qsub -N ${MODEL_SHORT}_NK_Pa -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=exact,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no,INITIAL_AGE=0${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/exact/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# Newton — Pardiso, lump-and-spray, exact JVP
qsub -N ${MODEL_SHORT}_NK_Pa -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=exact,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=yes,INITIAL_AGE=0${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
count=$((count + 1))
echo "[$count] Submitted newton/exact/Pardiso/LSprec. Waiting ${DELAY}s..."
sleep $DELAY

# # Newton — Pardiso, lump-and-spray, finitediff JVP
# qsub -N ${MODEL_SHORT}_NK_Pa -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/LSprec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, no lump-and-spray, finitediff JVP
# qsub -N ${MODEL_SHORT}_NK_PU -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, lump-and-spray, finitediff JVP
# qsub -N ${MODEL_SHORT}_NK_PU -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM -v JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/solvers/solve_periodic_NK.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/LSprec. Waiting ${DELAY}s..."

echo "Submitted ${count} jobs."
