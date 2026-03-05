#!/usr/bin/env bash
set -euo pipefail

# Submit all NONLINEAR_SOLVER × preconditioner/acceleration combinations.
# 60s delay between submissions to avoid concurrent precompilation OOM.
#
# Optional env vars forwarded to all jobs:
#   VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, CHECK_BOUNDS

DELAY=60  # seconds between submissions

# Forward optional env vars to all jobs
EXTRA_VARS=""
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
[ -n "$EXTRA_VARS" ] && echo "Forwarding:$EXTRA_VARS"

count=0

# # 1year
# qsub -v NONLINEAR_SOLVER=1year${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted 1year. Waiting ${DELAY}s..."
# sleep $DELAY

# # 10years
# qsub -v NONLINEAR_SOLVER=10years${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted 10years. Waiting ${DELAY}s..."
# sleep $DELAY

# # 100years
# qsub -v NONLINEAR_SOLVER=100years${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted 100years. Waiting ${DELAY}s..."
# sleep $DELAY

# Anderson — SpeedMapping
qsub -v NONLINEAR_SOLVER=anderson,AA_SOLVER=SpeedMapping${EXTRA_VARS},INITIAL_AGE=TMage scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/SpeedMapping. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — NLsolve
qsub -v NONLINEAR_SOLVER=anderson,AA_SOLVER=NLsolve${EXTRA_VARS},INITIAL_AGE=TMage scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/NLsolve. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — SIAMFANL
qsub -v NONLINEAR_SOLVER=anderson,AA_SOLVER=SIAMFANL${EXTRA_VARS},INITIAL_AGE=TMage scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/SIAMFANL. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — FixedPoint
qsub -v NONLINEAR_SOLVER=anderson,AA_SOLVER=FixedPoint${EXTRA_VARS,INITIAL_AGE=TMage} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/FixedPoint. Waiting ${DELAY}s..."
sleep $DELAY

# # Newton — Pardiso, no lump-and-spray (default), finitediff JVP
# qsub -v NONLINEAR_SOLVER=newton,JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — Pardiso, lump-and-spray, finitediff JVP
# qsub -v NONLINEAR_SOLVER=newton,JVP_METHOD=finitediff,LINEAR_SOLVER=Pardiso,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/Pardiso/LSprec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, no lump-and-spray, finitediff JVP
# qsub -v NONLINEAR_SOLVER=newton,JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=no${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/prec. Waiting ${DELAY}s..."
# sleep $DELAY

# # Newton — ParU, lump-and-spray, finitediff JVP
# qsub -v NONLINEAR_SOLVER=newton,JVP_METHOD=finitediff,LINEAR_SOLVER=ParU,LUMP_AND_SPRAY=yes${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
# count=$((count + 1))
# echo "[$count] Submitted newton/finitediff/ParU/LSprec. Waiting ${DELAY}s..."

echo "Submitted ${count} jobs."
