#!/usr/bin/env bash
set -euo pipefail

# Submit all SOLVE_METHOD × preconditioner/acceleration combinations.
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

# 1year
qsub -v SOLVE_METHOD=1year${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted 1year. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — speedmapping
qsub -v SOLVE_METHOD=anderson,ACCELERATION_METHOD=speedmapping${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/speedmapping. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — anderson
qsub -v SOLVE_METHOD=anderson,ACCELERATION_METHOD=anderson${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/anderson. Waiting ${DELAY}s..."
sleep $DELAY

# Newton — nonsym (default), finitediff JVP
qsub -v SOLVE_METHOD=newton,JVP_METHOD=finitediff${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted newton/finitediff/nonsym. Waiting ${DELAY}s..."
sleep $DELAY

# Newton — sym_cleaned, finitediff JVP
qsub -v SOLVE_METHOD=newton,JVP_METHOD=finitediff,PRECONDITIONER_MATRIX_TYPE=sym_cleaned${EXTRA_VARS} scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))

echo "Submitted ${count} jobs."
