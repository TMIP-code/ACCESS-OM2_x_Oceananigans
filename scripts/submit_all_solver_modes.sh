#!/usr/bin/env bash
set -euo pipefail

# Submit all SOLVE_METHOD × preconditioner/acceleration combinations.
# 60s delay between submissions to avoid concurrent precompilation OOM.

DELAY=60  # seconds between submissions

count=0

# 1year
qsub -v SOLVE_METHOD=1year scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted 1year. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — speedmapping
qsub -v SOLVE_METHOD=anderson,ACCELERATION_METHOD=speedmapping scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/speedmapping. Waiting ${DELAY}s..."
sleep $DELAY

# Anderson — anderson
qsub -v SOLVE_METHOD=anderson,ACCELERATION_METHOD=anderson scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted anderson/anderson. Waiting ${DELAY}s..."
sleep $DELAY

# Newton — nonsym (default), finitediff JVP
qsub -v SOLVE_METHOD=newton,JVP_METHOD=finitediff scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))
echo "[$count] Submitted newton/finitediff/nonsym. Waiting ${DELAY}s..."
sleep $DELAY

# Newton — sym_cleaned, finitediff JVP
qsub -v SOLVE_METHOD=newton,JVP_METHOD=finitediff,PRECONDITIONER_MATRIX_TYPE=sym_cleaned scripts/ACCESS-OM2-1_GPU_job.sh
count=$((count + 1))

echo "Submitted ${count} jobs."
