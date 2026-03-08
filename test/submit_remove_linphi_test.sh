#!/usr/bin/env bash
set -euo pipefail

# Submit the self-consistency test (GPU) and then the comparison (CPU).

echo "Submitting test_nolinphi (self-consistency test)..."
JOB_A=$(qsub test/test_nolinphi_job.sh)
echo "  Job A: $JOB_A"

JOB_A_ID=$(echo "$JOB_A" | sed 's/\..*//')

echo "Submitting comparison job (depends on test completing)..."
JOB_B=$(qsub -W depend=afterok:${JOB_A_ID} test/compare_job.sh)
echo "  Job B: $JOB_B (depends on $JOB_A_ID)"

echo ""
echo "Submitted 2 jobs:"
echo "  [A] test_nolinphi:  $JOB_A"
echo "  [B] compare:        $JOB_B (runs after A completes)"
echo ""
echo "Check results in logs/julia/test/"
