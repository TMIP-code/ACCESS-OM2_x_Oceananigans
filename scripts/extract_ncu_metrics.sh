#!/bin/bash
# Extract and compare NCU roofline metrics for K=12 vs K=24 strong-scaling tests
# Usage: bash scripts/extract_ncu_metrics.sh <job_id>
# Example: bash scripts/extract_ncu_metrics.sh 167737076

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/extract_ncu_metrics.sh <job_id>"
    echo "Example: bash scripts/extract_ncu_metrics.sh 167737076"
    exit 1
fi

JOB_ID=$1
LOG_DIR="logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun"

# Find the NCU report file for this job
NCU_REP=$(find "$LOG_DIR" -name "*${JOB_ID}*ncu*.ncu-rep" 2>/dev/null | head -1)

if [ -z "$NCU_REP" ]; then
    echo "ERROR: No .ncu-rep file found for job $JOB_ID"
    echo "Searched in: $LOG_DIR"
    exit 1
fi

echo "Found NCU report: $NCU_REP"
echo ""
echo "=== Extracting roofline metrics ==="
echo ""

module load cuda/12.9.0

# Extract per-kernel summary
echo "Per-kernel summary (roofline metrics):"
ncu --import "$NCU_REP" --print-summary per-kernel 2>&1 | head -150

echo ""
echo "=== Key metrics to compare ==="
echo "Look for:"
echo "  - Kernel Duration (ms)"
echo "  - Memory Throughput (%)"
echo "  - Compute Throughput (%)"
echo "  - SM Frequency (GHz)"
echo "  - DRAM Frequency (MHz)"
echo ""
echo "Compare against baseline (K=12 metrics documented in PROFILING_RESULTS.md)"
