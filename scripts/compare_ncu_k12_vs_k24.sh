#!/bin/bash
# Compare K=12 vs K=24 NCU metrics for strong-scaling analysis
# Jobs: K=12 (167715278/279), K=24 (167737076/090)

set -e

LOG_DIR="logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun"

echo "=== K=12 vs K=24 NCU Roofline Metrics Comparison ==="
echo ""

module load cuda/12.9.0

# Find NCU report files
echo "Searching for NCU report files..."
NCU_K12_1x4=$(find "$LOG_DIR" -name "*167715278*ncu*.ncu-rep" 2>/dev/null | head -1)
NCU_K12_1x8=$(find "$LOG_DIR" -name "*167715279*ncu*.ncu-rep" 2>/dev/null | head -1)
NCU_K24_1x4=$(find "$LOG_DIR" -name "*167737076*ncu*.ncu-rep" 2>/dev/null | head -1)
NCU_K24_1x8=$(find "$LOG_DIR" -name "*167737090*ncu*.ncu-rep" 2>/dev/null | head -1)

echo "K=12 1×4: $NCU_K12_1x4"
echo "K=12 1×8: $NCU_K12_1x8"
echo "K=24 1×4: $NCU_K24_1x4"
echo "K=24 1×8: $NCU_K24_1x8"
echo ""

# Extract K=12 baseline metrics
echo "=== K=12 Baseline (1×4) ==="
if [ -n "$NCU_K12_1x4" ]; then
    echo "File: $(basename "$NCU_K12_1x4")"
    ncu --import "$NCU_K12_1x4" --print-summary per-kernel 2>&1 | grep -A 20 "Kernel" | head -30
else
    echo "WARNING: K=12 1×4 report not found"
fi
echo ""

echo "=== K=12 Baseline (1×8) ==="
if [ -n "$NCU_K12_1x8" ]; then
    echo "File: $(basename "$NCU_K12_1x8")"
    ncu --import "$NCU_K12_1x8" --print-summary per-kernel 2>&1 | grep -A 20 "Kernel" | head -30
else
    echo "WARNING: K=12 1×8 report not found"
fi
echo ""

# Extract K=24 test metrics
echo "=== K=24 Test (1×4) ==="
if [ -n "$NCU_K24_1x4" ]; then
    echo "File: $(basename "$NCU_K24_1x4")"
    ncu --import "$NCU_K24_1x4" --print-summary per-kernel 2>&1 | grep -A 20 "Kernel" | head -30
else
    echo "WARNING: K=24 1×4 report not found"
fi
echo ""

echo "=== K=24 Test (1×8) ==="
if [ -n "$NCU_K24_1x8" ]; then
    echo "File: $(basename "$NCU_K24_1x8")"
    ncu --import "$NCU_K24_1x8" --print-summary per-kernel 2>&1 | grep -A 20 "Kernel" | head -30
else
    echo "WARNING: K=24 1×8 report not found"
fi
echo ""

echo "=== Analysis Summary ==="
echo ""
echo "Key metrics to compare:"
echo "  1. Kernel Duration (ms) — should scale linearly with partition size"
echo "  2. Memory Throughput (%) — should stay constant (>37% for 1×8)"
echo "  3. Compute Throughput (%) — should stay constant (~22%)"
echo ""
echo "Success criteria:"
echo "  - Per-kernel scaling maintained (1.90×) or improved"
echo "  - Memory throughput ≥37% (no degradation)"
echo "  - Wall-time improvement >4.4% (better than K=12 baseline)"
