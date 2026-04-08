#!/bin/bash
#PBS -N nsys_stats
#PBS -P y99
#PBS -q express
#PBS -l ncpus=12
#PBS -l mem=47GB
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/y99+scratch/y99
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

module load cuda/12.9.0

outdir=logs/nsys_stats
mkdir -p "$outdir"

# Profile files to analyze (rank 0 only for distributed, full for serial)
declare -A profiles
profiles[OM2-1_1x1]="logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602219.gadi-pbs_profile.nsys-rep"
profiles[OM2-1_2x2_rank0]="logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602222.gadi-pbs_profile_rank0.nsys-rep"
profiles[OM2-1_1x4_rank0]="logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602224.gadi-pbs_profile_rank0.nsys-rep"
profiles[OM2-025_1x1]="logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602226.gadi-pbs_profile.nsys-rep"
profiles[OM2-025_2x2_rank0]="logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602228.gadi-pbs_profile_rank0.nsys-rep"
profiles[OM2-025_1x4_rank0]="logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_164602231.gadi-pbs_profile_rank0.nsys-rep"

for name in OM2-1_1x1 OM2-1_2x2_rank0 OM2-1_1x4_rank0 OM2-025_1x1 OM2-025_2x2_rank0 OM2-025_1x4_rank0; do
    prof="${profiles[$name]}"
    outfile="$outdir/${name}.txt"
    echo "Processing $name → $outfile"

    if [ ! -f "$prof" ]; then
        echo "MISSING: $prof" > "$outfile"
        continue
    fi

    {
        echo "=== $name ==="
        echo "File: $prof ($(du -h "$prof" | cut -f1))"
        echo ""

        for report in cuda_gpu_kern_sum cuda_api_sum; do
            echo "--- $report ---"
            nsys stats --report="$report" --format=table "$prof" 2>&1
            echo ""
        done

        # MPI report only for distributed runs
        if [[ "$name" == *rank* ]]; then
            echo "--- mpi_event_sum ---"
            nsys stats --report=mpi_event_sum --format=table "$prof" 2>&1
            echo ""
        fi
    } > "$outfile" 2>&1

    echo "Done: $outfile"
done

echo "All done. Output in $outdir/"
