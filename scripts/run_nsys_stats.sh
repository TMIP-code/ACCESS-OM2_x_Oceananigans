#!/bin/bash
#PBS -P y99
#PBS -q express
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=00:10:00
#PBS -l storage=gdata/y99+scratch/y99
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

module load cuda/12.9.0

profiles=(
  "logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wdiagnosed_centered2_AB2_1yearfast_163863598.gadi-pbs_profile.nsys-rep"
  "logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wprescribed_centered2_AB2_1yearfast_163863027.gadi-pbs_profile_rank0.nsys-rep"
  "logs/julia/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wprescribed_centered2_AB2_1yearfast_163863599.gadi-pbs_profile.nsys-rep"
  "logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wprescribed_centered2_AB2_1yearfast_163863497.gadi-pbs_profile.nsys-rep"
  "logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1960-1979/standardrun/cgridtransports_wprescribed_centered2_AB2_1yearfast_163863501.gadi-pbs_profile_rank0.nsys-rep"
)

for p in "${profiles[@]}"; do
  echo "========================================"
  echo "PROFILE: $(basename $p)"
  echo "========================================"
  nsys stats --report cuda_gpu_kern_sum --force-export=true "$p" 2>&1 | head -80
  echo ""
done
