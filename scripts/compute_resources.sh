# Compute PBS resource variables from PARTITION, GPU_QUEUE, and CPU_QUEUE.
# Sourced by both env_defaults.sh (for PBS scripts) and driver.sh (for qsub).
#
# Expects these variables to be set:
#   PARTITION  (e.g., "1x1", "2x2")
#   GPU_QUEUE  (gpuvolta or gpuhopper)
#   CPU_QUEUE  (express, normal, hugemem, or megamem)
#
# Sets:
#   PARTITION_X, PARTITION_Y, RANKS
#   NGPUS, GPU_NCPUS, GPU_MEM, GPU_MEM_SINGLE, MEM_PER_GPU
#   CPU_NCPUS, CPU_MEM, MEM_PER_CPU

PARTITION=${PARTITION:-1x1}
PARTITION_X="${PARTITION%%x*}"
PARTITION_Y="${PARTITION#*x}"
RANKS=$(( PARTITION_X * PARTITION_Y ))

# GPU resources
NGPUS=$RANKS
GPU_NCPUS=$(( NGPUS * 12 ))
case "$GPU_QUEUE" in
    gpuvolta)  MEM_PER_GPU=96 ;;
    gpuhopper) MEM_PER_GPU=256 ;;
    *) echo "ERROR: Unknown GPU_QUEUE: $GPU_QUEUE (must be gpuvolta or gpuhopper)" >&2; exit 1 ;;
esac
GPU_MEM="$(( NGPUS * MEM_PER_GPU ))GB"
GPU_MEM_SINGLE="${MEM_PER_GPU}GB"

# CPU resources
CPU_NCPUS=$RANKS
case "$CPU_QUEUE" in
    express|normal) MEM_PER_CPU=4 ;;
    hugemem)        MEM_PER_CPU=32 ;;
    megamem)        MEM_PER_CPU=64 ;;
    *) echo "ERROR: Unknown CPU_QUEUE: $CPU_QUEUE (must be express, normal, hugemem, or megamem)" >&2; exit 1 ;;
esac
# Enforce queue minimum memory (hugemem ≥ 192 GB, megamem ≥ 1440 GB)
CPU_MEM_BALANCED=$(( CPU_NCPUS * MEM_PER_CPU ))
case "$CPU_QUEUE" in
    hugemem) CPU_MEM_MIN=192 ;;
    megamem) CPU_MEM_MIN=1440 ;;
    *)       CPU_MEM_MIN=0 ;;
esac
CPU_MEM_VAL=$(( CPU_MEM_BALANCED > CPU_MEM_MIN ? CPU_MEM_BALANCED : CPU_MEM_MIN ))
CPU_MEM="${CPU_MEM_VAL}GB"
