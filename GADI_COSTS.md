# Gadi SU Cost Reference

Regenerate this file with:
```bash
julia scripts/summarize_OU_logs.jl
```

## Formula

```
SU = Queue_Rate × Max(NCPUs, mem_requested_GB / mem_per_core_GB) × Walltime_hours
```

where `mem_per_core = node_RAM / node_CPUs` depends on the node type.

| Queue | Rate (SU/CPU·hr) | Node type | CPUs/node | RAM/node | Mem/core |
|-------|-----------------|-----------|-----------|----------|----------|
| normal | 2 | Cascade Lake | 48 | 192 GiB | 4 GiB |
| express | 6 | Cascade Lake | 48 | 192 GiB | 4 GiB |
| gpuvolta | 3 | V100 + Cascade Lake | 48 | 384 GiB | 8 GiB |

Source: https://opus.nci.org.au/spaces/Help/pages/236880942/Job+Costs

Formula validated against 340 `.OU` log files (see `scripts/parse_OU_logs.jl`).

## Configurations used and SU rates

| Queue | NCPUs | NGPUs | Mem | SU/hr | Jobs | Total SU |
|-------|-------|-------|-----|-------|------|----------|
| express | 12 | — | 47.0GB | 72 | 108 | 720.78 |
| express | 48 | — | 190.0GB | 288 | 12 | 108.48 |
| gpuvolta | 12 | 1 | 96.0GB | 36 | 6 | 20.20 |
| gpuvolta | 12 | 1 | 47.0GB | 36 | 149 | 1673.35 |
| normal | 12 | — | 47.0GB | 24 | 11 | 12.16 |
| normal | 48 | — | 190.0GB | 96 | 54 | 1303.25 |

**Total: 340 jobs, 3838.22 SU**
