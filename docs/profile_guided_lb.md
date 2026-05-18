# Profile-guided load balancing

The static LB methods catalogued in [partition_balance.md](partition_balance.md)
(`:cell`, `:surface`, `:mix`, `:minmax`) all assume some model of per-row work
— wet cells, wet columns, or a combination. The profiling results in
[profiling_results_v2.md](profiling_results_v2.md) showed those models don't
fully match real GPU work distribution: even `:surface` (the best static
method) still leaves OM2-01 1×8 with ~25% kernel-busy imbalance.

**This doc covers the ground-truth measurement tool and how to use it.**

---

## What it does

[`scripts/analysis/nsys_per_rank_busy.py`](../scripts/analysis/nsys_per_rank_busy.py)
ingests one or more `*_rank*.nsys-rep` files from a distributed profile and
reports, per rank, the **total GPU kernel-busy time** plus several imbalance
metrics. This is the literal ground truth of "how much compute work did this
GPU do" — independent of any partition-side model.

### Why kernel-busy time, not MPI_Waitall?

`MPI_Waitall` is anti-correlated with `kernel_busy` by definition: the slowest
rank finishes work last, so its wait is small; the fastest finishes early and
sits in MPI_Waitall. The wall is `kernel_busy + waitall + small_overhead`
on every rank, but only the *slowest* rank's `kernel_busy` actually shapes
the total wall time. Measuring `kernel_busy` gives the actionable signal
(real work imbalance); measuring `MPI_Waitall` just inverts it.

---

## How to run

The tool needs `nsys export --type=sqlite` to read `.nsys-rep` files. That
step is memory-hungry, so the **first run** must be on a compute node via PBS.
Subsequent runs (after SQLite is cached next to each `.nsys-rep`) can be done
on a login node.

### First run — PBS wrapper

[`scripts/tests/run_nsys_per_rank_busy.sh`](../scripts/tests/run_nsys_per_rank_busy.sh)
takes a shell glob via the `NSYS_GLOB` env var:

```bash
NSYS_GLOB='logs/julia/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2_LBS_1yearfast_168522455.gadi-pbs_profile_*_rank*.nsys-rep' \
    qsub -v NSYS_GLOB scripts/tests/run_nsys_per_rank_busy.sh
```

Express queue, 16 GB, 1 CPU, 30 min walltime; output to
`logs/analysis/nsys_per_rank_busy_<JOB_ID>.gadi-pbs.log`.

The PBS step does:
1. `module load cuda/12.9.0` (gives `nsys`).
2. For each rank `.nsys-rep` matched by the glob, runs
   `nsys export --type=sqlite ...` if the `.sqlite` doesn't yet exist (cached).
3. Runs the Python script which queries each SQLite and aggregates per rank.

### Subsequent re-runs — login node

Once the `.sqlite` files exist next to the `.nsys-rep` files, the Python
script alone is enough. Re-runs are essentially free (sub-second SQL queries):

```bash
python3 scripts/analysis/nsys_per_rank_busy.py \
    'logs/julia/.../cgridtransports_wdiagnosed_centered2_AB2_LBS_1yearfast_168522455.gadi-pbs_profile_*_rank*.nsys-rep' \
    --csv /scratch/y99/bp3051/per_rank_LBS_1x8.csv
```

---

## Reading the output

```
rank  n_kernels   kernel_busy   capture_wall   util%
----------------------------------------------------
   0       5806       14.995s        16.918s   88.6%
   1       7010       10.295s        18.103s   56.9%
   ...
   7       7248        8.965s        18.047s   49.7%
----------------------------------------------------
mean kernel_busy:      11.929s
max  kernel_busy:      14.995s    (rank 0, busiest)
min  kernel_busy:       8.965s    (rank 7, idlest)

Imbalance metrics:
  max-min absolute:                       6.03s   (rank 7's idle time waiting on rank 0)
  (max-min)/max → idlest's idle fraction:  40.2%
  max/min ratio:                         1.673×
  (max-mean)/max → system-wide loss:      20.4%   (24.53s total GPU-time wasted across 8 ranks)
  (max-mean)/mean → overload of busiest:   25.7%
```

Five imbalance metrics, ordered by usefulness:

| Metric | Formula | What it means |
|---|---|---|
| **max-min (s)** | `max(busy) − min(busy)` | How many seconds the idlest rank waited on the busiest. Direct cost. |
| **(max-min)/max** | `(max − min) / max` | The idlest rank's idle fraction — % of wall it spent doing nothing. |
| **max/min ratio** | `max / min` | Ratio of busiest-to-idlest work. 1.0 = perfect. |
| **(max-mean)/max** | `(max − mean) / max` | System-wide GPU-time loss: fraction of `N × max` wasted by imbalance. |
| **(max-mean)/mean** | `(max − mean) / mean` | How much the busiest rank is overloaded vs the average. Standard HPC "imbalance%". |

For wall-time impact, **(max-mean)/max** is the most relevant: if you
perfectly rebalanced (so every rank's busy time = mean), wall would drop
from `max` to `mean`. That's a `(max-mean)/max` reduction.

For OM2-01 1×8 +LBS: ~20% wall reduction is theoretically achievable from
rebalancing alone, i.e. ~20 min off the 1h 38m bench → ~1h 18m.

---

## Iterative refinement workflow

The static LB methods need a workload model; this measurement *is* the model.
So profile-guided LB is the natural extension:

```text
┌────────────────────────────────────────────────────────────┐
│ 1. Build initial partition (e.g. LOAD_BALANCE=surface)     │
│    → 1x8_LBS partition files                               │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│ 2. Run a SHORT nsys-instrumented bench at the target       │
│    partition (PROFILE=yes, BENCHMARK_STEPS=240)            │
│    → 8 per-rank .nsys-rep files                            │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│ 3. Run nsys_per_rank_busy.py via the PBS wrapper           │
│    → per-rank busy_i times                                 │
│    → (max-min)/max idle fraction                           │
└────────────────────────────────────────────────────────────┘
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
       below threshold           still imbalanced
       (e.g. < 5%)            (e.g. > 15%)
              ↓                       ↓
            STOP        ┌──────────────────────────────────┐
                        │ 4. Use busy_i as new workload    │
                        │    weights to rebuild partition: │
                        │    rank i is given fewer y-rows  │
                        │    proportional to its busy_i    │
                        └──────────────────────────────────┘
                                       ↓
                          (back to step 2 — usually 1–2 iters)
```

### Practical iteration recipe

A simple way to map per-rank `busy_i` back to a partition refinement:

1. Current partition has rank `i` owning `n_i` y-rows.
2. Measure `busy_i` per rank.
3. Compute the **target work** = `mean(busy_i)`.
4. For each rank, compute its **work density** = `busy_i / n_i`
   (busy time per y-row it owns — measured cost per row in that slab).
5. Adjust slab sizes: rank `i` should own `n_i' = target / work_density_i`
   y-rows (so that with its measured per-row cost, its slab finishes at the
   target). Renormalise so `Σ n_i' = Ny`.
6. Round to integers, ensure `min_size ≥ Hy + 2`.
7. Rebuild the partition with those explicit sizes, run nsys again.

This is a fixed-point iteration on the work-density estimate. In practice
one or two iterations should converge to <5% imbalance (modulo non-y-row
costs like halo packing that don't scale with slab size).

### What this needs (not yet implemented)

- A way to pass **explicit per-rank y-row sizes** into the partition
  builder. Currently `LOAD_BALANCE=cell|surface|mix|minmax|no` is the only
  knob; an `EXPLICIT_Y_SIZES="438,213,219,268,276,365,577,344"` env var
  would expose it. The greedy splitter in [load_balance.jl](../src/shared_utils/load_balance.jl)
  already produces such tuples internally — just needs an entry-point.
- A small driver script that runs steps 2–4 in a loop until convergence
  (or a max iteration count).

---

## Caveats

- **Capture range matters.** `BENCHMARK_STEPS=240` after 3 warmup steps gives
  a clean steady-state measurement. Don't include the warmup in the analysis
  window (the script doesn't filter; it just sums all kernels in the rep file).
- **`.sqlite` files are big** — ~150 MB per rank for 240-step bench traces,
  so 8 ranks = ~1.2 GB. Living next to the `.nsys-rep` files is fine on
  scratch / gdata but tidy up old ones if you start filling quota.
- **Single-precision metric.** Two ranks with similar `kernel_busy` but
  different `n_kernels` may differ in per-kernel cost (heavy kernels vs many
  light ones) — the analyser doesn't break that down. If you need a
  per-kernel breakdown, query the same SQLite with
  `GROUP BY demangledName` and order by `SUM(end-start) DESC`.
- **NVTX-aware per-y-row attribution** is a separate extension. To attribute
  cost to specific y-rows rather than just the whole rank's slab, we'd need
  NVTX ranges around row-batch loops. Not needed for the basic refinement
  loop above (which works at slab granularity).

---

## First measured result (OM2-01 1×8 +LBS, job 168522455)

| | value |
|---|---|
| busiest rank | 0 (south, 14.995 s) |
| idlest rank | 7 (north fold, 8.965 s) |
| idle waste (rank 7) | 6.03 s (40.2% of wall) |
| system-wide GPU-time waste | 24.53 s across 8 ranks (20.4%) |
| max/min ratio | 1.67× |

Static `:surface` predicted `imb%(surface) = 0.33%` for this partition —
near-perfect column balance. The real kernel imbalance is 25%; the static
model is missing whatever the dominant cost actually is (column depth,
halo perimeter, fold-specific kernels, ?). The profile-guided iteration
above is the way to close that gap empirically.
