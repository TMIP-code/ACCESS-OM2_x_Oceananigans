# Synchronized GC: implementation in this repo

> Combining sync-GC with temporal blocking and load balancing on a
> single 1×2 OM2-025 H200 run: see
> [COMBINED_SCALING_BENCHMARKS.md](COMBINED_SCALING_BENCHMARKS.md).
> Note: under TBLOCKING, `SYNC_GC_NSTEPS = N` means every N **batches**
> (= N·K raw steps), not every N raw steps.

For the rationale (why synchronized GC helps distributed Oceananigans runs, and the
ClimaAtmos pattern we follow), see [DISTRIBUTED_GC.md](DISTRIBUTED_GC.md). This doc
covers **what we wired up** and how to use it.

## Scope

Synchronized GC is currently only wired into the benchmark/profiling entry point
([src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl)). It is **not** active
in the main pipeline (`run_1year.jl`, `solve_periodic_NK.jl`, etc.). If the
benchmark results make the case for it, the same hook can be lifted into
`setup_simulation.jl` behind the same env var.

## What it does

On distributed runs only, registers an Oceananigans `Callback` that calls
`GC.gc(false)` every `SYNC_GC_NSTEPS` iterations of the benchmark loop. The callback
fires on the same iteration on every rank; the surrounding collectives (halo
exchange, reductions) provide the implicit barrier — no explicit `MPI.Barrier` in
the callback. `GC.gc(false)` is the *incremental* collection (cheap, a few ms);
full collection is avoided.

## Implementation

### [src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl)

Between the `reset!(simulation)` block and the `CUDA.@profile` wrap:

```julia
SYNC_GC_NSTEPS = parse(Int, get(ENV, "SYNC_GC_NSTEPS", "0"))
if arch isa Distributed && SYNC_GC_NSTEPS > 0
    sync_gc!(sim) = (GC.gc(false); nothing)
    add_callback!(simulation, sync_gc!, IterationInterval(SYNC_GC_NSTEPS))
    @info "Synchronized GC enabled: GC.gc(false) every $SYNC_GC_NSTEPS iterations"
end
```

Gating:
- `arch isa Distributed` — single-rank runs don't need the sync.
- `SYNC_GC_NSTEPS > 0` — env-controllable, default off.

### [scripts/standard_runs/run_1year_benchmark.sh](../scripts/standard_runs/run_1year_benchmark.sh)

Under `if [ "$PROFILE" = "yes" ]`:

- Auto-defaults `SYNC_GC_NSTEPS=5` so that profile runs have sync on by default.
- Computes `sync_gc_tag = syncGCyes_N<N>` or `syncGCno`, then appends it to the
  profile filename so with/without-sync runs land side by side:
  ```
  {MODEL_CONFIG}_1yearfast_{JOB_ID}_profile_{sync_gc_tag}_rank{N}.nsys-rep
  ```

### [scripts/driver.sh](../scripts/driver.sh)

The `run1yrfast` submission passes `SYNC_GC_NSTEPS` (and `BENCHMARK_STEPS`) through
`--vars` so they propagate into the PBS job:

```bash
--vars "PARTITION=${PARTITION},PROFILE=${PROFILE:-no},SYNC_GC_NSTEPS=${SYNC_GC_NSTEPS:-},BENCHMARK_STEPS=${BENCHMARK_STEPS:-}"
```

Unset → the PBS script's default (5 when `PROFILE=yes`) applies.

## Usage

Profile with sync GC (default when `PROFILE=yes`):

```bash
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 PROFILE=yes \
    JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

Profile without sync GC (baseline):

```bash
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 PROFILE=yes SYNC_GC_NSTEPS=0 \
    JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

Custom interval (e.g., for tuning N):

```bash
BENCHMARK_STEPS=200 SYNC_GC_NSTEPS=20 PARTITION=2x2 PROFILE=yes \
    JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

## Verifying it worked

1. **Job log** — look for `Synchronized GC enabled: GC.gc(false) every N iterations`
   on every rank (or `synchronized GC disabled (baseline)` when off).
2. **Profile filename** — the `syncGCyes_N<N>` or `syncGCno` tag disambiguates
   A/B runs in the same directory.
3. **Nsight Systems timeline** (see [PROFILING.md](PROFILING.md) for how to
   collect/view) —
   - With `JULIA_NVTX_CALLBACKS=gc` (auto-set when `PROFILE=yes`), Julia GC events
     appear as NVTX ranges.
   - Sync on: GC ranges align at the sync iterations across ranks.
   - Sync off: GC ranges stagger across ranks, typically correlating with long
     `MPI_Waitall` stalls on the *other* ranks.
4. **nsys stats reports** — `--report=mpi_event_sum` gives per-call MPI time;
   `--report=nvtx_pushpop_sum` gives GC time by rank. Both should move in the
   expected direction between `syncGCno` and `syncGCyes_N<N>` variants.

## Tuning N

See [DISTRIBUTED_GC.md § Choosing the interval N](DISTRIBUTED_GC.md#choosing-the-interval-n)
for the general rule (`N ≤ G_min / 2`, where `G_min` is the shortest organic GC
interval across ranks). The PBS-side default of `N=5` is a *make-it-visible* value
for short profile runs; proper tuning requires measuring `Base.gc_num().pause`
deltas per step across ranks in a baseline run first.

## Known caveats

- **Benchmark-only.** The hook only runs when
  [src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl) is the entry point.
  Production runs (`run_1year.jl` etc.) are unaffected — intentionally, until the
  A/B benchmarks show a clear win.
- **No `GC.enable(false)`.** ClimaAtmos tried disabling GC around the solve and
  reverted it because it leaked memory (see comment in their `solve.jl`). We
  follow their pattern and don't touch `GC.enable`.
- **Incremental only.** `GC.gc(false)` does not necessarily reclaim memory to the
  OS. For long runs, a heap that keeps growing will eventually trigger a full GC
  on one rank organically and the sync will be defeated. If that happens, consider
  stacking `GC.gc(true)` at a coarser interval (e.g., every 100 steps) — see
  [DISTRIBUTED_GC.md § Refinements](DISTRIBUTED_GC.md#refinements).
