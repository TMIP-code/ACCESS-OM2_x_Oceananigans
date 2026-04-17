# Synchronized GC for distributed Oceananigans (multi-GPU MPI)

## Problem

In multi-GPU MPI runs, Julia's GC triggers independently on each rank at
different times. One rank GCs → all others idle at the next collective (halo
exchange, reduction). Next timestep, another rank GCs. Wall clock becomes
dominated by GC-induced stragglers even though per-rank GC time is small.

## Recommended pattern (verified in CliMA/ClimaAtmos.jl)

Only production-grade implementation found across CliMA and NumericalEarth
orgs. Two components:

### 1. Periodic callback that runs incremental GC on every rank every N steps

From [ClimaAtmos/src/callbacks/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/callbacks.jl):

```julia
function gc_func(integrator)
    GC.gc(false)   # incremental; logging elided
    return nothing
end
```

Registered only when distributed, env-configurable interval (default 1000):

```julia
function gc_callback(comms_ctx)
    if is_distributed(comms_ctx)
        return (call_every_n_steps(gc_func,
                                   parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
                                   skip_first = true,),)
    end
    return ()
end
```

### 2. Barriers around the solve to bound startup/teardown skew

From [ClimaAtmos/src/solver/solve.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/solver/solve.jl):

```julia
if CA.is_distributed(comms_ctx)
    # GC.enable(false) # disabling GC causes a memory leak
    ClimaComms.barrier(comms_ctx)
    (sol, walltime) = timed_solve!(integrator)
    ClimaComms.barrier(comms_ctx)
    GC.enable(true)
```

## Key design notes

- **No explicit `MPI.Barrier` inside `gc_func`.** Synchronization is implicit —
  the callback fires on the same iteration on every rank, and the surrounding
  timestepping already has collectives. Simpler than double-barrier,
  apparently sufficient in practice.
- **`GC.gc(false)` (incremental)**, not full — cheaper; full collection only
  if needed.
- **Do NOT use `GC.enable(false)`.** The comment in `solve.jl` is
  load-bearing: ClimaAtmos tried it and reverted it because it leaks memory.
  No CliMA or NumericalEarth code uses `GC.enable(false)` in a stepping loop.
- **Only install the callback when distributed.** Single-rank runs don't need
  it.
- **ClimaAtmos's default interval of 1000 steps is workload-specific.** Do
  not copy it blindly — see tuning rule below. ClimaAtmos can afford N=1000
  because their per-step allocation rate is low enough that organic GC
  doesn't fire within 1000 steps on any rank.

## Choosing the interval N

**Rule:** the sync interval must be **shorter than the shortest organic
GC interval on any rank.**

Mechanism: a collective `GC.gc(false)` at step K resets heap pressure on
every rank. If rank-1 was on track to trigger organic GC at step K+5, the
collective at step K preempts it. But only if the collective fires *before*
any rank's heap threshold is reached.

Concrete tuning procedure:

1. **Measure organic GC cadence first.** Log `Base.gc_num().pause` deltas
   per step on each rank (without any sync callback). Find the shortest
   inter-GC interval across ranks — call it `G_min`.
2. **Set N ≤ G_min / 2** as a starting point. If you see organic GC every
   ~20 steps, start with N=10.
3. **Halve until stragglers disappear** or overhead shows up in total
   wall clock.
4. **Validate:** with the sync installed, `Base.gc_num().pause` should
   only increment on the sync-step — no rank should be triggering
   organic GC between syncs.

`GC.gc(false)` is incremental and typically costs a few ms; short
intervals (every 5–10 steps) are viable when allocation pressure is high.

## Refinements

- **Stack incremental + full.** `GC.gc(false)` every ~10 steps to keep
  heaps trimmed, plus `GC.gc(true)` every ~100 to actually reclaim to
  the OS and give `CUDA.reclaim()` something to do. Incremental does not
  guarantee reclamation; if the heap keeps growing despite the sync, one
  rank will still hit its full-GC threshold organically.
- **Fix the root cause in parallel.** Per-20-step GC means you're
  allocating fast — hundreds of MB/s range. Synchronized GC masks the
  latency symptom but doesn't reduce the wall-clock cost; every sync
  still stops everyone. Shrinking allocations pushes the organic interval
  out and makes a long N actually viable.
- **Sanity check it is GC.** A ~20-step straggler cadence is also
  consistent with checkpoint writes, NCCL handshakes, or MPI progress
  threads. `Base.gc_num()` deltas in the profile are the definitive
  signal. Don't build the callback against the wrong root cause.

## How to port to Oceananigans

Wire as a `Callback(gc_func, IterationInterval(N))`, gated on
`arch isa Distributed`. Oceananigans' `Callback` / `IterationInterval` is the
direct analog of ClimaAtmos's `call_every_n_steps`. No Oceananigans-side
changes needed — purely a user-driver-level hook.

## What was checked

- Local clones: `ClimaAtmos`, `ClimaCore`, `ClimaTimeSteppers`, `ClimaComms`
  in `$JULIA_DEPOT/dev`.
- `gh search code` across CliMA and NumericalEarth orgs for `GC.gc`,
  `GC.enable`, `MPI.Barrier` combinations.
- Julia Discourse — no dedicated thread on this pattern; adjacent threads are
  about GC internals.

## What else was found (not the pattern, but related)

- `release_gpu!` in
  `NumericalEarth/BreezyBaroclinicInstability.jl` (`experiments/single_gpu_cascade.jl`)
  uses `GC.gc(true); GC.gc(false); GC.gc(true); CUDA.reclaim()` — useful for
  **fully releasing a GPU-resident model between experiments**, not for
  in-loop sync.
- `MPI.Barrier`s across NumericalEarth are for phase boundaries (post-IC,
  post-build, pre-output), never paired with GC.
- Other CliMA repos (`ClimaCoupler`, `ClimaOcean`, `ClimaCore`,
  CliMA/Oceananigans) only call `GC.gc` at setup or in tests.

## Bottom line

The real long-term fix is **fewer allocations per step**. Periodic collective
GC hides the stragglers but still costs wall time on every rank. Start with
the ClimaAtmos callback as-is (N=1000, `GC.gc(false)`, no `GC.enable(false)`),
measure straggler reduction, then work on the allocation sources
(Adapt-per-kernel-launch, OffsetArray views in diagnostics, per-step closures)
that drive heap pressure in the first place.
