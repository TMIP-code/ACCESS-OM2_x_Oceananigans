"""
Benchmark the 1-year forward-map walltime (no output writing).

Runs the model for NWARMUP_STEPS timesteps to trigger JIT compilation,
resets, then times a clean 1-year simulation with no output writers.
This isolates the "hot loop" cost as used in the NK solver's Φ!.

Environment variables (same as run_1year.jl):
  PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER
  NWARMUP_STEPS – number of timesteps for JIT warmup (default: 3)
  BENCHMARK_STEPS – if set, override stop_time to BENCHMARK_STEPS × Δt (e.g., 20 for profiling)
  SYNC_GC_NSTEPS – if > 0 and arch isa Distributed, install a synchronized
                   GC.gc(false) callback every N iterations (see docs/DISTRIBUTED_GC.md)
  TBLOCKING – "no" (default) runs the standard Simulation/run! loop.
              Integer K ≥ 2 enables K-sub-step temporal blocking: K tracer
              sub-steps between MPI halo exchanges. Requires halos ≥ K+1,
              ADVECTION_SCHEME=centered2, TIMESTEPPER=AB2.
"""

include("setup_model.jl")
include("setup_simulation.jl")

using Oceananigans.Simulations: reset!

NWARMUP_STEPS = parse(Int, get(ENV, "NWARMUP_STEPS", "3"))

# Allow overriding stop_time via BENCHMARK_STEPS (e.g., BENCHMARK_STEPS=20 for profiling)
BENCHMARK_STEPS_STR = get(ENV, "BENCHMARK_STEPS", "")
if !isempty(BENCHMARK_STEPS_STR)
    benchmark_steps = parse(Int, BENCHMARK_STEPS_STR)
    stop_time = benchmark_steps * Δt
    @info "BENCHMARK_STEPS=$benchmark_steps: overriding stop_time to $(stop_time / year) yr ($benchmark_steps steps)"
end

################################################################################
# Temporal blocking setup (opt-in via TBLOCKING=<K>)
################################################################################

TBLOCKING_STR = lowercase(get(ENV, "TBLOCKING", "no"))
TBLOCKING = (TBLOCKING_STR == "no" || isempty(TBLOCKING_STR)) ? 0 : parse(Int, TBLOCKING_STR)

if TBLOCKING > 0
    @assert ADVECTION_SCHEME == "centered2" "TBLOCKING requires ADVECTION_SCHEME=centered2 (B=1 stencil)"
    @assert TIMESTEPPER == "AB2" "TBLOCKING requires TIMESTEPPER=AB2"
    GM_REDI && @warn "TBLOCKING + GM_REDI: T,S update path is untested — verify before trusting results"
    VELOCITY_SOURCE == "totaltransport" || @warn "TBLOCKING + VELOCITY_SOURCE=$VELOCITY_SOURCE: no GM in u,v — consider VELOCITY_SOURCE=totaltransport"
    Hx = grid.underlying_grid.Hx
    Hy = grid.underlying_grid.Hy
    @assert Hx ≥ TBLOCKING + 1 && Hy ≥ TBLOCKING + 1 "TBLOCKING=$TBLOCKING requires grid halos ≥ $(TBLOCKING + 1); got (Hx=$Hx, Hy=$Hy). Rebuild grid with GRID_HX/HY=$(TBLOCKING + 1)."

    total_steps = round(Int, stop_time / Δt)
    @assert total_steps % TBLOCKING == 0 "TBLOCKING=$TBLOCKING must divide total_steps=$total_steps"

    NWARMUP_STEPS = TBLOCKING  # warmup = one batch so JIT covers multi_time_step!

    include("temporal_blocking.jl")

    Nx_local, Ny_local, Nz_local = size(grid)

    # Per-sub-step updates for Field-backed prescribed quantities whose
    # Simulation callbacks won't fire inside the batch. η is intentionally
    # NOT in this list: PrescribedFreeSurface wraps η in a TimeSeriesInterpolation
    # (not a raw Field), and refreshing its clock without also refreshing
    # diagnosed w and z-scaling would be inconsistent. η, w, and z-scaling
    # all re-sync once per batch via update_state! — acceptable because
    # the FTS varies on monthly timescales and K·Δt ≪ 1 month.
    fts_update_list = Tuple{Any, Any}[]
    MONTHLY_KAPPAV && push!(fts_update_list, (κVField, κV_ts))
    GM_REDI && append!(fts_update_list, [(model.tracers.T, T_ts), (model.tracers.S, S_ts)])
    FTS_UPDATES = Tuple(fts_update_list)

    @info "Temporal blocking ENABLED: K=$TBLOCKING sub-steps per batch; halos=($Hx, $Hy); fts_updates=$(length(FTS_UPDATES))"
end

# Lightweight progress callback: iteration, time, wall time only (no age stats)
benchmark_progress(sim) = @info @sprintf(
    "  iter: %d, time: %.3f yr, wall: %.1f seconds",
    iteration(sim), time(sim) / year, sim.run_wall_time
)
TBLOCKING == 0 && add_callback!(simulation, benchmark_progress, TimeInterval(month))

################################################################################
# Warmup run (JIT compilation)
################################################################################

warmup_time = NWARMUP_STEPS * Δt
@info "Warmup: $NWARMUP_STEPS timestep(s) to trigger JIT (warmup_time=$(warmup_time)s)"
flush(stdout); flush(stderr)

if TBLOCKING == 0
    simulation.stop_time = warmup_time
    run!(simulation)
else
    multi_time_step!(model, Δt, Nx_local, Ny_local, Nz_local; K = TBLOCKING, fts_updates = FTS_UPDATES)
end

@info "Warmup complete — resetting for benchmark"
flush(stdout); flush(stderr)

################################################################################
# Benchmark: 1-year run, no output writers
################################################################################

reset!(simulation)
set!(model, age = Returns(0.0))

# Synchronized GC for distributed runs (see docs/DISTRIBUTED_GC.md). Implicit barrier
# via the surrounding collectives; off by default. (Callback path only — TBLOCKING
# bypasses Simulation callbacks.)
SYNC_GC_NSTEPS_STR = get(ENV, "SYNC_GC_NSTEPS", "0")
SYNC_GC_NSTEPS = isempty(SYNC_GC_NSTEPS_STR) ? 0 : parse(Int, SYNC_GC_NSTEPS_STR)
if TBLOCKING == 0 && arch isa Distributed && SYNC_GC_NSTEPS > 0
    sync_gc!(sim) = (GC.gc(false); nothing)
    add_callback!(simulation, sync_gc!, IterationInterval(SYNC_GC_NSTEPS))
    @info "Synchronized GC enabled: GC.gc(false) every $SYNC_GC_NSTEPS iterations"
end

@info "Benchmark: 1-year simulation, no output writers (stop_time=$(stop_time / year) yr)"
flush(stdout); flush(stderr)

t_start = Base.time()
CUDA.@profile external = true begin
    if TBLOCKING == 0
        simulation.stop_time = stop_time
        run!(simulation)
    else
        n_batches = total_steps ÷ TBLOCKING
        @info "Temporal-blocked loop: $n_batches batches × K=$TBLOCKING sub-steps = $total_steps total steps"
        flush(stdout); flush(stderr)
        for batch in 1:n_batches
            multi_time_step!(model, Δt, Nx_local, Ny_local, Nz_local; K = TBLOCKING, fts_updates = FTS_UPDATES)
            if batch % max(1, n_batches ÷ 12) == 0
                @info @sprintf(
                    "  batch: %d/%d, time: %.3f yr, wall: %.1f seconds",
                    batch, n_batches, model.clock.time / year, Base.time() - t_start
                )
                flush(stdout); flush(stderr)
            end
        end
    end
end
t_elapsed = Base.time() - t_start

@info "Benchmark complete" elapsed_seconds = round(t_elapsed; digits = 1) elapsed_minutes = round(t_elapsed / 60; digits = 2)

@info "run_1year_benchmark.jl complete"
flush(stdout); flush(stderr)
