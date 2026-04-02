"""
Benchmark the 1-year forward-map walltime (no output writing).

Runs the model for NWARMUP_STEPS timesteps to trigger JIT compilation,
resets, then times a clean 1-year simulation with no output writers.
This isolates the "hot loop" cost as used in the NK solver's Φ!.

Environment variables (same as run_1year.jl):
  PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER
  NWARMUP_STEPS – number of timesteps for JIT warmup (default: 3)
  BENCHMARK_STEPS – if set, override stop_time to BENCHMARK_STEPS × Δt (e.g., 20 for profiling)
"""

include("setup_model.jl")
include("setup_simulation.jl")

using Oceananigans.Simulations: reset!

NWARMUP_STEPS = parse(Int, get(ENV, "NWARMUP_STEPS", "3"))

# Allow overriding stop_time via BENCHMARK_STEPS (e.g., BENCHMARK_STEPS=20 for profiling)
BENCHMARK_STEPS_STR = get(ENV, "BENCHMARK_STEPS", nothing)
if BENCHMARK_STEPS_STR !== nothing
    benchmark_steps = parse(Int, BENCHMARK_STEPS_STR)
    stop_time = benchmark_steps * Δt
    @info "BENCHMARK_STEPS=$benchmark_steps: overriding stop_time to $(stop_time / year) yr ($benchmark_steps steps)"
end

# Lightweight progress callback: iteration, time, wall time only (no age stats)
benchmark_progress(sim) = @info @sprintf(
    "  iter: %d, time: %.3f yr, wall: %.1f seconds",
    iteration(sim), time(sim) / year, sim.run_wall_time
)
add_callback!(simulation, benchmark_progress, TimeInterval(month))

################################################################################
# Warmup run (JIT compilation)
################################################################################

warmup_time = NWARMUP_STEPS * Δt
@info "Warmup: $NWARMUP_STEPS timestep(s) to trigger JIT (warmup_time=$(warmup_time)s)"
flush(stdout); flush(stderr)

simulation.stop_time = warmup_time
run!(simulation)

@info "Warmup complete — resetting for benchmark"
flush(stdout); flush(stderr)

################################################################################
# Benchmark: 1-year run, no output writers
################################################################################

reset!(simulation)
set!(model, age = Returns(0.0))
simulation.stop_time = stop_time

@info "Benchmark: 1-year simulation, no output writers (stop_time=$(stop_time / year) yr)"
flush(stdout); flush(stderr)

t_start = Base.time()
run!(simulation)
t_elapsed = Base.time() - t_start

@info "Benchmark complete" elapsed_seconds = round(t_elapsed; digits = 1) elapsed_minutes = round(t_elapsed / 60; digits = 2)

@info "run_1year_benchmark.jl complete"
flush(stdout); flush(stderr)
