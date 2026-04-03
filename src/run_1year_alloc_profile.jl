"""
Profile CPU-side memory allocations during the simulation timestep loop.

Uses Profile.Allocs (Julia 1.8+) to sample allocations during a short
post-warmup run.  Results are saved as a PProf protobuf file (.pb.gz)
for flamegraph analysis.

Environment variables (same as run_1year_benchmark.jl, plus):
  NWARMUP_STEPS       – warmup steps for JIT (default: 3)
  ALLOC_PROFILE_STEPS – steps to profile (default: 3)
  ALLOC_SAMPLE_RATE   – fraction of allocations to sample (default: 0.01)
"""

include("setup_model.jl")
include("setup_simulation.jl")

using Profile
using PProf
using Oceananigans.Simulations: reset!

NWARMUP_STEPS = parse(Int, get(ENV, "NWARMUP_STEPS", "3"))
ALLOC_PROFILE_STEPS = parse(Int, get(ENV, "ALLOC_PROFILE_STEPS", "3"))
ALLOC_SAMPLE_RATE = parse(Float64, get(ENV, "ALLOC_SAMPLE_RATE", "0.01"))

# Lightweight progress callback
alloc_progress(sim) = @info @sprintf(
    "  iter: %d, time: %.3f yr, wall: %.1f seconds",
    iteration(sim), time(sim) / year, sim.run_wall_time
)
add_callback!(simulation, alloc_progress, TimeInterval(month))

################################################################################
# Warmup run (JIT compilation)
################################################################################

warmup_time = NWARMUP_STEPS * Δt
@info "Warmup: $NWARMUP_STEPS timestep(s) to trigger JIT (warmup_time=$(warmup_time)s)"
flush(stdout); flush(stderr)

simulation.stop_time = warmup_time
run!(simulation)

@info "Warmup complete — resetting for allocation profiling"
flush(stdout); flush(stderr)

################################################################################
# Profile allocations
################################################################################

GC.gc()
gc_before = Base.gc_num()

reset!(simulation)
set!(model, age = Returns(0.0))
profile_time = ALLOC_PROFILE_STEPS * Δt
simulation.stop_time = profile_time

@info "Profiling allocations" steps = ALLOC_PROFILE_STEPS sample_rate = ALLOC_SAMPLE_RATE
flush(stdout); flush(stderr)

Profile.Allocs.clear()
t_start = Base.time()
Profile.Allocs.@profile sample_rate = ALLOC_SAMPLE_RATE run!(simulation)
t_elapsed = Base.time() - t_start

gc_after = Base.gc_num()

@info "Profiling complete" elapsed_seconds = round(t_elapsed; digits = 1)
@info "GC summary" allocated_bytes = gc_after.allocd - gc_before.allocd alloc_count = gc_after.malloc - gc_before.malloc gc_pause_ns = gc_after.total_time - gc_before.total_time
flush(stdout); flush(stderr)

################################################################################
# Save results
################################################################################

results = Profile.Allocs.fetch()
@info "Fetched allocation profile" n_samples = length(results.allocs)

output_dir = get(ENV, "ALLOC_PROFILE_DIR", "logs/alloc_profiles")
mkpath(output_dir)
job_id = get(ENV, "PBS_JOBID", "interactive")

rank_suffix = arch isa Distributed ? "_rank$(arch.local_rank)" : ""
pprof_file = joinpath(output_dir, "alloc_profile_$(job_id)$(rank_suffix).pb.gz")
PProf.Allocs.pprof(results; out = pprof_file, web = false)
@info "PProf protobuf saved" file = pprof_file

@info "run_1year_alloc_profile.jl complete"
flush(stdout); flush(stderr)
