"""
Allocation benchmark via `CUDA.@time time_step!(model, Δt)`.

After model setup, runs a single warmup step (JIT compilation), then
times one step and a 20-step batch with `CUDA.@time` to report GPU
memory allocations alongside wallclock time.

Environment variables (same as run_1year_benchmark.jl):
  PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER
  ALLOC_BATCH_STEPS – number of steps in the batch measurement (default: 20)
"""

include("setup_model.jl")
include("setup_simulation.jl")

using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Fields: instantiated_location
using Oceananigans.Models: fields
import Oceananigans.Architectures

ALLOC_BATCH_STEPS = parse(Int, get(ENV, "ALLOC_BATCH_STEPS", "20"))

@info "Allocation benchmark: time_step!(model, Δt)"
@info "  Δt = $(Δt) s"
@info "  ALLOC_BATCH_STEPS = $ALLOC_BATCH_STEPS"
flush(stdout); flush(stderr)

################################################################################
# Warmup (compilation)
################################################################################

@info "Warmup: 1 timestep to trigger JIT"
flush(stdout); flush(stderr)
time_step!(model, Δt)

@info "Warmup complete"
flush(stdout); flush(stderr)

################################################################################
# Measure a single step
################################################################################

@info "Measuring single time_step! with CUDA.@time"
flush(stdout); flush(stderr)
CUDA.@time time_step!(model, Δt)
flush(stdout); flush(stderr)

################################################################################
# Measure a batch of steps
################################################################################

@info "Measuring $ALLOC_BATCH_STEPS time_step! calls with CUDA.@time"
flush(stdout); flush(stderr)
CUDA.@time for _ in 1:ALLOC_BATCH_STEPS
    time_step!(model, Δt)
end
flush(stdout); flush(stderr)

################################################################################
# convert_to_device probe (does TimeSeriesInterpolation allocate per call?)
################################################################################

@info "Probing convert_to_device(args) allocation"
flush(stdout); flush(stderr)

let tracer = first(values(model.tracers))
    args = (
        tracer.data, tracer.boundary_conditions, tracer.indices,
        instantiated_location(tracer), model.grid, model.clock, fields(model),
    )
    # First call (may include compilation)
    @time Oceananigans.Architectures.convert_to_device(model.architecture, args)
    # Second call (steady-state)
    @time Oceananigans.Architectures.convert_to_device(model.architecture, args)
    # Third call to confirm it's stable
    @time Oceananigans.Architectures.convert_to_device(model.architecture, args)
end
flush(stdout); flush(stderr)

@info "run_alloc_benchmark.jl complete"
flush(stdout); flush(stderr)
