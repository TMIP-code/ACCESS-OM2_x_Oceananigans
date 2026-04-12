"""
Run a 10-step diagnostic simulation, saving age at every time step.

Used to isolate where serial vs distributed results diverge.
Output files: `age_diag_part1.jld2` ... `age_diag_part11.jld2`
  (part 1 = t=0, part 2 = t=Δt, ..., part 11 = t=10Δt)

Usage:
```
PARENT_MODEL=ACCESS-OM2-1 julia --project test/run_diagnostic_steps.jl
```
"""

include("../src/setup_model.jl")

# Override stop_time to 10 timesteps for the diagnostic run
stop_time = 10 * Δt

include("../src/setup_simulation.jl")

################################################################################
# Output writers (initial condition is set by setup_simulation.jl)
################################################################################

age_output_dir = setup_age_simulation(
    simulation, outputdir, model_config, "diag";
    output_interval = Δt,
    progress_interval = Δt,
)

@info "Running 10-step diagnostic simulation (stop_time = $(stop_time / year) yr)"
@info "Grid:"
show(stdout, MIME"text/plain"(), model.grid)
println(stdout)
@info "Model:"
show(stdout, MIME"text/plain"(), model)
println(stdout)
@info "Simulation:"
show(stdout, MIME"text/plain"(), simulation)
println(stdout)

# Print grid metric sizes/types for diagnostic comparison
ug = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid
@info "Grid metric diagnostics:"
for (name, f) in [
        ("λᶜᶜᵃ", ug.λᶜᶜᵃ), ("φᶜᶜᵃ", ug.φᶜᶜᵃ), ("z", ug.z),
        ("Δxᶜᶜᵃ", ug.Δxᶜᶜᵃ), ("Δyᶜᶜᵃ", ug.Δyᶜᶜᵃ), ("Azᶜᶜᵃ", ug.Azᶜᶜᵃ),
    ]
    @info "  $name: $(typeof(f)), size=$(size(f))"
end
if model.grid isa ImmersedBoundaryGrid
    ib = model.grid.immersed_boundary
    @info "  immersed_boundary: $(typeof(ib))"
    if hasproperty(ib, :bottom_height)
        @info "  bottom_height: $(typeof(ib.bottom_height)), size=$(size(ib.bottom_height))"
    end
end
flush(stdout); flush(stderr)

run!(simulation)

@info "Diagnostic simulation complete"
@info "Output saved to $age_output_dir"
flush(stdout); flush(stderr)

@info "run_diagnostic_steps.jl complete"
flush(stdout); flush(stderr)
