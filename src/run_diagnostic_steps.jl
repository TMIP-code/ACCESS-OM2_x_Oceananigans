"""
Run a 10-step diagnostic simulation, saving age at every time step.

Used to isolate where serial vs distributed results diverge.
Output files: `age_diag_part1.jld2` ... `age_diag_part11.jld2`
  (part 1 = t=0, part 2 = t=Δt, ..., part 11 = t=10Δt)

Usage:
```
PARENT_MODEL=ACCESS-OM2-1 julia --project src/run_diagnostic_steps.jl
```
"""

include("setup_model.jl")

################################################################################
# Initial condition
################################################################################

@info "Setting initial condition: age = 0"
flush(stdout); flush(stderr)

set!(model, age = Returns(0.0))

################################################################################
# Simulation (10 time steps, output every step)
################################################################################

diag_stop_time = 10 * Δt

simulation, age_output_dir = setup_age_simulation(
    model, Δt, diag_stop_time, outputdir, model_config, "diag";
    output_interval = Δt,
    progress_interval = Δt,
)

@info "Running 10-step diagnostic simulation (stop_time = $(diag_stop_time / year) yr)"
flush(stdout); flush(stderr)

run!(simulation)

@info "Diagnostic simulation complete"
@info "Output saved to $age_output_dir"
flush(stdout); flush(stderr)

@info "run_diagnostic_steps.jl complete"
flush(stdout); flush(stderr)
