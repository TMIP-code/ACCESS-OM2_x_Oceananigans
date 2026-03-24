"""
Shared simulation setup: initializes tracers, creates a Simulation, and
registers prescribing callbacks for GM-Redi (T/S) and monthly κV.

This file is `include()`d after `setup_model.jl` by all downstream scripts.
It expects all variables from setup_model.jl to be in scope.

Produces: `simulation` (Simulation with prescribing callbacks registered)
"""

@info "Initializing tracers and creating simulation"
flush(stdout); flush(stderr)

################################################################################
# Initialize tracers
################################################################################

set!(model, age = Returns(0.0))

if GM_REDI
    @info "Initializing T and S from FieldTimeSeries (first snapshot)"
    set!(model.tracers.T, T_ts[1])
    set!(model.tracers.S, S_ts[1])
end

################################################################################
# Create simulation and register prescribing callbacks
################################################################################

simulation = Simulation(model; Δt, stop_time)

if GM_REDI
    add_callback!(simulation, prescribe_TS!, IterationInterval(1))
    @info "Registered T/S prescribing callback (every iteration)"
end
if MONTHLY_KAPPAV
    add_callback!(simulation, update_κV!, IterationInterval(1))
    @info "Registered κV update callback (every iteration)"
end

@info "setup_simulation.jl complete"
flush(stdout); flush(stderr)
