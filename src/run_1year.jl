"""
Run a standalone 1-year offline age simulation.

This is a lightweight test/debug script that runs the model for one year
and saves full output (age, u, v, w, η).

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/run_1year.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

include("setup_model.jl")

################################################################################
# Initial condition
################################################################################

@info "Setting initial condition: age = 0"
flush(stdout); flush(stderr)

set!(model, age = Returns(0.0))

# Initialize T and S from first month of FTS (if GM-Redi enabled)
if GM_REDI
    @info "Initializing T and S from FieldTimeSeries at t=0"
    set!(model.tracers.T, T_ts[1])
    set!(model.tracers.S, S_ts[1])
end

################################################################################
# Simulation
################################################################################

simulation, age_output_dir = setup_age_simulation(
    model, Δt, stop_time, outputdir, model_config, "1year";
    output_interval = prescribed_Δt,
    progress_interval = prescribed_Δt,
)

# Register callbacks for prescribed fields
if GM_REDI
    add_callback!(simulation, prescribe_TS!, IterationInterval(1))
    @info "Registered T/S prescribing callback (every iteration)"
end
if MONTHLY_KAPPAV
    add_callback!(simulation, update_κV!, IterationInterval(1))
    @info "Registered κV update callback (every iteration)"
end

@info "Running 1-year simulation"
flush(stdout); flush(stderr)

run!(simulation)

@info "1-year simulation complete"
flush(stdout); flush(stderr)

################################################################################
# Validate age field
################################################################################

if !(arch isa Distributed)
    validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label = "1-year")
end

@info "run_1year.jl complete"
@info "Run plot_1year_age.jl on CPU to generate age diagnostic plots"
flush(stdout); flush(stderr)
