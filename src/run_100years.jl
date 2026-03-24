"""
Run a standalone 100-year offline age simulation.

This script runs the model for one hundred years
and saves full output (age, u, v, w, η) every 10 years.

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=03:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/run_100years.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

include("setup_model.jl")

# Override stop_time for 100-year simulation
stop_time = 100 * 12 * prescribed_Δt

@info "Overriding stop_time for 100-year simulation: $(stop_time / year) years"
flush(stdout); flush(stderr)

include("setup_simulation.jl")

################################################################################
# Output writers
################################################################################

age_output_dir = setup_age_simulation(
    simulation, outputdir, model_config, "100years";
    output_interval = 10 * 12 * prescribed_Δt,
    progress_interval = 10 * 12 * prescribed_Δt,
)

@info "Running 100-year simulation"
flush(stdout); flush(stderr)

run!(simulation)

@info "100-year simulation complete"
flush(stdout); flush(stderr)

################################################################################
# Validate age field
################################################################################

if !(arch isa Distributed)
    validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label = "100-year")
end

@info "run_100years.jl complete"
@info "Run plot_standardrun_age.jl with DURATION=100years on CPU to generate age diagnostic plots"
flush(stdout); flush(stderr)
