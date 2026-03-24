"""
Run the model for 1 year from the converged periodic steady-state age,
saving half-monthly snapshots (25 total) for diagnostics and animations.

This script:
1. Loads the converged periodic age from the NK solver output
2. Sets it as the initial condition
3. Runs the model for 1 year with half-monthly JLD2 output (25 snapshots)
4. Saves to outputs/{PM}/periodic/{MC}/1year/{solver_tag}/

Usage — interactive:
```
qsub -I -P y99 -l mem=256GB -q gpuhopper -l walltime=02:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/run_periodic_1year.jl")
```

Environment variables (in addition to setup_model.jl):
  LINEAR_SOLVER  – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY – yes | no  (default: no)
"""

include("setup_model.jl")
include("setup_simulation.jl")

################################################################################
# Configuration — locate the converged NK solution
################################################################################

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) || error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"
solver_tag = "$(LINEAR_SOLVER)_$(lumpspray_tag)"

nk_output_dir = joinpath(outputdir, "periodic", model_config, "NK")
nk_file = joinpath(nk_output_dir, "age_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2")
isfile(nk_file) || error("Converged NK solution not found: $nk_file")

@info "run_periodic_1year.jl configuration"
@info "- LINEAR_SOLVER  = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY = $LUMP_AND_SPRAY (tag: $lumpspray_tag)"
@info "- NK solution    = $nk_file"
flush(stdout); flush(stderr)

################################################################################
# Load converged age and set as initial condition
################################################################################

@info "Loading converged periodic age from $nk_file"
flush(stdout); flush(stderr)

nk_data = load(nk_file)
age_steady_3D = nk_data["age"]  # (Nx', Ny', Nz') in seconds

@info "Overriding age initial condition from converged periodic state"
flush(stdout); flush(stderr)

set!(model.tracers.age, age_steady_3D)

################################################################################
# Output writers
################################################################################

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

# Output directory: outputs/{PM}/periodic/{MC}/1year/{solver_tag}/
periodic_1year_dir = joinpath(outputdir, "periodic", model_config, "1year", solver_tag)
mkpath(periodic_1year_dir)

output_fields = Dict(
    "age" => model.tracers.age,
)

output_prefix = joinpath(periodic_1year_dir, "age_periodic_1year")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(prescribed_Δt / 2),  # half-monthly snapshots (25 total)
    filename = output_prefix,
    overwrite_existing = true,
)

@info "Running 1-year simulation from converged periodic state"
@info "Output prefix: $output_prefix"
@info "Output directory: $periodic_1year_dir"
flush(stdout); flush(stderr)

run!(simulation)

@info "1-year periodic simulation complete"
flush(stdout); flush(stderr)

################################################################################
# Summary statistics
################################################################################

age_data = Array(interior(model.tracers.age))
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
age_wet = age_data[idx]

max_age_val = maximum(age_wet)
mean_age_val = mean(age_wet)

@info "Final age statistics:" max_age_years = max_age_val / year mean_age_years = mean_age_val / year n_wet_cells = Nidx
flush(stdout); flush(stderr)

@info "run_periodic_1year.jl complete"
@info "Run plot_periodic_1year_age.jl on CPU to generate diagnostic plots"
flush(stdout); flush(stderr)
