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
  VELOCITY_SOURCE  – cgridtransports | totaltransport (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

include("setup_model.jl")
include("setup_simulation.jl")

################################################################################
# Output writers
################################################################################

age_output_dir = setup_age_simulation(
    simulation, outputdir, model_config, "1year" * noACM_suffix();
    output_interval = prescribed_Δt,
    progress_interval = prescribed_Δt,
)

@info "Running 1-year simulation"
flush(stdout); flush(stderr)

run!(simulation)

@info "1-year simulation complete"
flush(stdout); flush(stderr)

################################################################################
# 1-year drift residual (per-rank) — mirrors G! in periodic_solver_common.jl
# so values are directly comparable to NK diagnostic output. For
# INITIAL_AGE=0, drift = age_final - 0 = age_final.
################################################################################

using MPI

INITIAL_AGE_LOCAL = get(ENV, "INITIAL_AGE", "0")
if INITIAL_AGE_LOCAL == "0"
    (; idx) = compute_wet_mask(grid)
    v1D = Array(interior(compute_volume(grid)))[idx]
    vol_norm_local = make_vol_norm(v1D, year)
    age_final_3D = Array(interior(model.tracers.age))
    drift_1D = age_final_3D[idx]

    vol_rms_drift_years = vol_norm_local(drift_1D)
    max_drift_years = maximum(abs, drift_1D) / year
    mean_drift_years = sum(abs, drift_1D) / length(drift_1D) / year

    rank_label = arch isa Distributed ?
        "rank " * string(MPI.Comm_rank(MPI.COMM_WORLD)) :
        "serial"
    @info "1-year drift residual ($rank_label, per-rank)" vol_rms_drift_years max_drift_years mean_drift_years Nidx_local = length(drift_1D)
    flush(stdout); flush(stderr)
else
    @info "Skipping drift residual print: INITIAL_AGE=$INITIAL_AGE_LOCAL ≠ 0 (drift = Φ(x_init) − x_init not yet supported here)"
    flush(stdout); flush(stderr)
end

################################################################################
# Validate age field
################################################################################

if !(arch isa Distributed)
    validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label = "1-year")
end

@info "run_1year.jl complete"
@info "Run plot_1year_age.jl on CPU to generate age diagnostic plots"
flush(stdout); flush(stderr)
