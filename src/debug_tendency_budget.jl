"""
Debug tendency budget at the max-age hotspot.

Rebuilds the model from setup_model.jl, loads the saved age field from the
1-year simulation, and evaluates each of the 5 tendency terms individually
at the hotspot (i=224, j=40, k=28).

Usage — CPU job or interactive:
```
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
export VELOCITY_SOURCE=cgridtransports W_FORMULATION=wdiagnosed ADVECTION_SCHEME=weno3
julia --project src/debug_tendency_budget.jl
```
"""

# ── Step 1: Rebuild the model via setup_model.jl ─────────────────────────

include("setup_model.jl")

# ── Step 2: Import Oceananigans internal tendency-term functions ──────────

using Oceananigans.Advection: div_Uc
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, closure_auxiliary_velocity
using Oceananigans.Biogeochemistry: biogeochemical_transition,
    biogeochemical_drift_velocity,
    biogeochemical_auxiliary_fields
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Utils: sum_of_velocities
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_fields,
    initialize_vertical_coordinate!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Fields: immersed_boundary_condition

# ── Step 3: Load saved age field from JLD2 output ────────────────────────

@info "Loading saved age field from 1-year simulation output"
flush(stdout)

age_output_dir = joinpath(outputdir, "age", run_mode_tag)
output_file = joinpath(age_output_dir, "age_1year_$(ADVECTION_SCHEME).jld2")
isfile(output_file) || error("Output file not found: $output_file")

# Read the last timestep
local age_data, final_time
jldopen(output_file) do f
    # Find the last iteration
    t_keys = sort(parse.(Int, filter(k -> all(isdigit, k), collect(keys(f["timeseries/t"])))))
    last_iter = t_keys[end]
    final_time = f["timeseries/t/$last_iter"]
    age_data = f["timeseries/age/$last_iter"]
    @info "Loaded age at iteration $last_iter, t = $(final_time / year) years"
end
flush(stdout)

# ── Step 4: Set model state to match end of simulation ───────────────────

@info "Setting model state to t = $(final_time / year) years"
flush(stdout)

# Set clock to final time so FieldTimeSeries (velocities, η) interpolate correctly
model.clock.time = final_time

# Set the age tracer
set!(model, age = age_data)

# Initialize z-star scaling from η at the current clock time
initialize_vertical_coordinate!(model.vertical_coordinate, model, model.grid)

# Fill halos, compute diagnosed w, update closure fields, etc.
update_state!(model)

@info "Model state updated"
flush(stdout)

# ── Step 5: Extract components and construct intermediates ────────────────

grid = model.grid
c = model.tracers.age
c_advection = model.advection[:age]
closure = model.closure
closure_fields = model.closure_fields
buoyancy = model.buoyancy
biogeochemistry = model.biogeochemistry
velocities = model.velocities  # = transport_velocities for PrescribedVelocityFields
free_surface = model.free_surface
tracers = model.tracers
auxiliary_fields = model.auxiliary_fields
clock = model.clock
forcing_age = model.forcing[:age]
c_immersed_bc = immersed_boundary_condition(c)

# Construct model_fields (same as inside hydrostatic_free_surface_tracer_tendency)
model_fields = merge(
    hydrostatic_fields(velocities, free_surface, tracers),
    auxiliary_fields,
    biogeochemical_auxiliary_fields(biogeochemistry),
)

# Construct total_velocities (same as inside hydrostatic_free_surface_tracer_tendency)
biogeo_vels = biogeochemical_drift_velocity(biogeochemistry, Val(:age))
closure_vels = closure_auxiliary_velocity(closure, closure_fields, Val(:age))
total_velocities = sum_of_velocities(velocities, biogeo_vels, closure_vels)
total_velocities = with_advective_forcing(forcing_age, total_velocities)

# ── Step 6: Evaluate each tendency term at the hotspot ────────────────────

i, j, k = 224, 40, 28

@info "Computing tendency budget at (i=$i, j=$j, k=$k)"
flush(stdout)

# The 5 terms (signs match the return statement in hydrostatic_free_surface_tracer_tendency)
term_advection = -div_Uc(i, j, k, grid, c_advection, total_velocities, c)
term_diffusion = -∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, Val(1), c, clock, model_fields, buoyancy)
term_immersed = -immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, Val(1), clock, model_fields)
term_biogeo = biogeochemical_transition(i, j, k, grid, biogeochemistry, Val(:age), clock, model_fields)
term_forcing = forcing_age(i, j, k, grid, clock, model_fields)

total_tendency = term_advection + term_diffusion + term_immersed + term_biogeo + term_forcing

# ── Step 7: Verify against full tendency function ─────────────────────────

full_tendency = hydrostatic_free_surface_tracer_tendency(
    i, j, k, grid,
    Val(1), Val(:age),
    c_advection,
    closure,
    c_immersed_bc,
    buoyancy,
    biogeochemistry,
    velocities,
    free_surface,
    tracers,
    closure_fields,
    auxiliary_fields,
    clock,
    forcing_age,
)

# ── Step 8: Print results ─────────────────────────────────────────────────

z_centers = Array(znodes(grid, Center(), Center(), Center()))
age_val = @allowscalar c[i, j, k]

@info "═══════════════════════════════════════════════════════════════"
@info "Tendency budget at hotspot (i=$i, j=$j, k=$k) — z = $(z_centers[k]) m"
@info "═══════════════════════════════════════════════════════════════"
@info @sprintf("  age value          = %12.6e s  (%.4f yr)", age_val, age_val / year)
@info "───────────────────────────────────────────────────────────────"
@info @sprintf("  1. Advection       = %+12.6e s/s", term_advection)
@info @sprintf("  2. Diffusion       = %+12.6e s/s", term_diffusion)
@info @sprintf("  3. Immersed BC     = %+12.6e s/s", term_immersed)
@info @sprintf("  4. Biogeochemistry = %+12.6e s/s", term_biogeo)
@info @sprintf("  5. Forcing         = %+12.6e s/s", term_forcing)
@info "───────────────────────────────────────────────────────────────"
@info @sprintf("  Sum of terms       = %+12.6e s/s", total_tendency)
@info @sprintf("  Full tendency      = %+12.6e s/s", full_tendency)
@info @sprintf("  Difference         = %+12.6e s/s", total_tendency - full_tendency)
@info "───────────────────────────────────────────────────────────────"
@info @sprintf("  Net tendency sign: %s (age is %s)", total_tendency > 0 ? "POSITIVE" : "NEGATIVE", total_tendency > 0 ? "increasing" : "decreasing")
@info "═══════════════════════════════════════════════════════════════"

# Also print neighbor values for context
@info "Neighbor age values (yr):"
Nx′, Ny′, Nz′ = size(interior(c))
for dk in -1:1, dj in -1:1, di in -1:1
    ni, nj, nk = i + di, j + dj, k + dk
    if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′
        nval = @allowscalar c[ni, nj, nk]
        @info @sprintf("  (%+d,%+d,%+d) → age = %8.4f yr", di, dj, dk, nval / year)
    end
end

@info "debug_tendency_budget.jl complete"
flush(stdout)
