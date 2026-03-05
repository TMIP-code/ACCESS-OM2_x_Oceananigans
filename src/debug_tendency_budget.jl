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

using Oceananigans.Advection: div_Uc,
    _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
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
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, Vᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell

# ── Step 2b: Tendency check at t=0 (age=0 everywhere) ────────────────────

@info "╔═══════════════════════════════════════════════════════════════╗"
@info "║  Tendency check at t=0 (age = 0 everywhere)                 ║"
@info "╚═══════════════════════════════════════════════════════════════╝"
flush(stdout); flush(stderr)

# Model starts at t=0 with age=0 by default after setup_model.jl
# Just need to initialize vertical coordinate and update state
initialize_vertical_coordinate!(model.vertical_coordinate, model, model.grid)
update_state!(model)

@info "Model state updated at t=0"
flush(stdout); flush(stderr)

# Extract components at t=0
begin
    local grid = model.grid
    local c = model.tracers.age
    local c_advection = model.advection[:age]
    local closure = model.closure
    local closure_fields = model.closure_fields
    local buoyancy = model.buoyancy
    local biogeochemistry = model.biogeochemistry
    local velocities = model.velocities
    local free_surface = model.free_surface
    local tracers = model.tracers
    local auxiliary_fields = model.auxiliary_fields
    local clock = model.clock
    local forcing_age = model.forcing[:age]
    local c_immersed_bc = immersed_boundary_condition(c)

    local model_fields = merge(
        hydrostatic_fields(velocities, free_surface, tracers),
        auxiliary_fields,
        biogeochemical_auxiliary_fields(biogeochemistry),
    )

    local biogeo_vels = biogeochemical_drift_velocity(biogeochemistry, Val(:age))
    local closure_vels = closure_auxiliary_velocity(closure, closure_fields, Val(:age))
    local total_velocities = sum_of_velocities(velocities, biogeo_vels, closure_vels)
    total_velocities = with_advective_forcing(forcing_age, total_velocities)

    local i, j, k = 224, 40, 28

    @info "Computing tendency budget at (i=$i, j=$j, k=$k) at t=0"
    flush(stdout); flush(stderr)

    local term_advection = -div_Uc(i, j, k, grid, c_advection, total_velocities, c)
    local term_diffusion = -∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, Val(1), c, clock, model_fields, buoyancy)
    local term_immersed = -immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, Val(1), clock, model_fields)
    local term_biogeo = biogeochemical_transition(i, j, k, grid, biogeochemistry, Val(:age), clock, model_fields)
    local term_forcing = forcing_age(i, j, k, grid, clock, model_fields)

    local total_tendency = term_advection + term_diffusion + term_immersed + term_biogeo + term_forcing

    local full_tendency = hydrostatic_free_surface_tracer_tendency(
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

    local z_centers = Array(znodes(grid, Center(), Center(), Center()))
    local age_val = @allowscalar c[i, j, k]

    @info "═══════════════════════════════════════════════════════════════"
    @info "Tendency budget at t=0 — hotspot (i=$i, j=$j, k=$k) — z = $(z_centers[k]) m"
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
    @info @sprintf("  ⚠ Advection at t=0 should be 0 (age=0 everywhere). Actual: %+.6e", term_advection)
    @info "═══════════════════════════════════════════════════════════════"

    # Print neighbor values at t=0
    @info "Neighbor age values at t=0 (yr):"
    local Nx′, Ny′, Nz′ = size(interior(c))
    for dk in -1:1, dj in -1:1, di in -1:1
        ni, nj, nk = i + di, j + dj, k + dk
        if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′
            nval = @allowscalar c[ni, nj, nk]
            is_immersed = immersed_cell(ni, nj, nk, grid)
            is_partial = !is_immersed && any(
                immersed_cell(ni + di′, nj + dj′, nk + dk′, grid)
                    for (di′, dj′, dk′) in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
            )
            tag = is_immersed ? " [immersed]" : is_partial ? " [partial]" : ""
            @info @sprintf("  (%+d,%+d,%+d) → age = %8.4f yr", di, dj, dk, nval / year) * tag
        end
    end
end

flush(stdout); flush(stderr)

# ── Step 2c: Tendency check at t=0 (age=1 everywhere) ────────────────────

@info "╔═══════════════════════════════════════════════════════════════╗"
@info "║  Tendency check at t=0 (age = 1 everywhere)                 ║"
@info "╚═══════════════════════════════════════════════════════════════╝"
flush(stdout); flush(stderr)

# Set age=1 everywhere, keep clock at t=0
set!(model, age = 1year)
update_state!(model)

@info "Model state updated at t=0 with age=1"
flush(stdout); flush(stderr)

# Extract components at t=0, age=1
begin
    local grid = model.grid
    local c = model.tracers.age
    local c_advection = model.advection[:age]
    local closure = model.closure
    local closure_fields = model.closure_fields
    local buoyancy = model.buoyancy
    local biogeochemistry = model.biogeochemistry
    local velocities = model.velocities
    local free_surface = model.free_surface
    local tracers = model.tracers
    local auxiliary_fields = model.auxiliary_fields
    local clock = model.clock
    local forcing_age = model.forcing[:age]
    local c_immersed_bc = immersed_boundary_condition(c)

    local model_fields = merge(
        hydrostatic_fields(velocities, free_surface, tracers),
        auxiliary_fields,
        biogeochemical_auxiliary_fields(biogeochemistry),
    )

    local biogeo_vels = biogeochemical_drift_velocity(biogeochemistry, Val(:age))
    local closure_vels = closure_auxiliary_velocity(closure, closure_fields, Val(:age))
    local total_velocities = sum_of_velocities(velocities, biogeo_vels, closure_vels)
    total_velocities = with_advective_forcing(forcing_age, total_velocities)

    local i, j, k = 224, 40, 28

    @info "Computing tendency budget at (i=$i, j=$j, k=$k) at t=0, age=1"
    flush(stdout); flush(stderr)

    local term_advection = -div_Uc(i, j, k, grid, c_advection, total_velocities, c)
    local term_diffusion = -∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, Val(1), c, clock, model_fields, buoyancy)
    local term_immersed = -immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, Val(1), clock, model_fields)
    local term_biogeo = biogeochemical_transition(i, j, k, grid, biogeochemistry, Val(:age), clock, model_fields)
    local term_forcing = forcing_age(i, j, k, grid, clock, model_fields)

    local total_tendency = term_advection + term_diffusion + term_immersed + term_biogeo + term_forcing

    local full_tendency = hydrostatic_free_surface_tracer_tendency(
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

    local z_centers = Array(znodes(grid, Center(), Center(), Center()))
    local age_val = @allowscalar c[i, j, k]

    @info "═══════════════════════════════════════════════════════════════"
    @info "Tendency budget at t=0, age=1 — hotspot (i=$i, j=$j, k=$k) — z = $(z_centers[k]) m"
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
    @info @sprintf("  ⚠ Advection at t=0 should be 0 (uniform age). Actual: %+.6e", term_advection)
    @info "═══════════════════════════════════════════════════════════════"

    # Print neighbor values at t=0, age=1
    @info "Neighbor age values at t=0, age=1 (yr):"
    local Nx′, Ny′, Nz′ = size(interior(c))
    for dk in -1:1, dj in -1:1, di in -1:1
        ni, nj, nk = i + di, j + dj, k + dk
        if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′
            nval = @allowscalar c[ni, nj, nk]
            is_immersed = immersed_cell(ni, nj, nk, grid)
            is_partial = !is_immersed && any(
                immersed_cell(ni + di′, nj + dj′, nk + dk′, grid)
                    for (di′, dj′, dk′) in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
            )
            tag = is_immersed ? " [immersed]" : is_partial ? " [partial]" : ""
            @info @sprintf("  (%+d,%+d,%+d) → age = %8.4f yr", di, dj, dk, nval / year) * tag
        end
    end
end

flush(stdout); flush(stderr)

# Reset age back to 0 before loading saved field
set!(model, age = 0)

# ── Step 3: Load saved age field from JLD2 output ────────────────────────

@info "Loading saved age field from 1-year simulation output"
flush(stdout); flush(stderr)

age_output_dir = joinpath(outputdir, "age", model_config)
output_file = joinpath(age_output_dir, "age_1year.jld2")
isfile(output_file) || error("Output file not found: $output_file")

# Read the last timestep
age_lazy = FieldTimeSeries(output_file, "age")
age_times = age_lazy.times
final_time_index = length(age_times)
@info "Final time index in JLD2 output: $final_time_index"
final_time = age_times[final_time_index]
@info "Final time in JLD2 output: $(final_time / year) years"
age_final = age_lazy[final_time_index]
show(age_final)  # load data into memory

flush(stdout); flush(stderr)

# ── Step 4: Set model state to match end of simulation ───────────────────

@info "Setting model state to t = $(final_time / year) years"
flush(stdout); flush(stderr)

# Set clock to final time so FieldTimeSeries (velocities, η) interpolate correctly
model.clock.time = final_time

# Set the age tracer
set!(model, age = age_final)

# Initialize z-star scaling from η at the current clock time
initialize_vertical_coordinate!(model.vertical_coordinate, model, model.grid)

# Fill halos, compute diagnosed w, update closure fields, etc.
update_state!(model)

@info "Model state updated"
flush(stdout); flush(stderr)

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
flush(stdout); flush(stderr)

# The 5 terms (signs match the return statement in hydrostatic_free_surface_tracer_tendency)
term_advection = -div_Uc(i, j, k, grid, c_advection, total_velocities, c)

# Directional decomposition of advection
inv_V = 1 / Vᶜᶜᶜ(i, j, k, grid)
adv_x = -inv_V * δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, c_advection, total_velocities.u, c)
adv_y = -inv_V * δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, c_advection, total_velocities.v, c)
adv_z = -inv_V * δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, c_advection, total_velocities.w, c)

# Face velocities
u_west = @allowscalar total_velocities.u[i, j, k]
u_east = @allowscalar total_velocities.u[i + 1, j, k]
v_south = @allowscalar total_velocities.v[i, j, k]
v_north = @allowscalar total_velocities.v[i, j + 1, k]
w_bot = @allowscalar total_velocities.w[i, j, k]
w_top = @allowscalar total_velocities.w[i, j, k + 1]

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
@info @sprintf("     1a. Adv (x)     = %+12.6e s/s", adv_x)
@info @sprintf("     1b. Adv (y)     = %+12.6e s/s", adv_y)
@info @sprintf("     1c. Adv (z)     = %+12.6e s/s", adv_z)
@info @sprintf("     1d. Adv check   = %+12.6e s/s  (x+y+z)", adv_x + adv_y + adv_z)
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
@info "───────────────────────────────────────────────────────────────"
@info "  Face velocities at (i=$i, j=$j, k=$k):"
@info @sprintf("    u_west  (i  ) = %+12.6e m/s", u_west)
@info @sprintf("    u_east  (i+1) = %+12.6e m/s", u_east)
@info @sprintf("    v_south (j  ) = %+12.6e m/s", v_south)
@info @sprintf("    v_north (j+1) = %+12.6e m/s", v_north)
@info @sprintf("    w_bot   (k  ) = %+12.6e m/s", w_bot)
@info @sprintf("    w_top   (k+1) = %+12.6e m/s", w_top)
@info "═══════════════════════════════════════════════════════════════"

# Also print neighbor values for context
@info "Neighbor age values (yr):"
Nx′, Ny′, Nz′ = size(interior(c))
for dk in -1:1, dj in -1:1, di in -1:1
    ni, nj, nk = i + di, j + dj, k + dk
    if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′
        nval = @allowscalar c[ni, nj, nk]
        is_immersed = immersed_cell(ni, nj, nk, grid)
        is_partial = !is_immersed && any(
            immersed_cell(ni + di′, nj + dj′, nk + dk′, grid)
                for (di′, dj′, dk′) in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        )
        tag = is_immersed ? " [immersed]" : is_partial ? " [partial]" : ""
        @info @sprintf("  (%+d,%+d,%+d) → age = %8.4f yr", di, dj, dk, nval / year) * tag
    end
end

@info "debug_tendency_budget.jl complete"
flush(stdout); flush(stderr)
