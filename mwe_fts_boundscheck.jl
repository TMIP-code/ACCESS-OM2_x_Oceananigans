# MWE: FieldTimeSeries BoundsError in HydrostaticFreeSurfaceModel
#
# Reproduces a GPU BoundsError that occurs when using PrescribedFreeSurface
# with an η FieldTimeSeries. The η FTS is z-reduced {Center, Center, Nothing},
# but unlike a regular ZReducedField, the FTS getindex does NOT collapse k to 1.
# When _update_zstar_scaling! accesses η[i, j, Nz+1], it triggers a BoundsError.
#
# This MWE mimics the real workflow: write FTS to JLD2 via OnDisk, then reload
# with InMemory(4) + Cyclical(1year), exactly as create_velocities.jl → setup_model.jl.
#
# Run with: julia --check-bounds=yes --project mwe_fts_boundscheck.jl

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Grids: znode, MutableVerticalDiscretization
using Oceananigans.OutputReaders: Cyclical, InMemory, OnDisk
using Oceananigans.Units: second, seconds, day, days

year = years = 365.25days
month = months = year / 12

# ── Architecture ─────────────────────────────────────────────────────────
arch = try
    using CUDA
    CUDA.functional() ? GPU() : CPU()
catch
    CPU()
end
@info "Using $arch"

# ── Tiny TripolarGrid (RightFaceFolded, like ACCESS-OM2) ─────────────────
# Note: Nx must be even for TripolarGrid; Ny gets +1 for RightFaceFolded
# MutableVerticalDiscretization enables z-star coordinate scaling, which is
# what the real ACCESS-OM2 grid uses. Without it, _update_zstar_scaling! never
# runs and the suspected η[i, j, Nz+1] BoundsError cannot be triggered.
Nz = 4
z_faces = collect(range(-5000, 0, length = Nz + 1))
underlying_grid = TripolarGrid(
    arch;
    size = (8, 9, Nz),
    z = MutableVerticalDiscretization(z_faces),
    halo = (4, 4, 4),
    fold_topology = RightFaceFolded,
)
# For RightFaceFolded TripolarGrid, size(grid) includes the fold point (Ny=9),
# but Center-located field interiors exclude it (Ny=8). The bottom array must
# match the interior size, so it's Nx × (Ny-1) = 8 × 8.
#
# Build a non-trivial bottom topography so some cells are actually immersed.
# z levels are evenly spaced: z = [-5000, -3750, -2500, -1250, 0]
# Bottom depths vary from -5000m (full depth) to -1000m (shallow), creating
# real partial cells at various levels.
bottom = [
    -5000.0 -4000.0 -3000.0 -2000.0 -1000.0 -2000.0 -3000.0 -4000.0
    -4000.0 -3500.0 -2500.0 -1500.0 -1000.0 -1500.0 -2500.0 -3500.0
    -3000.0 -2500.0 -2000.0 -1000.0 -1000.0 -1000.0 -2000.0 -2500.0
    -2000.0 -1500.0 -1000.0 -1000.0 -1000.0 -1000.0 -1000.0 -1500.0
    -3000.0 -2500.0 -2000.0 -1000.0 -1000.0 -1000.0 -2000.0 -2500.0
    -4000.0 -3500.0 -2500.0 -1500.0 -1000.0 -1500.0 -2500.0 -3500.0
    -5000.0 -4000.0 -3000.0 -2000.0 -1000.0 -2000.0 -3000.0 -4000.0
    -5000.0 -5000.0 -4000.0 -3000.0 -2000.0 -3000.0 -4000.0 -5000.0
]
@assert size(bottom) == (8, 8)  # Nx × (Ny - 1)

grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(bottom);
    active_cells_map = true,
    active_z_columns = true,
)

# ── Write FieldTimeSeries to JLD2 (mimics create_velocities.jl) ──────────
prescribed_dt = 1month
fts_times = collect(((1:12) .- 0.5) * prescribed_dt)
stop_time = 12 * prescribed_dt

tmpdir = mktempdir()
u_file = joinpath(tmpdir, "u_periodic.jld2")
v_file = joinpath(tmpdir, "v_periodic.jld2")
η_file = joinpath(tmpdir, "eta_periodic.jld2")

@info "Writing FTS to JLD2 in $tmpdir"

# u: Face, Center, Center — uniform eastward flow
u_ondisk = FieldTimeSeries{Face, Center, Center}(grid, fts_times;
    backend = OnDisk(), path = u_file, name = "u",
    time_indexing = Cyclical(stop_time))
for n in 1:12
    u_field = Field{Face, Center, Center}(grid)
    set!(u_field, 0.01)
    set!(u_ondisk, u_field, n)
end

# v: Center, Face, Center — zero
v_ondisk = FieldTimeSeries{Center, Face, Center}(grid, fts_times;
    backend = OnDisk(), path = v_file, name = "v",
    time_indexing = Cyclical(stop_time))
for n in 1:12
    v_field = Field{Center, Face, Center}(grid)
    set!(v_field, 0.0)
    set!(v_ondisk, v_field, n)
end

# η: Center, Center, Face at k = Nz+1 (surface face).
# Using Face in z with indices restricted to Nz+1 ensures _update_zstar_scaling!
# can access η[i, j, Nz+1] without a BoundsError.
# η_ondisk = FieldTimeSeries{Center, Center, Face}(grid, fts_times;
η_ondisk = FieldTimeSeries{Center, Center, Face}(grid, fts_times;
    indices = (:, :, Nz + 1),
    backend = OnDisk(), path = η_file, name = "η",
    time_indexing = Cyclical(stop_time))
for n in 1:12
    # η_field = Field{Center, Center, Face}(grid; indices = (:, :, Nz + 1))
    η_field = Field{Center, Center, Face}(grid; indices = (:, :, Nz + 1))
    set!(η_field, sin(2pi * n / 12))
    set!(η_ondisk, η_field, n)
end

@info "FTS written to JLD2"

# ── Reload from JLD2 (mimics setup_model.jl) ────────────────────────────
N_in_mem = 4
backend = InMemory(N_in_mem)
time_indexing = Cyclical(1year)

u_ts = FieldTimeSeries(u_file, "u";
    architecture = arch, grid, backend, time_indexing)
@show u_ts

v_ts = FieldTimeSeries(v_file, "v";
    architecture = arch, grid, backend, time_indexing)
@show v_ts

η_ts = FieldTimeSeries(η_file, "η";
    architecture = arch, grid, backend, time_indexing)
@show η_ts

# ── Velocities and free surface ──────────────────────────────────────────
velocities = PrescribedVelocityFields(u = u_ts, v = v_ts,
    formulation = DiagnosticVerticalVelocity())

free_surface = PrescribedFreeSurface(displacement = η_ts)

# ── Closures (mimics setup_model.jl) ─────────────────────────────────────
z_center = znodes(grid, Center(), Center(), Center())

# Simple MLD-based vertical diffusivity: strong in top layer, weak below
mld = -500.0  # uniform mixed layer depth (m)
is_above_mld = z_center .> mld
κV_data = 0.1 .* is_above_mld .+ 3e-5 .* .!is_above_mld
κV_field = CenterField(grid)
set!(κV_field, reshape(κV_data, 1, 1, Nz))

implicit_vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κV_field,
)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

closure = (
    horizontal_diffusion,
    implicit_vertical_diffusion,
)

# ── Age forcing (mimics setup_model.jl) ──────────────────────────────────
dt = 5400seconds
age_parameters = (;
    relaxation_timescale = 3dt,
    source_rate = 1.0,
)

@inline age_source_sink(i, j, k, grid, clock, fields, params) =
    ifelse(k >= grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)

age_dynamics = Forcing(
    age_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)

forcing = (age = age_dynamics,)

# ── Model ────────────────────────────────────────────────────────────────
model = HydrostaticFreeSurfaceModel(
    grid;
    tracer_advection = Centered(order = 2),
    velocities,
    free_surface,
    tracers = (; age = CenterField(grid)),
    closure,
    forcing,
)

set!(model, age = 0.0)

# ── Simulation ───────────────────────────────────────────────────────────
simulation = Simulation(model; Δt = dt, stop_time)

function progress(sim)
    @info "Iteration $(iteration(sim)), time = $(time(sim) / day) days"
end
add_callback!(simulation, progress, TimeInterval(prescribed_dt))

@info "Running simulation for $(stop_time / day) days with Δt = $(dt / second) s"
run!(simulation)

@info "Simulation complete — no BoundsError occurred"
