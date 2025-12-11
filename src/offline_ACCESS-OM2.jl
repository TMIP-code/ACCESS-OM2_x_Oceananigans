# qsub -I -P y99 -q gpuvolta -l mem=96GB -l walltime=01:00:00 -l ngpus=1 -l ncpus=12

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels

using CUDA
CUDA.set_runtime_version!(v"12.9.1")
@show CUDA.versioninfo()
using Adapt

using Statistics
using JLD2
using Printf
using CairoMakie

Nx = 360
Ny = 300
Nz = 50
latitude = (-60, 60)
longitude = (-180, 180)
z = (-6000, 0)

# A spherical domain
grid = LatitudeLongitudeGrid(
    GPU(),
    size = (Nx, Ny, Nz),
    latitude = latitude,
    longitude = longitude,
    z = z,
)

# grid = TripolarGrid(
#     arch = GPU();
#     size,
#     southernmost_latitude = -80,
#     halo = (4, 4, 4),
#     radius = Oceananigans.defaults.planet_radius,
#     z = (-1000, 0),
#     north_poles_latitude = 55,
#     first_pole_longitude = 70,
# )

u(x, y, z, t) = 1.0


horizontal_closure = HorizontalScalarDiffusivity(κ = 300)
vertical_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 3.0e-5)

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    tracers = :c,
    velocities = PrescribedVelocityFields(u = u),
    closure = (horizontal_closure, vertical_closure),
    buoyancy = nothing
)

# Tracer patch for visualization
Gaussian(λ, φ, L) = exp(-(λ^2 + φ^2) / 2L^2)
Gaussian(z, Lz) = exp(-(z^2) / 2Lz^2)

# Tracer patch parameters
L = 12 # degree
φ₀ = 0 # degrees
Lz = 100 # meters
z₀ = -500 # meters

cᵢ(λ, φ, z) = Gaussian(λ, φ - φ₀, L) * Gaussian(z - z₀, Lz)

set!(model, c = cᵢ)

Δt = 4500 # seconds

simulation = Simulation(
    model;
    Δt = Δt,
    stop_time = 1000Δt,
)

function progress_message(sim)
    max_c, idx = findmax(adapt(Array, sim.model.tracers.c))
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(c) = %.1e at (%d, %d, %d) wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_c, idx.I..., walltime
    )
end

add_callback!(simulation, progress_message, IterationInterval(100))

c = model.tracers.c
output_fields = (; c, Cyz = Average(c, dims = 1), Cxy = Average(c, dims = 3))

output_prefix = joinpath("output", "offline_ACCESS-OM2_Nx$(grid.Nx)_Ny$(grid.Ny)_Nz$(grid.Nz)")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(20Δt),
    filename = output_prefix,
    overwrite_existing = true
)

run!(simulation)

λ = λnodes(grid, Center(), Center(), Center())
φ = φnodes(grid, Center(), Center(), Center())

λ = repeat(reshape(λ, Nx, 1), 1, Ny)
φ = repeat(reshape(φ, 1, Ny), Nx, 1)

c_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "c")
cxy_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "Cxy")
cyz_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "Cyz")
times = c_timeseries.times

set_theme!(Theme(fontsize = 30))

fig = Figure(size = (1920, 1080))

n = Observable(1)
title = @lift "Tracer spot on a latlon at k = 46, t = " * prettytime(times[$n])

plot_title = "hi"

c = @lift c_timeseries[$n]

ck46ₙ = @lift view(c_timeseries[$n], :, :, 46)

fig = Figure(size = (1920, 1080))

ax = fig[1, 1] = Axis(
    fig,
    xlabel = "λ",
    ylabel = "φ",
    title = title,
)

hm = heatmap!(ax, ck46ₙ;
    colorrange = (-1, 1),
    # extendhigh = auto,
    # extendlow = auto,
    # colorscale = SymLog(0.01),
    colormap = :RdBu,
)
Colorbar(fig[1, 2], hm)

frames = 1:length(times)

@info "Making an animation..."

Makie.record(fig, joinpath("output", "offline_GPUlatlon_diffusion.mp4"), frames, framerate = 60) do i
    n[] = i
end

# for n in 1:length(times)
#     fig = Figure(size = (1920, 1080))
#     ax = Axis(fig[1, 1])
#     hm = heatmap!(ax, cyz_timeseries[n])
#     Colorbar(fig[1, 2], hm)
#     save(joinpath("output", "Cyz_$n.png"), fig)
# end
