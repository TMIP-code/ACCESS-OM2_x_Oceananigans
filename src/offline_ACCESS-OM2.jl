"""
To run this on Gadi interactively, use

```
qsub -I -P y99 -l mem=47GB -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+scratch/y99
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia
include("src/ACCESS-OM2_grid.jl")
```

And on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+scratch/y99
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia
include("src/ACCESS-OM2_grid.jl")
```
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels

# Comment/uncomment the following lines to enable/disable GPU
# using CUDA
# CUDA.set_runtime_version!(v"12.9.1")
# @show CUDA.versioninfo()
# using Adapt

using Statistics
using YAXArrays
using DimensionalData
using NetCDF
using JLD2
using Printf
using CairoMakie

###########################
# 1. Horizontal supergrid #
###########################

model = "ACCESS-OM2-1"
modelsupergridfile = "mom$(split(model, "-")[end])deg.nc"
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", modelsupergridfile)
supergrid_ds = open_dataset(supergridfile)
@show supergrid_ds

include("tripolargrid_reader.jl")

# Unpack supergrid data
# TODO: I think best to extract the raw data here
# instead of passing YAXArrays
# TODO: For dimensions, just get the lengths instead of index ranges
# Not sure this matters but it is a bit more consistent
# with Nx, Ny, etc. used elsewhere where "N" or "n" means number of points
supergrid = (;
    x = readcubedata(supergrid_ds.x).data,
    y = readcubedata(supergrid_ds.y).data,
    dx = readcubedata(supergrid_ds.dx).data,
    dy = readcubedata(supergrid_ds.dy).data,
    area = readcubedata(supergrid_ds.area).data,
    nx = length(supergrid_ds.nx.val),
    nxp = length(supergrid_ds.nxp.val),
    ny = length(supergrid_ds.ny.val),
    nyp = length(supergrid_ds.nyp.val),
)

supergrid = convert_Fpointpivot_to_Tpointpivot(; supergrid...)

####################
# 2. Vertical grid #
####################

experiment = "1deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$model/$experiment/$time_window"
dht_ds = open_dataset(joinpath(inputdir, "dht.nc")) # <- (new) cell thickness?
dht = readcubedata(dht_ds.dht)
# Start with constant vertical grid as an approximation
# Chose the deepest water column as the reference
# TODO: match the vertical grid more closely later
dht0 = replace(dht.data, NaN => 0.0)
bottom = -dropdims(sum(dht0, dims = 3), dims = 3) # in MOM z increases downward
max_dht, imax = findmin(bottom, dims = (1, 2))
izmax, jzmax = Tuple(imax[1])
z = reverse(-[0; cumsum(dht0[izmax, jzmax, :])])
Nz = length(z) - 1

# TODO: I am not so sure what happens of merged wet/dry cells
# CHECK for both u/v and volumes etc.
underlying_grid = tripolargrid_from_supergrid(; supergrid..., z, Nz)

# Then immerge the grid cells with partial cells at the bottom
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

#################
# 3. Velocities #
#################

u_ds = open_dataset(joinpath(inputdir, "u.nc"))
u_data = replace(readcubedata(u_ds.u).data, NaN => 0.0)
v_ds = open_dataset(joinpath(inputdir, "v.nc"))
v_data = replace(readcubedata(v_ds.v).data, NaN => 0.0)

# Place u and v data on Oceananigans B-grid
u_Bgrid = Bgrid_velocity_from_MOM(grid, u_data)
v_Bgrid = Bgrid_velocity_from_MOM(grid, v_data)

# Then interpolate to C-grid
interp_u = @at (Face, Center, Center) 1 * u_Bgrid
u_Cgrid = Field{Face, Center, Center}(grid)
u_Cgrid .= interp_u
interp_v = @at (Center, Face, Center) 1 * v_Bgrid
v_Cgrid = Field{Center, Face, Center}(grid)
v_Cgrid .= interp_v

# Then compute w from continuity
w_Cgrid = Field{Center, Center, Face}(grid)
velocities = (u_Cgrid, v_Cgrid, w_Cgrid)
HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, CPU(), grid)

################
# 4. Diffusion #
################

horizontal_closure = HorizontalScalarDiffusivity(κ = 300)
vertical_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 3.0e-5)

############
# 5. Model #
############

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    tracers = :c,
    velocities = PrescribedVelocityFields(velocities...),
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

hm = heatmap!(
    ax, ck46ₙ;
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
