"""
To run this on Gadi interactively, use

```
qsub -I -P y99 -l mem=47GB -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+gdata/ik11+scratch/y99
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia
include("src/ACCESS-OM2_grid.jl")
```

And on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99
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
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Adapt

# Comment/uncomment the following lines to enable/disable GPU
# using CUDA
# CUDA.set_runtime_version!(v"12.9.1")
# @show CUDA.versioninfo()

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

# TODO: Maybe I should only use the supergrid for the locations
# of center/face/corner points but otherwise use the "standard"
# grid metrics available from the MOM outputs?
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
# Here I use the z-coordinate grid (not using time-dependent dht output)
# TODO: use the dht from zstar coordinate (commented out below)
# FIXME: This path will need to be updated for different models/experiments
@show MOM_output_grid_inputdir = "/g/data/ik11/outputs/access-om2/$experiment/output305/ocean/"
MOM_output_grid_ds = open_dataset(joinpath(MOM_output_grid_inputdir, "ocean_grid.nc"))
ht = readcubedata(MOM_output_grid_ds.ht).data
ht = replace(ht, missing => 0.0)
kmt = readcubedata(MOM_output_grid_ds.kmt).data

@show MOM_input_vgrid_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_1deg/ocean_vgrid.nc"
z_ds = open_dataset(MOM_input_vgrid_file)
z = -reverse(vec(z_ds["zeta"].data[1:2:end]))
Nz = length(z) - 1
kbottom = round.(Union{Missing, Int}, Nz .- kmt .+ 1)

@show MOM_input_topo_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_1deg/topog.nc"
bottom_ds = open_dataset(MOM_input_topo_file)
bottom = -readcubedata(bottom_ds["depth"]).data
bottom = replace(bottom, 9999.0 => 0.0)

# Check if topography matches kmt and ht
@assert ht == -bottom
for idx in eachindex(kbottom)
    k = kbottom[idx]
    ismissing(kmt[idx]) && continue
    @assert z[k] ≤ bottom[idx] < z[k + 1]
end
@info "z coordinate/grid checks passed."

# Now since I am merging cells on the north fold,
# I just use the maximum depth on either side of the fold.
# TODO: This is a hack, so remove once the F-point pivot grid is implemented
# in Oceananigans!
Nx, Ny = size(bottom)
for i in 1:Nx
    bottom[i, Ny] = max(bottom[i, Ny], bottom[Nx - i + 1, Ny])
end

time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$model/$experiment/$time_window"

# dht_ds = open_dataset(joinpath(inputdir, "dht.nc")) # <- (new) cell thickness?
# dht = readcubedata(dht_ds.dht)

# TODO: I am not so sure what happens of merged wet/dry cells
# CHECK for both u/v and volumes etc.
underlying_grid = tripolargrid_from_supergrid(; supergrid..., z, Nz)

# Then immerge the grid cells with partial cells at the bottom
grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(bottom);
    active_cells_map = true,
    active_z_columns = true,

)

Nx, Ny, Nz = size(grid)

h = grid.immersed_boundary.bottom_height
fig = Figure()
ax = Axis(fig[2, 1], aspect = 1)
hm = surface!(
    ax,
    1:Nx, #view(grid.underlying_grid.λᶜᶜᵃ, 1:Nx, 1:Ny),
    1:Ny, #view(grid.underlying_grid.φᶜᶜᵃ, 1:Nx, 1:Ny),
    view(h.data, 1:Nx, 1:Ny, 1);
    colormap = :viridis
)
Colorbar(fig[1, 1], hm, vertical = false, label = "Bottom height (m)")
save(joinpath("output", "bottom_height_heatmap.png"), fig)

# #################
# # 3. Velocities #
# #################

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
u, v, w = velocities

fill_halo_regions!(u)
fill_halo_regions!(v)
fill_halo_regions!(w)
mask_immersed_field!(v, 0.0)
mask_immersed_field!(w, 0.0)
mask_immersed_field!(u, 0.0)

# TODO: Check velocities look reasonable (maybe against tx_trans etc.)

# u2, v2, w2 = deepcopy(u), deepcopy(v), deepcopy(w)
# mask_immersed_field!(v2, NaN)
# mask_immersed_field!(w2, NaN)
# mask_immersed_field!(u2, NaN)
# k = Nz
# opt = (; colormap = :RdBu, colorrange = (-1, 1), nan_color = (:black, 1))
# fig, ax, plt = heatmap(view(u2.data, 1:Nx, 1:Ny, k); opt..., axis = (; title = "u at k = $k"))
# plt2 = heatmap(fig[2, 1], view(u2.data, 1:Nx, 1:Ny, k - 1); opt..., axis = (; title = "u at k = $(k - 1)"))
# Label(fig[0, 1], "Near surface u (black = NaNs)", tellwidth = false)
# save(joinpath("output", "surface_u_heatmap.png"), fig)

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
    velocities = PrescribedVelocityFields(; u, v, w),
    tracer_advection = :RungeKutta3,
    closure = (horizontal_closure, vertical_closure),
    buoyancy = nothing
)

# Tracer patch for visualization
Gaussian(x, x₀, L) = exp(-((x - x₀)^2) / 2L^2)

# Tracer patch parameters
Lλ = 2 # degree
λ₀ = 80 # degrees
Lφ = 100 # degree
φ₀ = 0 # degrees
Lz = 100 # meters
z₀ = 0 # meters

cᵢ(λ, φ, z) = Gaussian(λ, λ₀, Lλ) * Gaussian(φ, φ₀, Lφ) * Gaussian(z, z₀, Lz)

set!(model, c = cᵢ)

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1])
mask = ones(Nx, Ny)
mask[bottom .== 0] .= NaN
hm = heatmap!(ax, model.tracers.c.data[1:Nx, 1:Ny, Nz] .* mask; colormap = :RdBu_9, colorrange = (-1, 1))
Colorbar(fig[1, 2], hm)
save(joinpath("output", "initial_c_surface.png"), fig)

Δt = 4500 # seconds

simulation = Simulation(
    model;
    Δt = Δt,
    stop_time = 100Δt,
)

function progress_message(sim)
    max_c, idx = findmax(adapt(Array, sim.model.tracers.c))
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(c) = %.1e at (%d, %d, %d) wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_c, idx.I..., walltime
    )
end

add_callback!(simulation, progress_message, IterationInterval(1))

c = model.tracers.c
output_fields = (; c, Cyz = Average(c, dims = 1), Cxy = Average(c, dims = 3))

output_prefix = joinpath("output", "offline_ACCESS-OM2_Nx$(grid.Nx)_Ny$(grid.Ny)_Nz$(grid.Nz)")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(Δt),
    filename = output_prefix,
    overwrite_existing = true
)

run!(simulation)

# ############
# # Plotting #
# ############

c_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "c")
cxy_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "Cxy")
cyz_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "Cyz")
times = c_timeseries.times

land = bottom .== 0
c_timeseries[land, :, :] .= NaN

set_theme!(Theme(fontsize = 30))

fig = Figure(size = (1920, 1080))

n = Observable(1)
k = 50
title = @lift "Tracer spot on offline OM2 at k = $k, t = " * prettytime(times[$n])

c = @lift c_timeseries[$n]

ckₙ = @lift view(($c).data, 1:Nx, 1:Ny, k)

ax = fig[1, 1] = Axis(
    fig,
    xlabel = "longitude index",
    ylabel = "latitude index",
    title = title,
)

hm = heatmap!(
    ax, ckₙ;
    colorrange = (-1, 1),
    # extendhigh = auto,
    # extendlow = auto,
    # colorscale = SymLog(0.01),
    colormap = :RdBu_9,
    nan_color = (:black, 1),
)
Colorbar(fig[1, 2], hm)

frames = 1:length(times)

@info "Making an animation..."

Makie.record(fig, joinpath("output", "offline_OM2_test.mp4"), frames, framerate = 25) do i
    n[] = i
end

# for n in 1:length(times)
#     fig = Figure(size = (1920, 1080))
#     ax = Axis(fig[1, 1])
#     hm = heatmap!(ax, cyz_timeseries[n])
#     Colorbar(fig[1, 2], hm)
#     save(joinpath("output", "Cyz_$n.png"), fig)
# end
