"""
To run this on Gadi interactively, use

```
qsub -I -P y99 -l mem=47GB -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+gdata/ik11+scratch/y99
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia
include("src/offline_ACCESS-OM2.jl")
```

And on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia
include("src/offline_ACCESS-OM2.jl")
```
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

#########################################
@info "0. Loading packages and functions"
#########################################

using Oceananigans

# Comment/uncomment the following lines to enable/disable GPU
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit=true)
    @show CUDA.versioninfo()
    arch = GPU()
else
    arch = CPU()
end
@info "Using $arch architecture"

using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Adapt: adapt

using Statistics
using YAXArrays
using DimensionalData
using NetCDF
using JLD2
using Printf
using CairoMakie

# TODO: Maybe I should only use the supergrid for the locations
# of center/face/corner points but otherwise use the "standard"
# grid metrics available from the MOM outputs?
include("tripolargrid_reader.jl")

model = "ACCESS-OM2-1"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output"

###############################
@info "1. Horizontal supergrid"
###############################

modelsupergridfile = "mom$(split(model, "-")[end])deg.nc"
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", modelsupergridfile)
supergrid_ds = open_dataset(supergridfile)
@show supergrid_ds



# Unpack supergrid data
# TODO: I think best to extract the raw data here
# instead of passing YAXArrays
# TODO: For dimensions, just get the lengths instead of index ranges
# Not sure this matters but it is a bit more consistent
# with Nx, Ny, etc. used elsewhere where "N" or "n" means number of points
println("Reading supergrid data into memory...")
MOMsupergrid = (;
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

println("Reading vertical grid data into memory...")
@show MOM_input_vgrid_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_1deg/ocean_vgrid.nc"
z_ds = open_dataset(MOM_input_vgrid_file)
z = -reverse(vec(z_ds["zeta"].data[1:2:end])) # from surface to bottom
Nz = length(z) - 1


println("Building Horizontal grid...")
underlying_grid = tripolargrid_from_supergrid(
    arch;
    MOMsupergrid...,
    halosize = (4, 4, 4),
    radius = Oceananigans.defaults.planet_radius,
    z,
    Nz,
)

for metric in (
        :λᶜᶜᵃ, :φᶜᶜᵃ, #:Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ,
        :λᶜᶠᵃ, :φᶜᶠᵃ, #:Δxᶜᶠᵃ, :Δyᶜᶠᵃ, :Azᶜᶠᵃ,
        :λᶠᶜᵃ, :φᶠᶜᵃ, #:Δxᶠᶜᵃ, :Δyᶠᶜᵃ, :Azᶠᶜᵃ,
        :λᶠᶠᵃ, :φᶠᶠᵃ, #:Δxᶠᶠᵃ, :Δyᶠᶠᵃ, :Azᶠᶠᵃ,
    )
    plot_surface_field(underlying_grid, metric)
end


########################
@info "2. Vertical grid"
########################

experiment = "1deg_jra55_iaf_omip2_cycle6"
# Here I use the z-coordinate grid (not using time-dependent dht output)
# TODO: use the dht from zstar coordinate (commented out below)
# FIXME: This path will need to be updated for different models/experiments
@show MOM_output_grid_inputdir = "/g/data/ik11/outputs/access-om2/$experiment/output305/ocean/"
MOM_output_grid_ds = open_dataset(joinpath(MOM_output_grid_inputdir, "ocean_grid.nc"))
ht = readcubedata(MOM_output_grid_ds.ht).data
ht = replace(ht, missing => 0.0)
kmt = readcubedata(MOM_output_grid_ds.kmt).data
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

time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$model/$experiment/$time_window"

# TODO: use time-dependent dht or η to adjust the vertical coordinates like in MOM.
# dht_ds = open_dataset(joinpath(inputdir, "dht.nc")) # <- (new) cell thickness?
# dht = readcubedata(dht_ds.dht)

# Then immerge the grid cells with partial cells at the bottom
bottom = on_architecture(arch, bottom)
grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(bottom);
    active_cells_map = true,
    active_z_columns = true,
)

Nx, Ny, Nz = size(grid)

h = on_architecture(CPU(), grid.immersed_boundary.bottom_height)
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
save(joinpath(outputdir, "bottom_height_heatmap.png"), fig)



#####################
@info "3. Velocities"
#####################

u_ds = open_dataset(joinpath(inputdir, "u.nc"))
u_data = replace(readcubedata(u_ds.u).data, NaN => 0.0)
v_ds = open_dataset(joinpath(inputdir, "v.nc"))
v_data = replace(readcubedata(v_ds.v).data, NaN => 0.0)


# Place u and v data on Oceananigans B-grid
u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)


fig = Figure(size = (1000, 1000))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, make_plottable_array(u_Bgrid)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)
ax = Axis(fig[2, 1])
hm = heatmap!(ax, make_plottable_array(v_Bgrid)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)
Colorbar(fig[3, 1], hm; vertical = false, tellwidth = false)
@show filepath = joinpath(outputdir, "surface_BGrid_u_v_halos.png")
save(filepath, fig)



# Then interpolate to C-grid
u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)


fig = Figure(size = (1000, 1000))

ax = Axis(fig[1, 1], title = "C-grid u")
hm = heatmap!(ax, make_plottable_array(u)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)

ax = Axis(fig[2, 1], title = "C-grid v")
hm = heatmap!(ax, make_plottable_array(v)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)

Colorbar(fig[3, 1], hm; vertical = false, tellwidth = false)
save(joinpath(outputdir, "surface_CGrid_u_v_halos.png"), fig)


# Then compute w from continuity
w = Field{Center, Center, Face}(grid)
velocities = (u, v, w)
mask_immersed_field!(u, 0.0)
mask_immersed_field!(v, 0.0)
fill_halo_regions!(u)
fill_halo_regions!(v)
HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, grid)
u, v, w = velocities



fig = Figure(size = (1000, 1000))

ax = Axis(fig[1, 1])
hm = heatmap!(ax, make_plottable_array(u)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)

ax = Axis(fig[2, 1])
hm = heatmap!(ax, make_plottable_array(v)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)

Colorbar(fig[3, 1], hm; vertical = false, tellwidth = false)
save(joinpath(outputdir, "surface_u_v_filled_halos.png"), fig)



fill_halo_regions!(w)
mask_immersed_field!(w, 0.0)

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, make_plottable_array(w)[:, :, Nz + 1]; colormap = :RdBu_9, colorrange = (-0.1, 0.1))
Colorbar(fig[1, 2], hm)
save(joinpath(outputdir, "w.png"), fig)



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
# save(joinpath(outputdir, "surface_u_heatmap.png"), fig)

####################
@info "4. Diffusion"
####################

horizontal_closure = HorizontalScalarDiffusivity(κ = 300)
vertical_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 3.0e-5)

################
@info "5. Model"
################

model = HydrostaticFreeSurfaceModel(
    grid;
    tracers = :c,
    velocities = PrescribedVelocityFields(; u, v, w),
    closure = (horizontal_closure, vertical_closure),
    buoyancy = nothing
)


############################
@info "6. Initial condition"
############################

# Gaussian for making a tracer patch as an initial condition
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
fill_halo_regions!(model.tracers.c)

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, make_plottable_array(model.tracers.c)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-1, 1))
Colorbar(fig[1, 2], hm)
save(joinpath(outputdir, "initial_c_surface.png"), fig)

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, make_plottable_array(model.tracers.c)[:, :, Nz]; colormap = :RdBu_9, colorrange = (-1, 1))
Colorbar(fig[1, 2], hm)
save(joinpath(outputdir, "initial_c_surface.png"), fig)


# fig = Figure(size = (1920, 1080))
# # mask = CenterField(grid)
# # mask .= 1.0
# # mask_immersed_field!(mask, NaN)
for (ishift, shift) in enumerate(-2:3)
    # ax = Axis(fig[ishift, 1])
    Δu = u.data[Nx + shift, 1:Ny, 1:Nz] - u.data[0 + shift, 1:Ny, 1:Nz]
    @show extrema(Δu)
    Δw = w.data[Nx + shift, 1:Ny, 1:Nz] - w.data[0 + shift, 1:Ny, 1:Nz]
    @show extrema(Δw)
    Δv = v.data[Nx + shift, 1:Ny, 1:Nz] - v.data[0 + shift, 1:Ny, 1:Nz]
    @show extrema(Δv)
    c = model.tracers.c
    Δc = c.data[Nx + shift, 1:Ny, 1:Nz] - c.data[0 + shift, 1:Ny, 1:Nz]
    @show extrema(Δc)
end
# Colorbar(fig[end + 1, 1:2], hm; vertical = false, tellwidth = false)
# save(joinpath(outputdir, "velocities_diff_periodic_BC.png"), fig)

#####################
@info "7. Simulation"
#####################

Δt = 4500 # seconds

simulation = Simulation(
    model;
    Δt = Δt,
    stop_time = 200Δt,
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

output_prefix = joinpath(outputdir, "offline_ACCESS-OM2_Nx$(grid.Nx)_Ny$(grid.Ny)_Nz$(grid.Nz)")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(Δt),
    filename = output_prefix,
    overwrite_existing = true
)

run!(simulation)

###################
@info "8. Plotting"
###################

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

ckₙ = @lift make_plottable_array($c)[1:Nx, 1:Ny, k]

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

Makie.record(fig, joinpath(outputdir, "offline_OM2_test.mp4"), frames, framerate = 25) do i
    n[] = i
end

