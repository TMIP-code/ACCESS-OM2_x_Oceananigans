"""
To run this on Gadi interactively on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99 -o scratch_output/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/offline_ACCESS-OM2.jl")
```
"""

#########################################
@info "0. Loading packages and functions"
#########################################

using Oceananigans

# Comment/uncomment the following lines to enable/disable GPU
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
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
using Oceananigans.Grids: znode
using Adapt: adapt
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days


using Statistics
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2
using Printf
using CairoMakie

# TODO: Maybe I should only use the supergrid for the locations
# of center/face/corner points but otherwise use the "standard"
# grid metrics available from the MOM outputs?
# -> No, instead I should use the same inputs! I think that is what I am doing now,
# but needs to be checked. I think best would be to read the input file location
# from the config file.

include("tripolargrid_reader.jl")

parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
mkpath(outputdir)

###############################
@info "1. Horizontal supergrid"
###############################

resolution_str = split(parentmodel, "-")[end]
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", "mom$(resolution_str)deg.nc")
supergrid_ds = open_dataset(supergridfile)


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
@show MOM_input_vgrid_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_$(resolution_str)deg/ocean_vgrid.nc"
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
        :λᶜᶜᵃ, :φᶜᶜᵃ, :Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ,
        :λᶜᶠᵃ, :φᶜᶠᵃ, :Δxᶜᶠᵃ, :Δyᶜᶠᵃ, :Azᶜᶠᵃ,
        :λᶠᶜᵃ, :φᶠᶜᵃ, :Δxᶠᶜᵃ, :Δyᶠᶜᵃ, :Azᶠᶜᵃ,
        :λᶠᶠᵃ, :φᶠᶠᵃ, :Δxᶠᶠᵃ, :Δyᶠᶠᵃ, :Azᶠᶠᵃ,
    )
    plot_surface_field(underlying_grid, metric)
end


########################
@info "2. Vertical grid"
########################

parentmodel_ik11path = parentmodel == "ACCESS-OM2-1" ? "access-om2" : "access-om2-$(resolution_str)"
# TODO: Fix this string for 0.1°
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
# Here I use the z-coordinate grid (not using time-dependent dht output)
# TODO: use the dht from zstar coordinate (commented out below)
# FIXME: This path will need to be updated for different models/experiments
@show MOM_output_grid_inputdir = "/g/data/ik11/outputs/$(parentmodel_ik11path)/$experiment/output305/ocean/"
MOM_output_grid_ds = open_dataset(joinpath(MOM_output_grid_inputdir, "ocean_grid.nc"))
ht = readcubedata(MOM_output_grid_ds.ht).data
ht = replace(ht, missing => 0.0)
kmt = readcubedata(MOM_output_grid_ds.kmt).data
kbottom = round.(Union{Missing, Int}, Nz .- kmt .+ 1)

@show MOM_input_topo_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_$(resolution_str)deg/topog.nc"
bottom_ds = open_dataset(MOM_input_topo_file)
bottom = -readcubedata(bottom_ds["depth"]).data
bottom = replace(bottom, 9999.0 => 0.0)

# Check if topography matches kmt and ht
@assert ht == -bottom
for idx in eachindex(kbottom)
    local k = kbottom[idx]
    ismissing(kmt[idx]) && continue
    @assert z[k] ≤ bottom[idx] < z[k + 1]
end
@info "z coordinate/grid checks passed."

time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

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

plottable_u = make_plottable_array(u)
plottable_v = make_plottable_array(v)
for k in 1:50
    local fig = Figure(size = (1000, 1000))
    local ax = Axis(fig[1, 1], title = "C-grid u")
    local hm = heatmap!(ax, plottable_u[:, :, k]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)
    ax = Axis(fig[2, 1], title = "C-grid v")
    hm = heatmap!(ax, plottable_v[:, :, k]; colormap = :RdBu_9, colorrange = (-0.1, 0.1), nan_color = :black)
    Colorbar(fig[3, 1], hm; vertical = false, tellwidth = false)
    save(joinpath(outputdir, "surface_CGrid_u_v_halos_k$k.png"), fig)
end



# Then compute w from continuity
w = Field{Center, Center, Face}(grid)
velocities = (u, v, w)
mask_immersed_field!(u, 0.0)
mask_immersed_field!(v, 0.0)
fill_halo_regions!(u)
fill_halo_regions!(v)
HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, grid)
u, v, w = velocities

plottable_u = make_plottable_array(u)
plottable_v = make_plottable_array(v)
plottable_w = make_plottable_array(w)
for k in 1:50
    local fig = Figure(size = (1200, 1800))
    local ax = Axis(fig[1, 1], title = "C-grid u")
    local velocity2D = plottable_u[:, :, k]
    local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    Colorbar(fig[1, 2], hm)
    ax = Axis(fig[2, 1], title = "C-grid v")
    velocity2D = plottable_v[:, :, k]
    maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    Colorbar(fig[2, 2], hm)
    ax = Axis(fig[3, 1], title = "C-grid w")
    velocity2D = plottable_w[:, :, k + 1]
    maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    Colorbar(fig[3, 2], hm)
    save(joinpath(outputdir, "CGrid_velocities_filledhalos_k$k.png"), fig)
end


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

# TODO: Try to match ACCESS-OM2 as much as possible

# Add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
κVML = 0.1 # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)
vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κVField,
)
fig, ax, plt = heatmap(
    make_plottable_array(κVField)[:, :, 10];
    colorscale = log10,
    colormap = :viridis,
    axis = (; title = "Vertical diffusivity at k = 10"),
    colorrange = (1e-5, 3e-1),
    lowclip = :red,
    highclip = :cyan,
)
Colorbar(fig[1, 2], plt, ticks = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
save(joinpath(outputdir, "vertical_diffusivity_k10.png"), fig)


horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
# horizontal_diffusion = IsopycnalSkewSymmetricDiffusivity(
#     time_disc = VerticallyImplicitTimeDiscretization();
#     κ_skew = 300,
#     κ_symmetric = 300,
#     # skew_flux_formulation = DiffusiveFormulation(),
#     # isopycnal_tensor = SmallSlopeIsopycnalTensor(),
#     # slope_limiter = FluxTapering(1e-2),
# )

# Combine them all into a single diffusion closure
# TODO: Remove mixed_layer_diffusion if 0.1°?
closure = (
    horizontal_diffusion,
    vertical_diffusion,
)

################
@info "5. Model"
################

Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds

# Maybe I should clamp the age manually after all?
@kernel function _age_forcing_callback(age, Nz, age_initial, elapsed_time)
    i, j, k = @index(Global, NTuple)

    age[i, j, k] = ifelse(k == Nz, 0, age[i, j, k])
    age[i, j, k] = max(age[i, j, k], 0)
    age[i, j, k] = min(age[i, j, k], age_initial[i, j, k] + elapsed_time)

end

function age_forcing_callback(sim)
    age = sim.model.tracers.age
    Nz = sim.model.grid.Nz
    age[:,:,Nz] .= 0.0
    clamp!(age[:,:,Nz], 0.0, Inf)
end


age_parameters = (;
    # relaxation_timescale = 3Δt,     # Relaxation timescale for removing age at surface
    source_rate = 1.0 / year,         # Source for the age (in years)
)

# @inline age_source(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)
@inline age_source(i, j, k, grid, clock, fields, params) = params.source_rate

age_dynamics = Forcing(
    age_source,
    parameters = age_parameters,
    discrete_form = true,
)

model = HydrostaticFreeSurfaceModel(
    grid;
    tracers = (:age,),
    # timestepper = :SplitRungeKutta3, # <- to try and improve numerical stability over AB2
    velocities = PrescribedVelocityFields(; u, v, w),
    closure = closure,
    forcing = (; age = age_dynamics),
    buoyancy = nothing,
)

############################
@info "6. Initial condition"
############################

# # Gaussian for making a tracer patch as an initial condition
# Gaussian(x, x₀, L) = exp(-((x - x₀)^2) / 2L^2)

# # Tracer patch parameters
# Lλ = 2 # degree
# λ₀ = 80 # degrees
# Lφ = 100 # degree
# φ₀ = 0 # degrees
# Lz = 100 # meters
# z₀ = 0 # meters

# cᵢ(λ, φ, z) = Gaussian(λ, λ₀, Lλ) * Gaussian(φ, φ₀, Lφ) * Gaussian(z, z₀, Lz)
ageᵢ(λ, φ, z) = 0

set!(model, age = ageᵢ)
# set!(model, Returns(0.0))
# fill_halo_regions!(model.tracers.age)

fig, ax, plt = heatmap(
    make_plottable_array(model.tracers.age)[:, :, Nz];
    colormap = :viridis,
    axis = (; title = "Initial age at surface (years)"),
)
Colorbar(fig[1, 2], plt)
save(joinpath(outputdir, "initial_age_surface.png"), fig)

#####################
@info "7. Simulation"
#####################

# stop_time = 2years
stop_time = 1day

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx = findmax(adapt(Array, sim.model.tracers.age)) # in years
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d) wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age, idx.I..., walltime
    )
end

# add_callback!(simulation, progress_message, TimeInterval(1year))
add_callback!(simulation, progress_message, IterationInterval(1))
add_callback!(simulation, zero_age_callback, IterationInterval(1))


output_fields = Dict(
    # "age" => model -> make_plottable_array(model.tracers.age) / year, # save age in years
    "age" => model.tracers.age, # save age in years
)
# output_dims = Dict(
#     "age" => ("x_ccc", "y_ccc", "z_ccc"),
# )

output_prefix = joinpath(outputdir, "offline_age_$parentmodel")

# simulation.output_writers[:fields] = NetCDFWriter(
simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(1year),
    filename = output_prefix,
    # dimensions=output_dims,
    # include_grid_metrics = true,
    # verbose = true,
    # array_type = Array{Float32},
    overwrite_existing = true,
)

run!(simulation)

###################
@info "8. Plotting"
###################

plottable_age = make_plottable_array(model.tracers.age)
for k in 1:50
    local fig, ax, plt = heatmap(
        plottable_age[:, :, k];
        # on_architecture(CPU(), model.tracers.age.data.parent)[:, :, Nz];
        nan_color = :black,
        colorrange = (0, 5 / 4 * stop_time / year),
        colormap = cgrad(cgrad(:tab20b; categorical = true)[1:20], categorical = true),
        lowclip = :yellow,
        highclip = :cyan,
        axis = (; title = "Final age (years) at level k = $k"),
    )
    Colorbar(fig[1, 2], plt)
    save(joinpath(outputdir, "final_age_k$(k)_$(parentmodel).png"), fig)
end

foo


# age_lazy = open_dataset(simulation.output_writers[:fields].filepath)["age"]
age_lazy = FieldTimeSeries(simulation.output_writers[:fields].filepath, "age")
times = age_lazy.times

set_theme!(Theme(fontsize = 30))

fig = Figure(size = (1920, 1080))

n = Observable(1)
k = 20
title = @lift "age on offline OM2 at k = $k, t = " * prettytime(times[$n])

# ckₙ = @lift readcubedata(age_lazy[At(k = $k, times = [$n])]) # in years
ckₙ = @lift make_plottable_array(age_lazy[$n])[:, :, k] / year # in years

ax = fig[1, 1] = Axis(
    fig,
    xlabel = "longitude index",
    ylabel = "latitude index",
    title = title,
)

hm = heatmap!(
    ax, ckₙ;
    colorrange = (-2, 2),
    # extendhigh = auto,
    # extendlow = auto,
    # colorscale = SymLog(0.01),
    colormap = :RdBu,
    nan_color = (:black, 1),
)
Colorbar(fig[1, 2], hm)

frames = 1:length(times)

@info "Making an animation..."

Makie.record(fig, joinpath(outputdir, "offline_age_OM2_test.mp4"), frames, framerate = 25) do i
    println("frame $i/$(length(frames))")
    n[] = i
end
