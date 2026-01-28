"""
To run this on Gadi interactively on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q normal -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+gdata/ik11+scratch/y99 -o scratch_output/PBS/ -j oe
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99 -o scratch_output/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/offline_ACCESS-OM2.jl")
```
"""

@info "0. Loading packages and functions"

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
month = months = year / 12

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

################################################################################

@info "1. Horizontal supergrid"

resolution_str = split(parentmodel, "-")[end]
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", "mom$(resolution_str)deg.nc")
supergrid_ds = open_dataset(supergridfile)

# Unpack supergrid data
# TODO: I think best to extract the raw data here
# instead of passing YAXArrays
println("Reading supergrid data into memory...")
MOMsupergrid = (;
    x = readcubedata(supergrid_ds.x).data,
    y = readcubedata(supergrid_ds.y).data,
    dx = readcubedata(supergrid_ds.dx).data,
    dy = readcubedata(supergrid_ds.dy).data,
    area = readcubedata(supergrid_ds.area).data,
    # For dimensions, use indices lengths instead of full ranges.
    # This is more consistent with Oceananigans.jl conventions for Nx, Ny, etc.
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

################################################################################

@info "2. Vertical grid"

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
save(joinpath(outputdir, "bottom_height_heatmap_$(arch).png"), fig)

################################################################################

@info "3. Velocities"

# TODO: Probably factor this out into separate setup file,
# to be run once to sabe to JLD2 or something else,
# And then just load it here.
# just in case it fills up all the GPU memory (not sure it does)
# TODO: Figure out the best way to load the data for performance (IO).

# TODO: Comment/uncomment below. Setting times as 12 days for testing only.
# presribed_Δt = 1month
presribed_Δt = 1day
times = adapt(arch, ((1:12) .- 0.5) * presribed_Δt)

u_ts = FieldTimeSeries{Face, Center, Center}(grid, times)
v_ts = FieldTimeSeries{Center, Face, Center}(grid, times)
w_ts = FieldTimeSeries{Center, Center, Face}(grid, times)

u_ds = open_dataset(joinpath(inputdir, "u_periodic.nc"))
v_ds = open_dataset(joinpath(inputdir, "v_periodic.nc"))

print("month ")
for month in 1:12
    print("$month, ")

    u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
    v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)

    # Place u and v data on Oceananigans B-grid
    u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)

    # plottable_u = make_plottable_array(u_Bgrid)
    # plottable_v = make_plottable_array(v_Bgrid)
    # for k in 1:50
    #     local fig = Figure(size = (1200, 1200))
    #     local ax = Axis(fig[1, 1], title = "C-grid u")
    #     local velocity2D = plottable_u[:, :, k]
    #     local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[1, 2], hm)
    #     ax = Axis(fig[2, 1], title = "C-grid v")
    #     velocity2D = plottable_v[:, :, k]
    #     maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[2, 2], hm)
    #     save(joinpath(outputdir, "velocities/BGrid_velocities_$(k)_month$(month)_$(arch).png"), fig)
    # end

    # Then interpolate to C-grid
    u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)

    # plottable_u = make_plottable_array(u)
    # plottable_v = make_plottable_array(v)
    # for k in 1:50
    #     local fig = Figure(size = (1200, 1200))
    #     local ax = Axis(fig[1, 1], title = "C-grid u")
    #     local velocity2D = plottable_u[:, :, k]
    #     local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[1, 2], hm)
    #     ax = Axis(fig[2, 1], title = "C-grid v")
    #     velocity2D = plottable_v[:, :, k]
    #     maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[2, 2], hm)
    #     save(joinpath(outputdir, "velocities/CGrid_velocities_$(k)_month$(month)_$(arch).png"), fig)
    # end

    # Then compute w from continuity
    w = Field{Center, Center, Face}(grid)
    mask_immersed_field!(u, 0.0)
    mask_immersed_field!(v, 0.0)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    velocities = (u, v, w)
    HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, grid)
    u, v, w = velocities

    # plottable_u = make_plottable_array(u)
    # plottable_v = make_plottable_array(v)
    # plottable_w = make_plottable_array(w)
    # for k in 1:50
    #     local fig = Figure(size = (1200, 1800))
    #     local ax = Axis(fig[1, 1], title = "C-grid u")
    #     local velocity2D = plottable_u[:, :, k]
    #     local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[1, 2], hm)
    #     ax = Axis(fig[2, 1], title = "C-grid v")
    #     velocity2D = plottable_v[:, :, k]
    #     maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[2, 2], hm)
    #     ax = Axis(fig[3, 1], title = "C-grid w")
    #     velocity2D = plottable_w[:, :, k + 1]
    #     maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[3, 2], hm)
    #     save(joinpath(outputdir, "velocities/CGrid_velocities_final_k$(k)_month$(month)_$(arch).png"), fig)
    # end


    # u2, v2, w2 = deepcopy(u), deepcopy(v), deepcopy(w)
    # mask_immersed_field!(v2, NaN)
    # mask_immersed_field!(w2, NaN)
    # mask_immersed_field!(u2, NaN)
    # k = Nz
    # opt = (; colormap = :RdBu, colorrange = (-1, 1), nan_color = (:black, 1))
    # fig, ax, plt = heatmap(view(u2.data, 1:Nx, 1:Ny, k); opt..., axis = (; title = "u at k = $k"))
    # plt2 = heatmap(fig[2, 1], view(u2.data, 1:Nx, 1:Ny, k - 1); opt..., axis = (; title = "u at k = $(k - 1)"))
    # Label(fig[0, 1], "Near surface u (black = NaNs)", tellwidth = false)
    # save(joinpath(outputdir, "velocities/surface_u_heatmap_$(arch).png"), fig)

    set!(u_ts, u, month)
    set!(v_ts, v, month)
    set!(w_ts, w, month)

end
println("Done!")

velocities = PrescribedVelocityFields(; u = u_ts, v = v_ts, w = w_ts)

################################################################################

@info "4. Diffusion"

# TODO: Try to match ACCESS-OM2 as much as possible

κVField_ts = FieldTimeSeries{Center, Center, Center}(grid, times)

# Load MLD to add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld_periodic.nc"))

print("month ")
for month in 1:12
    print("$month, ")

    mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld[month = At(month)]).data, NaN => 0.0))
    z_center = znodes(grid, Center(), Center(), Center())
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    κVField = CenterField(grid)
    κVML = 0.1 # m^2/s in the mixed layer
    κVBG = 3.0e-5 # m^2/s in the ocean interior (background)
    set!(κVField, κVML * is_mld + κVBG * .!is_mld)

    # fig, ax, plt = heatmap(
    #     make_plottable_array(κVField)[:, :, 10];
    #     colorscale = log10,
    #     colormap = :viridis,
    #     axis = (; title = "Vertical diffusivity at k = 10"),
    #     colorrange = (1e-5, 3e-1),
    #     lowclip = :red,
    #     highclip = :cyan,
    # )
    # Colorbar(fig[1, 2], plt, ticks = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    # save(joinpath(outputdir, "vertical_diffusivity_k10_month$(month)_$(arch).png"), fig)

    set!(κVField_ts, κVField, month)
end
println("Done!")

vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κVField_ts,
)

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

################################################################################

@info "5. Model"

Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds

# Maybe I should clamp the age manually after all?
# @kernel function _age_forcing_callback(age, Nz, age_initial, elapsed_time)
#     i, j, k = @index(Global, NTuple)

#     age[i, j, k] = ifelse(k == Nz, 0, age[i, j, k])
#     age[i, j, k] = max(age[i, j, k], 0)
#     age[i, j, k] = min(age[i, j, k], age_initial[i, j, k] + elapsed_time)

# end

# function age_forcing_callback(sim)
#     age = sim.model.tracers.age
#     Nz = sim.model.grid.Nz
#     age[:,:,Nz] .= 0.0
#     clamp!(age[:,:,Nz], 0.0, Inf)
# end


age_parameters = (;
    relaxation_timescale = 3Δt,     # Relaxation timescale for removing age at surface
    source_rate = 1.0 / year,         # Source for the age (in years)
)

@inline age_source(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)
# @inline age_source(i, j, k, grid, clock, fields, params) = params.source_rate

age_dynamics = Forcing(
    age_source,
    parameters = age_parameters,
    discrete_form = true,
)

model = HydrostaticFreeSurfaceModel(
    grid;
    tracers = (:age,),
    tracer_advection = WENO(order = 5),
    # tracer_advection = UpwindBiased(),
    # timestepper = :SplitRungeKutta3, # <- to try and improve numerical stability over AB2
    velocities = velocities,
    closure = closure,
    forcing = (; age = age_dynamics),
    buoyancy = nothing,
)

################################################################################

@info "6. Initial condition"

set!(model, age = Returns(0.0)) # TODO: Unneccessary as fields are initialized to zero by default.
# fill_halo_regions!(model.tracers.age)

fig, ax, plt = heatmap(
    make_plottable_array(model.tracers.age)[:, :, Nz];
    colormap = :viridis,
    axis = (; title = "Initial age at surface (years)"),
)
Colorbar(fig[1, 2], plt)
save(joinpath(outputdir, "initial_age_surface_$(arch).png"), fig)

################################################################################

@info "7. Simulation"

stop_time = 12 * presribed_Δt

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx = findmax(adapt(Array, sim.model.tracers.age)) # in years
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        # "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d) wall time: %s\n",
        # iteration(sim), time(sim), sim.Δt, max_age, idx.I..., walltime
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age)/time = %.1e at (%d, %d, %d) wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age / (time(sim)/year), idx.I..., walltime
    )
end

# add_callback!(simulation, progress_message, TimeInterval(1year))
add_callback!(simulation, progress_message, TimeInterval(presribed_Δt))
# add_callback!(simulation, zero_age_callback, IterationInterval(1))


output_fields = Dict(
    # "age" => model -> make_plottable_array(model.tracers.age) / year, # save age in years
    "age" => model.tracers.age, # save age in years
    "u" => model.velocities.u,
)
# output_dims = Dict(
#     "age" => ("x_ccc", "y_ccc", "z_ccc"),
# )

output_prefix = joinpath(outputdir, "offline_age_$(parentmodel)_$(arch)")

# simulation.output_writers[:fields] = NetCDFWriter(
simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(presribed_Δt),
    # schedule = IterationInterval(1),
    filename = output_prefix,
    # dimensions=output_dims,
    # include_grid_metrics = true,
    # verbose = true,
    # array_type = Array{Float32},
    overwrite_existing = true,
)

run!(simulation)

################################################################################

@info "8. Plotting"

plottable_age = make_plottable_array(model.tracers.age)
for k in 1:50
    local fig, ax, plt = heatmap(
        plottable_age[:, :, k];
        # on_architecture(CPU(), model.tracers.age.data.parent)[:, :, Nz];
        nan_color = :black,
        colorrange = (0, stop_time / year),
        colormap = cgrad(cgrad(:tab20b; categorical = true)[1:16], categorical = true),
        lowclip = :red,
        highclip = :yellow,
        axis = (; title = "Final age (years) at level k = $k"),
    )
    Colorbar(fig[1, 2], plt)
    save(joinpath(outputdir, "final_age_k$(k)_$(parentmodel)_$(arch).png"), fig)
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
