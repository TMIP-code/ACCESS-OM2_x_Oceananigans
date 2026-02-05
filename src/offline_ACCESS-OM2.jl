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

@info "Loading packages and functions"

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
# using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode, get_active_cells_map
using Oceananigans.Simulations: reset!
using Oceananigans.OutputReaders: Cyclical
using Oceananigans.Advection: div_Uc
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using Statistics
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2
using Printf
using CairoMakie
using NonlinearSolve
using SpeedMapping
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings

parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
mkpath(outputdir)
save_grid = true

# TODO: Maybe I should only use the supergrid for the locations
# of center/face/corner points but otherwise use the "standard"
# grid metrics available from the MOM outputs?
# -> No, instead I should use the same inputs! I think that is what I am doing now,
# but needs to be checked. I think best would be to read the input file location
# from the config file.

include("tripolargrid_reader.jl")

################################################################################
################################################################################
################################################################################

@info "Horizontal supergrid"

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
################################################################################
################################################################################

@info "Vertical grid"

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

if save_grid
    @info "Saving grid"
    code_to_reconstruct_the_grid = """
        gd = load(grid_file) # gd for grid Dict
        underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
            arch,
            gd["Nx"], gd["Ny"], gd["Nz"],
            gd["Hx"], gd["Hy"], gd["Hz"],
            convert(FT, gd["Lz"]),
            on_architecture(arch, map(FT, gd["λᶜᶜᵃ"])),
            on_architecture(arch, map(FT, gd["λᶠᶜᵃ"])),
            on_architecture(arch, map(FT, gd["λᶜᶠᵃ"])),
            on_architecture(arch, map(FT, gd["λᶠᶠᵃ"])),
            on_architecture(arch, map(FT, gd["φᶜᶜᵃ"])),
            on_architecture(arch, map(FT, gd["φᶠᶜᵃ"])),
            on_architecture(arch, map(FT, gd["φᶜᶠᵃ"])),
            on_architecture(arch, map(FT, gd["φᶠᶠᵃ"])),
            on_architecture(arch, gd["z"]),
            on_architecture(arch, map(FT, gd["Δxᶜᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Δxᶠᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Δxᶜᶠᵃ"])),
            on_architecture(arch, map(FT, gd["Δxᶠᶠᵃ"])),
            on_architecture(arch, map(FT, gd["Δyᶜᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Δyᶠᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Δyᶜᶠᵃ"])),
            on_architecture(arch, map(FT, gd["Δyᶠᶠᵃ"])),
            on_architecture(arch, map(FT, gd["Azᶜᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Azᶠᶜᵃ"])),
            on_architecture(arch, map(FT, gd["Azᶜᶠᵃ"])),
            on_architecture(arch, map(FT, gd["Azᶠᶠᵃ"])),
            convert(FT, gd["radius"]),
            # TODO: this mapping to Tripolar should be replaced with a custom one
            Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"])
        )
        grid = ImmersedBoundaryGrid(
            underlying_grid, PartialCellBottom(gd["bottom"]);
            active_cells_map = true,
            active_z_columns = true,
        )
    """
    save(
        joinpath(outputdir, "$(parentmodel)_grid.jld2"),
        Dict(
            "Note" => "This file was created by Benoit Pasquier (2026) from work in progress and thus comes with zero guarantees!",
            "Nx" => underlying_grid.Nx,
            "Ny" => underlying_grid.Ny,
            "Nz" => underlying_grid.Nz,
            "Hx" => underlying_grid.Hx,
            "Hy" => underlying_grid.Hy,
            "Hz" => underlying_grid.Hz,
            "Lz" => underlying_grid.Lz,
            "λᶜᶜᵃ" => underlying_grid.λᶜᶜᵃ,
            "λᶠᶜᵃ" => underlying_grid.λᶠᶜᵃ,
            "λᶜᶠᵃ" => underlying_grid.λᶜᶠᵃ,
            "λᶠᶠᵃ" => underlying_grid.λᶠᶠᵃ,
            "φᶜᶜᵃ" => underlying_grid.φᶜᶜᵃ,
            "φᶠᶜᵃ" => underlying_grid.φᶠᶜᵃ,
            "φᶜᶠᵃ" => underlying_grid.φᶜᶠᵃ,
            "φᶠᶠᵃ" => underlying_grid.φᶠᶠᵃ,
            "Δxᶜᶜᵃ" => underlying_grid.Δxᶜᶜᵃ,
            "Δxᶠᶜᵃ" => underlying_grid.Δxᶠᶜᵃ,
            "Δxᶜᶠᵃ" => underlying_grid.Δxᶜᶠᵃ,
            "Δxᶠᶠᵃ" => underlying_grid.Δxᶠᶠᵃ,
            "Δyᶜᶜᵃ" => underlying_grid.Δyᶜᶜᵃ,
            "Δyᶠᶜᵃ" => underlying_grid.Δyᶠᶜᵃ,
            "Δyᶜᶠᵃ" => underlying_grid.Δyᶜᶠᵃ,
            "Δyᶠᶠᵃ" => underlying_grid.Δyᶠᶠᵃ,
            "Azᶜᶜᵃ" => underlying_grid.Azᶜᶜᵃ,
            "Azᶠᶜᵃ" => underlying_grid.Azᶠᶜᵃ,
            "Azᶜᶠᵃ" => underlying_grid.Azᶜᶠᵃ,
            "Azᶠᶠᵃ" => underlying_grid.Azᶠᶠᵃ,
            "z" => underlying_grid.z,
            "bottom" => bottom,
            "radius" => underlying_grid.radius,
            "north_poles_latitude" => underlying_grid.conformal_mapping.north_poles_latitude,
            "first_pole_longitude" => underlying_grid.conformal_mapping.first_pole_longitude,
            "southernmost_latitude" => underlying_grid.conformal_mapping.southernmost_latitude,
            "code_to_reconstruct_the_grid" => code_to_reconstruct_the_grid,
        )
    )

end

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
save(joinpath(outputdir, "bottom_height_heatmap_$(typeof(arch)).png"), fig)

################################################################################
################################################################################
################################################################################

@info "Velocities"

# TODO: Probably factor this out into separate setup file,
# to be run once to sabe to JLD2 or something else,
# And then just load it here.
# just in case it fills up all the GPU memory (not sure it does)
# TODO: Figure out the best way to load the data for performance (IO).

u_ds = open_dataset(joinpath(inputdir, "u_periodic.nc"))
v_ds = open_dataset(joinpath(inputdir, "v_periodic.nc"))

# TODO: Comment/uncomment below. Setting times as 12 days for testing only.
# prescribed_Δt = 1month
prescribed_Δt = 1day
times = ((1:12) .- 0.5) * prescribed_Δt

u_ts = FieldTimeSeries{Face, Center, Center}(grid, times; time_indexing = Cyclical(1year))
v_ts = FieldTimeSeries{Center, Face, Center}(grid, times; time_indexing = Cyclical(1year))
w_ts = FieldTimeSeries{Center, Center, Face}(grid, times; time_indexing = Cyclical(1year))

print("month ")
for month in 1:12
    print("$month, ")

    u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
    v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)

    # Place u and v data on Oceananigans B-grid
    u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)

    plottable_u = make_plottable_array(u_Bgrid)
    plottable_v = make_plottable_array(v_Bgrid)
    # for k in 1:50
    for k in 25:25
        local fig = Figure(size = (1200, 1200))
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
        save(joinpath(outputdir, "velocities/BGrid_velocities_$(k)_month$(month)_$(typeof(arch)).png"), fig)
    end

    # Then interpolate to C-grid
    u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)

    plottable_u = make_plottable_array(u)
    plottable_v = make_plottable_array(v)
    # for k in 1:50
    for k in 25:25
        local fig = Figure(size = (1200, 1200))
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
        save(joinpath(outputdir, "velocities/CGrid_velocities_$(k)_month$(month)_$(typeof(arch)).png"), fig)
    end

    # Then compute w from continuity
    w = Field{Center, Center, Face}(grid)
    mask_immersed_field!(u, 0.0)
    mask_immersed_field!(v, 0.0)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    velocities = (u, v, w)
    HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, grid)
    u, v, w = velocities

    plottable_u = make_plottable_array(u)
    plottable_v = make_plottable_array(v)
    plottable_w = make_plottable_array(w)
    # for k in 1:50
    for k in 25:25
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
        save(joinpath(outputdir, "velocities/CGrid_velocities_final_k$(k)_month$(month)_$(typeof(arch)).png"), fig)
    end

    # u2, v2, w2 = deepcopy(u), deepcopy(v), deepcopy(w)
    # mask_immersed_field!(v2, NaN)
    # mask_immersed_field!(w2, NaN)
    # mask_immersed_field!(u2, NaN)
    # k = Nz
    # opt = (; colormap = :RdBu, colorrange = (-1, 1), nan_color = (:black, 1))
    # fig, ax, plt = heatmap(view(u2.data, 1:Nx, 1:Ny, k); opt..., axis = (; title = "u at k = $k"))
    # plt2 = heatmap(fig[2, 1], view(u2.data, 1:Nx, 1:Ny, k - 1); opt..., axis = (; title = "u at k = $(k - 1)"))
    # Label(fig[0, 1], "Near surface u (black = NaNs)", tellwidth = false)
    # save(joinpath(outputdir, "velocities/surface_u_heatmap_$(typeof(arch)).png"), fig)

    set!(u_ts, u, month)
    set!(v_ts, v, month)
    set!(w_ts, w, month)

end
println("Done!")

velocities = PrescribedVelocityFields(; u = u_ts, v = v_ts, w = w_ts)

# TODO: Below was attemps at a callback before GLW PR for prescribed time series.
# Can probably already delete.
#
# month = 1
#
# function get_monthly_velocities(grid, month)
#     u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
#     v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)
#     u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)
#     # Then interpolate to C-grid
#     u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)
#     # Then compute w from continuity
#     w = Field{Center, Center, Face}(grid)
#     mask_immersed_field!(u, 0.0)
#     mask_immersed_field!(v, 0.0)
#     fill_halo_regions!(u)
#     fill_halo_regions!(v)
#     velocities = (u, v, w)
#     HydrostaticFreeSurfaceModels.compute_w_from_continuity!(velocities, grid)
#     u, v, w = velocities
#     return PrescribedVelocityFields(; u, v, w)
# end
# velocities = get_monthly_velocities(grid, 1)
# # Add callback to update velocities
# function update_monthly_mean_velocities(sim)
#     current_time = time(sim)
#     current_month = current_time \div month
#     last_time = current_time - sim.Δt
#     velocities = get_monthly_velocities(grid, 1)
# add_callback!(sim, update_monthly_mean_velocities, IterationInterval(1), name = :update_velocities)

################################################################################
################################################################################
################################################################################

@info "Diffusion"

# TODO: Try to match ACCESS-OM2 as much as possible

κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# κVField_ts = FieldTimeSeries{Center, Center, Center}(grid, times)

# # Load MLD to add strong vertical diffusion in the mixed layer
# mld_ds = open_dataset(joinpath(inputdir, "mld_periodic.nc"))

# print("month ")
# for month in 1:12
#     print("$month, ")

#     mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld[month = At(month)]).data, NaN => 0.0))
#     z_center = znodes(grid, Center(), Center(), Center())
#     is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
#     κVField = CenterField(grid)
#     set!(κVField, κVML * is_mld + κVBG * .!is_mld)

#     # fig, ax, plt = heatmap(
#     #     make_plottable_array(κVField)[:, :, 10];
#     #     colorscale = log10,
#     #     colormap = :viridis,
#     #     axis = (; title = "Vertical diffusivity at k = 10"),
#     #     colorrange = (1e-5, 3e-1),
#     #     lowclip = :red,
#     #     highclip = :cyan,
#     # )
#     # Colorbar(fig[1, 2], plt, ticks = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
#     # save(joinpath(outputdir, "vertical_diffusivity_k10_month$(month)_$(typeof(arch)).png"), fig)

#     set!(κVField_ts, κVField, month)
# end
# println("Done!")

# TODO: Implement FieldTimeSeries support for diffusivity closures?
# Load MLD to add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

implicit_vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization(); # <- TODO: Check if needed (I think this is the default)
    # κ = κVField_ts, # <- need FieldTimeSeries support for that
    κ = κVField
)
explicit_vertical_diffusion = VerticalScalarDiffusivity(
    ExplicitTimeDiscretization();
    # κ = κVField_ts, # <- need FieldTimeSeries support for that
    κ = κVField
)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
# TODO: Try GM + Redi diffusion
# horizontal_diffusion = IsopycnalSkewSymmetricDiffusivity(
#     κ_skew = 300,
#     κ_symmetric = 300,
# )

# Combine them all into a single diffusion closure
# TODO: Remove mixed_layer_diffusion if 0.1°?
closure = (
    horizontal_diffusion,
    implicit_vertical_diffusion,
)
explicit_closure = (
    horizontal_diffusion,
    explicit_vertical_diffusion,
)
################################################################################
################################################################################
################################################################################

@info "Model"

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

@inline age_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)
@inline linear_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, 0.0)
# @inline age_jvp_source(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, 0.0)
# @inline age_source_sink(i, j, k, grid, clock, fields, params) = params.source_rate

age_dynamics = Forcing(
    age_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)
linear_dynamics = Forcing(
    linear_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)
# age_jvp_dynamics = Forcing(
#     age_jvp_source,
#     parameters = age_parameters,
#     discrete_form = true,
# )
forcing = (
    age = age_dynamics,
)
linear_forcing = (
    c = linear_dynamics,
)

# Tracer for autodiff tracing
ADtracer0 = CenterField(grid, Real)
age0 = CenterField(grid)

# For building the Jacobian via autodiff I need to create a second model
# with linear forcings and explicit diffusion. So here I use common kwargs
# to make sure they share all the other pieces.
model_common_kwargs = (
    tracer_advection = WENO(order = 5),
    # tracer_advection = UpwindBiased(),
    # timestepper = :SplitRungeKutta3, # <- to try and improve numerical stability over AB2
    velocities = velocities,
    buoyancy = nothing,
)
model_kwargs = (;
    model_common_kwargs...,
    tracers = (age = age0),
    closure = closure,
    forcing = forcing,
    )
jacobian_model_kwargs = (
    model_common_kwargs...,
    tracers = (ADtracer = ADtracer0),
    closure = explicit_closure,
    forcing = linear_forcing,
)

model = HydrostaticFreeSurfaceModel(grid; model_kwargs...)

################################################################################
################################################################################
################################################################################

@info "Initial condition"

# set!(model, age = Returns(0.0), age_jvp = Returns(0.0)) # TODO: Unneccessary as fields are initialized to zero by default.
set!(model, age = Returns(0.0)) # TODO: Unneccessary as fields are initialized to zero by default.
# fill_halo_regions!(model.tracers.age)

fig, ax, plt = heatmap(
    make_plottable_array(model.tracers.age)[:, :, Nz];
    colormap = :viridis,
    axis = (; title = "Initial age at surface (years)"),
)
Colorbar(fig[1, 2], plt)
save(joinpath(outputdir, "initial_age_surface_$(typeof(arch)).png"), fig)

################################################################################
################################################################################
################################################################################

@info "Simulation"

stop_time = 12 * prescribed_Δt

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx = findmax(adapt(Array, sim.model.tracers.age)) # in years
    mean_age = mean(adapt(Array, sim.model.tracers.age))
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        # "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d) wall time: %s\n",
        # iteration(sim), time(sim), sim.Δt, max_age, idx.I..., walltime
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age)/time = %.1e at (%d, %d, %d), mean(age) = %.1e, wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age / (time(sim) / year), idx.I..., mean_age, walltime
    )
end

# add_callback!(simulation, progress_message, TimeInterval(1year))
add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))
# add_callback!(simulation, zero_age_callback, IterationInterval(1))


output_fields = Dict(
    "age" => model.tracers.age,
    # "u" => model.velocities.u,
)

output_prefix = joinpath(outputdir, "offline_age_$(parentmodel)_$(typeof(arch))")

# simulation.output_writers[:fields] = NetCDFWriter(
simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(prescribed_Δt),
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
################################################################################
################################################################################

@info "Plotting"

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
    save(joinpath(outputdir, "final_age_k$(k)_$(parentmodel)_$(typeof(arch)).png"), fig)
end


# age_lazy = open_dataset(simulation.output_writers[:fields].filepath)["age"]
age_lazy = FieldTimeSeries(simulation.output_writers[:fields].filepath, "age")
# u_lazy = FieldTimeSeries(simulation.output_writers[:fields].filepath, "u")
times = age_lazy.times

set_theme!(Theme(fontsize = 30))

# fig = Figure(size = (1200, 1200))
fig = Figure(size = (1200, 600))

n = Observable(1)
k = 25
agetitle = @lift "age and u on offline OM2 at k = $k, t = " * prettytime(times[$n])
# utitle = @lift "u at k = $k, t = " * prettytime(times[$n])

# agekₙ = @lift readcubedata(age_lazy[At(k = $k, times = [$n])]) # in years
agekₙ = @lift make_plottable_array(age_lazy[$n])[:, :, k] # in years
# ukₙ = @lift make_plottable_array(u_lazy[$n])[:, :, k] # in m/s

ax = fig[1, 1] = Axis(
    fig,
    xlabel = "longitude index",
    ylabel = "latitude index",
    title = agetitle,
)

hm = heatmap!(
    ax, agekₙ;
    colorrange = (0, stop_time / year),
    # extendhigh = auto,
    # extendlow = auto,
    # colorscale = SymLog(0.01),
    colormap = :viridis,
    nan_color = (:black, 1),
)
Colorbar(fig[1, 2], hm)

# ax = fig[2, 1] = Axis(
#     fig,
#     xlabel = "longitude index",
#     ylabel = "latitude index",
#     title = utitle,
# )

# hm = heatmap!(
#     ax, ukₙ;
#     colorrange = (-0.1, 0.1),
#     colormap = :RdBu,
#     nan_color = (:black, 1),
# )
# Colorbar(fig[2, 2], hm)

frames = 1:length(times)

@info "Making an animation..."

Makie.record(fig, joinpath(outputdir, "offline_age_OM2_test_$(typeof(arch)).mp4"), frames, framerate = 25) do i
    println("frame $i/$(length(frames))")
    n[] = i
end

################################################################################
################################################################################
################################################################################

@info "Build matrix"

jacobian_model = HydrostaticFreeSurfaceModel(grid; jacobian_model_kwargs...)

@warn "Adding newton_div method to allow sparsity tracer to pass through WENO"
ADTypes = Union{SparseConnectivityTracer.AbstractTracer, SparseConnectivityTracer.Dual, ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b) where {FT <: ADTypes} = a / b

J = let

    @info "Functions to get vector of tendencies"

    Nx′, Ny′, Nz′ = size(ADtracer0)
    N = Nx′ * Ny′ * Nz′
    fNaN = CenterField(grid)
    mask_immersed_field!(fNaN, NaN)
    idx = findall(!isnan, interior(fNaN))
    Nidx = length(idx)
    @show N, Nidx
    c0 = ones(Nidx)
    ADtracer3D = zeros(Real, Nx′, Ny′, Nz′)
    ADtracer_advection = jacobian_model.advection[:ADtracer]
    total_velocities = jacobian_model.transport_velocities
    kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
    active_cells_map = get_active_cells_map(grid, Val(:interior))

    # TODO: Check if I really need to rewrite these. Not sure why but I think I had to.
    # (These are copy-pasta from Oceananigans.)
    @kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
        i, j, k = @index(Global, NTuple)
        @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
    end

    function mytendency(ADtracer, clock)
        ADtracer3D[idx] .= ADtracer
        set!(model, ADtracer = ADtracer3D)
        ADtracer_tendency = CenterField(grid, Real)

        ADtracer_advection = model.advection[:ADtracer]
        ADtracer_forcing = model.forcing[:ADtracer]
        ADtracer_immersed_bc = immersed_boundary_condition(model.tracers[:ADtracer])

        args = tuple(
            Val(1),
            Val(:ADtracer),
            ADtracer_advection,
            model.closure,
            ADtracer_immersed_bc,
            model.buoyancy,
            model.biogeochemistry,
            model.transport_velocities,
            model.free_surface,
            model.tracers,
            model.closure_fields,
            model.auxiliary_fields,
            clock,
            ADtracer_forcing
        )

        launch!(
            CPU(), grid, kernel_parameters,
            compute_hydrostatic_free_surface_Gc!,
            ADtracer_tendency,
            grid,
            args;
            active_cells_map
        )

        return interior(ADtracer_tendency)[idx]
    end

    @info "Autodiff setup"

    sparse_forward_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )

    @info "Prepare sparsity pattern the Jacobian"
    prep = prepare_jacobian(mytendency, sparse_forward_backend, ADtracer0, times[1])

    @info "Compute the Jacobian"
    J = 1 / 12 * mapreduce(+, eachindex(times)) do i
        @info "month $i"
        jacobian(mytendency, prep, sparse_forward_backend, ADtracer0, times[i])
    end

    J
end

foo

################################################################################
################################################################################
################################################################################

@info "Periodic-state solver"

function G!(dage, age, p) # SciML syntax
    @show "calling G!"
    model = simulation.model
    reset!(simulation)
    simulation.stop_time = 12 * prescribed_Δt
    (Nx, Ny, Nz) = size(model.tracers.age)
    age3D = zeros(Nx, Ny, Nz)
    age3D[wet3D_CPU] .= age
    age3D_arch = on_architecture(arch, age3D)
    @show typeof(age3D_arch), size(age3D_arch)
    set!(model, age = age3D_arch)
    # model.tracers.age.data .= age3D_arch
    run!(simulation)
    # dage .= model.tracers.age.data.parent
    dage .= adapt(Array, interior(model.tracers.age))[wet3D_CPU]
    dage .-= age
    @show extrema(dage)
    return dage
end

age0 = CenterField(grid)
mask_immersed_field!(age0, NaN)
age0_interior_CPU = on_architecture(CPU(), interior(age0))
wet3D_CPU = findall(.!isnan.(age0_interior_CPU))
age0_vec = zeros(length(wet3D_CPU))

f! = NonlinearFunction(G!)
nonlinearprob! = NonlinearProblem(f!, age0_vec, [])

# @time sol = solve(nonlinearprob, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs)), verbose = true, reltol=1e-10, abstol=Inf);
# @time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-12)); show_trace = Val(true), reltol = Inf, abstol = 1.0e-10norm(u0, Inf));
# @time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(rtol = 1.0e-12), jvp_autodiff = AutoFiniteDiff()); show_trace = Val(true), reltol = Inf, abstol = 1.0);
@time sol! = solve(nonlinearprob!, SpeedMappingJL(); show_trace = Val(true), reltol = Inf, abstol = 0.001 * 12 * prescribed_Δt / year, verbose = true);

# TODO: Add the matrix precondioner by usnig the matrix-generation code.
# Need to use my branch/fork so that it works for WENO!
