"""
Create and save the velocities and free surface to JLD2 format.

Requires: create_grid.jl to have been run first.

Usage:
    julia --project create_velocities.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.OutputReaders: Cyclical, OnDisk, InMemory
using Oceananigans.Units: seconds
using Adapt: adapt
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2
using CairoMakie
using Printf
using Statistics

# Set up architecture
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = GPU()
else
    arch = CPU()
end
@info "Using $arch architecture"

# Configuration
parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
mkpath(outputdir)
mkpath(joinpath(outputdir, "velocities"))

Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds

include("tripolargrid_reader.jl")

################################################################################
# Load grid from JLD2
################################################################################

@info "Loading grid data from JLD2"

FT = Float64
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")

@info "Constructing grid"
# This below is copy-pasted from the saved string in gd
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
    Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"])
)

grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(gd["bottom"]);
    active_cells_map = true,
    active_z_columns = true,
)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Create velocities
################################################################################

@info "Creating velocity and sea surface height field time series"

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

u_ds = open_dataset(joinpath(inputdir, "u_periodic.nc"))
v_ds = open_dataset(joinpath(inputdir, "v_periodic.nc"))
eta_ds = open_dataset(joinpath(inputdir, "eta_t_periodic.nc"))

prescribed_Δt = 1Δt
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

u_file = joinpath(outputdir, "$(parentmodel)_u_ts.jld2")
v_file = joinpath(outputdir, "$(parentmodel)_v_ts.jld2")
η_file = joinpath(outputdir, "$(parentmodel)_eta_ts.jld2")

# remove old files if they exist
rm(u_file; force = true)
rm(v_file; force = true)
rm(η_file; force = true)

# Create FieldTimeSeries with OnDisk backend directly to write data as we process it
u_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_file, name = "u", time_indexing = Cyclical(stop_time))
v_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_file, name = "v", time_indexing = Cyclical(stop_time))
η_ts = FieldTimeSeries{Center, Center, Center}(grid, fts_times; backend = OnDisk(), path = η_file, name = "η", time_indexing = Cyclical(stop_time), indices=(:, :, Nz:Nz))

print("month ")
for month in 1:12
    print("$month, ")

    u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
    v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)
    η_data = replace(readcubedata(eta_ds.eta_t[month = At(month)]).data, NaN => 0.0)

    # For sea surface height just fill the field and halos
    η = CenterField(grid, indices=(:, :, Nz))
    set!(η, η_data)
    fill_halo_regions!(η)

    # Place u and v data on Oceananigans B-grid
    u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)

    # Then interpolate to C-grid
    u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)

    # Mask immersed fields
    # TODO: this should not be needed. If some data is erased with zeros it means there is an issue!
    error("""Fix masking before you any further: You probably don't need it anyway.
             Maybe check that it does not erase anything for u and v,
             But do not mask η at all, as it will silently mess everything up
             as it calls a kernel expecting the full vertical column,
             which ends up with segfaults and malloc errors
             (because of elided bounds checks with @inbounds).
             """)
    @info "masking u"
    mask_immersed_field!(u, 0.0)
    @info "masking v"
    mask_immersed_field!(v, 0.0)
    @info "masking η"
    mask_immersed_field!(η, 0.0)

    # Fill halos
    @info "Filling u"
    fill_halo_regions!(u)
    @info "Filling v"
    fill_halo_regions!(v)
    @info "Filling η"
    fill_halo_regions!(η)

    @info "Setting field time series for month $month"
    set!(u_ts, u, month)
    set!(v_ts, v, month)
    set!(η_ts, η, month)


    # Visualization (for k=25 only, as in original)
    for k in 25:25
        local fig = Figure(size = (1200, 1200))
        local ax = Axis(fig[1, 1], title = "B-grid u[k=$k, month=$month]")
        local velocity2D = view(make_plottable_array(u_Bgrid), :, :, k)
        local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[1, 2], hm)
        ax = Axis(fig[2, 1], title = "B-grid v[k=$k, month=$month]")
        velocity2D = view(make_plottable_array(v_Bgrid), :, :, k)
        maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[2, 2], hm)
        save(joinpath(outputdir, "velocities/BGrid_velocities_$(k)_month$(month)_$(typeof(arch)).png"), fig)
    end

    # Visualization
    for k in 25:25
        local fig = Figure(size = (1200, 1200))
        local ax = Axis(fig[1, 1], title = "C-grid u[k=$k, month=$month]")
        local velocity2D = view(make_plottable_array(u), :, :, k)
        local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[1, 2], hm)
        ax = Axis(fig[2, 1], title = "C-grid v[k=$k, month=$month]")
        velocity2D = view(make_plottable_array(v), :, :, k)
        maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[2, 2], hm)
        save(joinpath(outputdir, "velocities/CGrid_velocities_$(k)_month$(month)_$(typeof(arch)).png"), fig)

    end

    fig = Figure(size = (1200, 600))
    ax = Axis(fig[1, 1], title = "sea surface height[month=$month]")
    @info "plottable_η"
    plottable_η = view(η.data, 1:Nx, 1:Ny, Nz)
    @show typeof(plottable_η)
    @info "maxη"
    maxη = maximum(abs.(plottable_η[.!isnan.(plottable_η)]))
    @info "hm"
    hm = heatmap!(ax, plottable_η; colormap = :RdBu_9, colorrange = maxη .* (-1, 1), nan_color = :black)
    @info "colorbar"
    Colorbar(fig[1, 2], hm)
    @info "savefig"
    save(joinpath(outputdir, "velocities/sea_surface_height_month$(month)_$(typeof(arch)).png"), fig)

    #  TODO move this visualization to after a simulation that outputs u, v, w, and η,
    # to confirm that the interpolated velocities and computed w look reasonable when used
    # in a simulation context (i.e. with the grid's z updated by the free surface height)
    # # Visualization
    # for k in 25:25
    #     local fig = Figure(size = (1200, 1800))
    #     local ax = Axis(fig[1, 1], title = "C-grid u[k=$k, month=$month]")
    #     local velocity2D = view(make_plottable_array(u), :, :, k)
    #     local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[1, 2], hm)
    #     ax = Axis(fig[2, 1], title = "C-grid v[k=$k, month=$month]")
    #     velocity2D = view(make_plottable_array(v), :, :, k)
    #     maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[2, 2], hm)
    #     ax = Axis(fig[3, 1], title = "C-grid w[k=$k, month=$month]")
    #     velocity2D = view(make_plottable_array(w), :, :, k + 1)
    #     maxvelocity = maximum(abs.(velocity2D[.!isnan.(velocity2D)]))
    #     hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
    #     Colorbar(fig[3, 2], hm)
    #     save(joinpath(outputdir, "velocities/CGrid_velocities_final_k$(k)_month$(month)_$(typeof(arch)).png"), fig)
    # end

end
println("Done!")

@info "Velocities and sea surface height saved to $(velocities_file)"
