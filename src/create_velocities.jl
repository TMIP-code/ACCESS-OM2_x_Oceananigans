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

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")
grid = load_tripolar_grid(grid_file)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Functions + kernels for creating velocities
################################################################################

@kernel function compute_Bgrid_velocity_from_MOM_output!(
        u, v, Nx, Ny, Nz, # (Face, Face) u and v fields on Oceananigans
        u_data, v_data    # B-grid u and v from MOM
    )

    i, j, k = @index(Global, NTuple)

    # The MOM B-grid places u and V at NE corners of the cells,
    # while Oceananigans places them at SW corners.
    # So we need to shift the data by one index in both i and j.
    # That means we need to wrap around the i index (periodic longitude),
    # and set j = 1 row to zero (both u and v).
    # Also, MOM vertical coordinate is flipped compared to Oceananigans,
    # so we need to flip that as well.

    𝑖 = mod1(i - 1, Nx)
    𝑗 = max(j - 1, 1)
    zero_first_row = ifelse(j == 1, 0.0, 1.0)
    𝑘 = Nz - k + 1 # flip vertical

    u[i, j, k] = zero_first_row * u_data[𝑖, 𝑗, 𝑘]
    v[i, j, k] = zero_first_row * v_data[𝑖, 𝑗, 𝑘]
end

"""
Places u or v data on the Oceananigans B-grid from MOM output.

It shifts the data from the NE corners (MOM convention)
to the SW corners (Oceananigans convention).
It also flips the vertical coordinate.
j = 1 row is set to zero (both u and v).
i = 1 column is set by wrapping around the data (periodic longitude).
"""
function Bgrid_velocity_from_MOM_output(grid, u_data, v_data)
    # north_bc = Oceananigans.BoundaryCondition(Oceananigans.BoundaryConditions.Zipper{FPivot}(), -1)
    north = FPivotZipperBoundaryCondition(-1)

    loc = (Face(), Face(), Center())
    boundary_conditions = FieldBoundaryConditions(grid, loc; north)

    u = Field(loc, grid; boundary_conditions)
    v = Field(loc, grid; boundary_conditions)

    Nx, Ny, Nz = size(grid)

    kp = KernelParameters(1:Nx, 1:Ny, 1:Nz)

    arch = architecture(grid)

    launch!(arch, grid, kp, compute_Bgrid_velocity_from_MOM_output!,
        u, v, Nx, Ny, Nz, # (Face, Face) u and v fields on Oceananigans
        on_architecture(arch, u_data), on_architecture(arch, v_data)    # B-grid u and v from MOM
    )

    Oceananigans.BoundaryConditions.fill_halo_regions!(u)
    Oceananigans.BoundaryConditions.fill_halo_regions!(v)

    return u, v
end

function interpolate_velocities_from_Bgrid_to_Cgrid(grid, uFF, vFF)

    north = FPivotZipperBoundaryCondition(-1)

    ubcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north)
    vbcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north)

    u = XFaceField(grid; boundary_conditions = ubcs)
    v = YFaceField(grid; boundary_conditions = vbcs)

    interp_u = @at (Face, Center, Center) 1 * uFF
    interp_v = @at (Center, Face, Center) 1 * vFF

    u .= interp_u
    v .= interp_v

    return u, v
end




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
η_ts = FieldTimeSeries{Center, Center, Nothing}(grid, fts_times; backend = OnDisk(), path = η_file, name = "η", time_indexing = Cyclical(stop_time), indices=(:, :, Nz:Nz))

print("month ")
for month in 1:12
    print("$month, ")

    # Do η first because it's smaller and thus faster
    η_data = replace(readcubedata(eta_ds.eta_t[month = At(month)]).data, NaN => 0.0)
    # For sea surface height use a ReducedField (Nothing in the z direction otherwise segfaults)
    η = Field{Center, Center, Nothing}(grid, indices=(:, :, Nz))
    set!(η, η_data)

    # Check if masking immersed fields is needed
    ηold = deepcopy(η)
    mask_immersed_field!(η, 0.0)
    @assert interior(η) == interior(ηold)
    fill_halo_regions!(η)
    set!(η_ts, η, month)

    # Load u and v data
    u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
    v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)
    # Place u and v data on Oceananigans B-grid
    u_Bgrid, v_Bgrid = Bgrid_velocity_from_MOM_output(grid, u_data, v_data)
    # Then interpolate to C-grid
    u, v = interpolate_velocities_from_Bgrid_to_Cgrid(grid, u_Bgrid, v_Bgrid)
    uold = deepcopy(u)
    vold = deepcopy(v)
    mask_immersed_field!(u, 0.0)
    mask_immersed_field!(v, 0.0)
    @assert interior(u) == interior(uold)
    @assert interior(v) == interior(vold)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    set!(u_ts, u, month)
    set!(v_ts, v, month)

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
    plottable_η = view(make_plottable_array(η), :, :, 1)
    maxη = maximum(abs.(plottable_η[.!isnan.(plottable_η)]))
    hm = heatmap!(ax, plottable_η; colormap = :RdBu_9, colorrange = maxη .* (-1, 1), nan_color = :black)
    Colorbar(fig[1, 2], hm)
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

@info("""
Velocities and sea surface height saved to
- $(u_file)
- $(v_file)
- $(η_file)
""")
