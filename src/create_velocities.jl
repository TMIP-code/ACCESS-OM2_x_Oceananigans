"""
Create and save the velocities and free surface to JLD2 format.

Requires: create_grid.jl to have been run first.

Usage:
    julia --project create_velocities.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode, xspacings, yspacings, zspacings
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.OutputReaders: Cyclical, OnDisk, InMemory
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

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
    arch_str = "GPU"
else
    arch = CPU()
    arch_str = "CPU"
end
@info "Using $arch architecture"

# Configuration
parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
mkpath(outputdir)
mkpath(joinpath(outputdir, "velocities"))
run_mode_tag = get(ENV, "RUN_MODE_TAG", "bgridvelocities_wdiagnosed_etaprescribed")
uv_plot_dir = joinpath(outputdir, "velocities", "uv", run_mode_tag)
w_plot_dir = joinpath(outputdir, "velocities", "w", run_mode_tag)
eta_plot_dir = joinpath(outputdir, "velocities", "eta", run_mode_tag)
mkpath(uv_plot_dir)
mkpath(w_plot_dir)
mkpath(eta_plot_dir)

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
        u, v, Nx, Nz,     # (Face, Face) u and v fields on Oceananigans
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
    @inbounds begin
        u[mod1(i + 1, Nx), j + 1, k] = u_data[i, j, Nz + 1 - k]
        v[mod1(i + 1, Nx), j + 1, k] = v_data[i, j, Nz + 1 - k]
    end
end

@kernel function compute_w_from_MOM_output!(w, w_data, Nz)
    i, j, k = @index(Global, NTuple)
    # The w is on the C grid already but flipped in the vertical
    # and starts at the bottom of the top layer (k = 1)
    @inbounds begin
        w[i, j, Nz + 1 - k] = w_data[i, j, k]
    end
end

"""
Places u or v data on the Oceananigans B-grid from MOM output.

It shifts the data from the NE corners (MOM convention)
to the SW corners (Oceananigans convention).
It also flips the vertical coordinate.
j = 1 row is set to zero (both u and v).
i = 1 column is set by wrapping around the data (periodic longitude).
"""
function fill_Bgrid_velocity_from_MOM_output!(u, v, grid, u_data, v_data)
    Nx, Ny, Nz = size(u_data)
    @assert size(v_data) == (Nx, Ny, Nz)

    kp = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    arch = architecture(grid)

    launch!(arch, grid, kp, compute_Bgrid_velocity_from_MOM_output!,
        u, v, Nx, Nz,
        on_architecture(arch, u_data), on_architecture(arch, v_data))

    Oceananigans.BoundaryConditions.fill_halo_regions!(u)
    Oceananigans.BoundaryConditions.fill_halo_regions!(v)
end

function fill_w_from_MOM_output!(w, grid, w_data)
    Nx, Ny, Nz = size(w_data)

    kp   = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    arch = architecture(grid)

    launch!(arch, grid, kp, compute_w_from_MOM_output!,
        w, on_architecture(arch, w_data), Nz)

    Oceananigans.BoundaryConditions.fill_halo_regions!(w)
end

function interpolate_velocities_from_Bgrid_to_Cgrid!(u, v, grid, uFF, vFF, Δxᶠᶠᶜ, Δyᶠᶠᶜ)
    Δzᶠᶠᶜ = zspacings(grid, Face(), Face(), Center())  # lazy, always current

    # Δy * Δz weighted average for the interpolation
    interp_u = (@at (Face, Center, Center) uFF * Δyᶠᶠᶜ * Δzᶠᶠᶜ) / (@at (Face, Center, Center) Δyᶠᶠᶜ * Δzᶠᶠᶜ)
    interp_v = (@at (Center, Face, Center) vFF * Δxᶠᶠᶜ * Δzᶠᶠᶜ) / (@at (Center, Face, Center) Δxᶠᶠᶜ * Δzᶠᶠᶜ)

    u .= interp_u
    v .= interp_v

    fill_halo_regions!(u)
    fill_halo_regions!(v)
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
wt_ds = open_dataset(joinpath(inputdir, "wt_periodic.nc"))
eta_ds = open_dataset(joinpath(inputdir, "eta_t_periodic.nc"))
dht_ds = open_dataset(joinpath(inputdir, "dht_periodic.nc"))
dht_var_name = hasproperty(dht_ds, :dht) ? :dht : error("Could not find variable `dht` in dht_periodic.nc")
wt_var_name  = hasproperty(wt_ds, :wt) ? :wt : hasproperty(wt_ds, :w) ? :w : error("Could not find variable `wt` or `w` in wt_periodic.nc")

# prescribed_Δt = 1Δt
prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

u_file = joinpath(outputdir, "$(parentmodel)_u_ts.jld2")
v_file = joinpath(outputdir, "$(parentmodel)_v_ts.jld2")
w_file = joinpath(outputdir, "$(parentmodel)_w_ts.jld2")
η_file = joinpath(outputdir, "$(parentmodel)_eta_ts.jld2")

# remove old files if they exist
rm(u_file; force = true)
rm(v_file; force = true)
rm(w_file; force = true)
rm(η_file; force = true)

# Create FieldTimeSeries with OnDisk backend directly to write data as we process it
u_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_file, name = "u", time_indexing = Cyclical(stop_time))
v_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_file, name = "v", time_indexing = Cyclical(stop_time))
w_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_file, name = "w", time_indexing = Cyclical(stop_time))
η_ts = FieldTimeSeries{Center, Center, Nothing}(grid, fts_times; backend = OnDisk(), path = η_file, name = "η", time_indexing = Cyclical(stop_time), indices=(:, :, Nz:Nz))

println("Grid spacings for B-grid to C-grid interpolation (computed once and reused)")
Δxᶠᶠᶜ = Field(xspacings(grid, Face(), Face(), Center()))
Δyᶠᶠᶜ = Field(yspacings(grid, Face(), Face(), Center()))
compute!(Δxᶠᶠᶜ)
compute!(Δyᶠᶠᶜ)
fill_halo_regions!(Δxᶠᶠᶜ)
fill_halo_regions!(Δyᶠᶠᶜ)

# Pre-allocate fields reused every month to avoid per-month allocations
η        = Field{Center, Center, Nothing}(grid, indices=(:, :, 1))
dht_diag = Field{Center, Center, Center}(grid)
Δzstar   = Field(zspacings(grid, Center(), Center(), Center()))

north_ff = FPivotZipperBoundaryCondition(-1)
ff_bcs   = FieldBoundaryConditions(grid, (Face(), Face(), Center()); north=north_ff)
u_Bgrid  = Field((Face(), Face(), Center()), grid; boundary_conditions=ff_bcs)
v_Bgrid  = Field((Face(), Face(), Center()), grid; boundary_conditions=ff_bcs)

ubcs = FieldBoundaryConditions(grid, (Face(),   Center(), Center()); north=FPivotZipperBoundaryCondition(-1))
vbcs = FieldBoundaryConditions(grid, (Center(), Face(),   Center()); north=FPivotZipperBoundaryCondition(-1))
u    = XFaceField(grid; boundary_conditions=ubcs)
v    = YFaceField(grid; boundary_conditions=vbcs)

wbcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north=FPivotZipperBoundaryCondition(1))
w    = ZFaceField(grid; boundary_conditions=wbcs)

for month in 1:12
    println("month $month")

    # ── η (set first so _update_zstar_scaling! runs before the dht check) ────
    println("- η")
    η_data = replace(readcubedata(eta_ds.eta_t[month = At(month)]).data, NaN => 0.0)
    set!(η, η_data)
    mask_immersed_field!(η, 0.0)
    fill_halo_regions!(η)
    set!(η_ts, η, month)

    # Update z-star grid scaling (before dht check so Δzstar reflects current η)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η, grid)

    # ── dht consistency check ────────────────────────────────────────────────
    # mask_immersed_field! sets immersed cells to NaN in both fields so that
    # filter(!isnan, ratio) selects only wet cells (Δrᶜᶜᶜ is non-zero for immersed
    # cells in PartialCellBottom grids, so Δzᶜᶜᶜ_data .> 0 would be incorrect).
    println("- dht consistency check")
    dht_data = replace(readcubedata(getproperty(dht_ds, dht_var_name)[month = At(month)]).data, NaN => 0.0)
    size(dht_data) == (Nx, Ny - 1, Nz) || error("Unexpected dht monthly shape $(size(dht_data)); expected ($Nx, $(Ny-1), $Nz)")
    dht_data = dht_data[:, :, Nz:-1:1]

    set!(dht_diag, dht_data)
    mask_immersed_field!(dht_diag, NaN)

    compute!(Δzstar)                   # Δr * σⁿ with current σ after _update_zstar_scaling!
    mask_immersed_field!(Δzstar, NaN)

    wet_ratio = filter(!isnan, vec(Array(interior(dht_diag)) ./ Array(interior(Δzstar))))
    println("  - dht/Δzstar wet-cell ratio: min=$(minimum(wet_ratio)), mean=$(mean(wet_ratio)), max=$(maximum(wet_ratio))")

    # ── u and v ──────────────────────────────────────────────────────────────
    println("- u and v:")
    println("  - loading from MOM B grid")
    u_data = replace(readcubedata(u_ds.u[month = At(month)]).data, NaN => 0.0)
    v_data = replace(readcubedata(v_ds.v[month = At(month)]).data, NaN => 0.0)
    println("  - index shift to Oceananigans B grid")
    fill_Bgrid_velocity_from_MOM_output!(u_Bgrid, v_Bgrid, grid, u_data, v_data)
    println("  - Interpolate to Oceananigans C grid")
    interpolate_velocities_from_Bgrid_to_Cgrid!(u, v, grid, u_Bgrid, v_Bgrid, Δxᶠᶠᶜ, Δyᶠᶠᶜ)
    println("  - Masking immersed fields")
    mask_immersed_field!(u, 0.0)
    mask_immersed_field!(v, 0.0)
    println("  - Filling halo regions")
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    println("  - Set FieldTimeSeries for u and v")
    set!(u_ts, u, month)
    set!(v_ts, v, month)

    # ── w ────────────────────────────────────────────────────────────────────
    println("- w")
    println("  - loading from MOM wt output")
    w_data = replace(readcubedata(getproperty(wt_ds, wt_var_name)[month = At(month)]).data, NaN => 0.0)
    println("  - to Oceananigans C grid")
    fill_w_from_MOM_output!(w, grid, w_data)
    println("  - Masking immersed w")
    mask_immersed_field!(w, 0.0)
    println("  - Filling halo regions for w")
    fill_halo_regions!(w)
    println("  - Set FieldTimeSeries for w")
    set!(w_ts, w, month)

    println("  - Plot B grid u and v")
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
        k_dir = joinpath(uv_plot_dir, "k$(k)")
        mkpath(k_dir)
        save(joinpath(k_dir, "BGrid_velocities_$(k)_month$(month)_$(arch_str).png"), fig)
    end

    println("  - Plot C grid u and v")
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
        k_dir = joinpath(uv_plot_dir, "k$(k)")
        mkpath(k_dir)
        save(joinpath(k_dir, "CGrid_velocities_$(k)_month$(month)_$(arch_str).png"), fig)

    end

    println("- Plot η")
    fig = Figure(size = (1200, 600))
    ax = Axis(fig[1, 1], title = "sea surface height[month=$month]")
    plottable_η = view(make_plottable_array(η), :, :, 1)
    maxη = maximum(abs.(plottable_η[.!isnan.(plottable_η)]))
    hm = heatmap!(ax, plottable_η; colormap = :RdBu_9, colorrange = maxη .* (-1, 1), nan_color = :black)
    Colorbar(fig[1, 2], hm)
    save(joinpath(eta_plot_dir, "sea_surface_height_month$(month)_$(arch_str).png"), fig)

    println("- Plot w")
    for k in 25:25
        local fig = Figure(size = (1200, 600))
        local ax = Axis(fig[1, 1], title = "C-grid w[k=$k, month=$month]")
        local velocity2D = view(make_plottable_array(w), :, :, k + 1)
        local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[1, 2], hm)
        k_dir = joinpath(w_plot_dir, "k$(k)")
        mkpath(k_dir)
        save(joinpath(k_dir, "CGrid_w_$(k)_month$(month)_$(arch_str).png"), fig)
    end



end
println("Done!")

@show u_ts
@info "saved to $(u_file)"

@show v_ts
@info "saved to $(v_file)"

@show w_ts
@info "saved to $(w_file)"

@show η_ts
@info "saved to $(η_file)"
