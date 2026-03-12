"""
Create and save the velocities and free surface to JLD2 format.

Requires: create_grid.jl to have been run first.

Usage:
    julia --project create_velocities.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.AbstractOperations: grid_metric_operation, Ax, Ay, Az
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: xspacings, yspacings, zspacings
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ
using Oceananigans.OutputReaders: Cyclical, OnDisk
using Oceananigans.Units: days, seconds
year = years = 365.25days
month = months = year / 12

using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2
using TOML
using CairoMakie
using Statistics: mean, quantile

const ρ₀ = 1035.0 # kg/m^3

include("select_architecture.jl")
include("shared_functions.jl")

# Configuration
(; parentmodel) = load_project_config()

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
mkpath(preprocessed_inputs_dir)

make_plots = lowercase(get(ENV, "MAKE_PLOTS", "no")) ∈ ("yes", "true", "1")
@info "Plotting enabled: $make_plots"

if make_plots
    plots_dir = joinpath(preprocessed_inputs_dir, "plots")
    bgrid_u_plot_dir = joinpath(plots_dir, "u")
    bgrid_v_plot_dir = joinpath(plots_dir, "v")
    u_interpolated_plot_dir = joinpath(plots_dir, "u_interpolated")
    v_interpolated_plot_dir = joinpath(plots_dir, "v_interpolated")
    w_plot_dir = joinpath(plots_dir, "w")
    η_plot_dir = joinpath(plots_dir, "eta")
    u_mt_plot_dir = joinpath(plots_dir, "u_from_mass_transport")
    v_mt_plot_dir = joinpath(plots_dir, "v_from_mass_transport")
    w_mt_plot_dir = joinpath(plots_dir, "w_from_mass_transport")
    mkpath(bgrid_u_plot_dir)
    mkpath(bgrid_v_plot_dir)
    mkpath(u_interpolated_plot_dir)
    mkpath(v_interpolated_plot_dir)
    mkpath(w_plot_dir)
    mkpath(η_plot_dir)
    mkpath(u_mt_plot_dir)
    mkpath(v_mt_plot_dir)
    mkpath(w_mt_plot_dir)
end

Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds

################################################################################
# Load grid from JLD2
################################################################################

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Functions + kernels for creating velocities
################################################################################

@kernel function set_kreversed_kernel!(field, data, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = data[i, j, Nz + 1 - k]
end

function set_kreversed!(field, data)
    arch = architecture(field)
    Nx, Ny, Nz = size(data)
    kp = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    launch!(arch, field.grid, kp, set_kreversed_kernel!, field, on_architecture(arch, data), Nz)
    return nothing
end

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

@kernel function shift_transport_to_oceananigans_convention!(tx, ty, Nx, Ny, Nz, tx_data, ty_data)
    i, j, k = @index(Global, NTuple)
    𝑘 = Nz + 1 - k

    @inbounds begin
        tx[i, j, k] = tx_data[mod1(i - 1, Nx), j, 𝑘]
        ty[i, j + 1, k] = ty_data[i, j, 𝑘]
    end
end

function fill_Cgrid_transport_from_MOM_output!(tx, ty, grid, tx_data, ty_data)
    Nx, Ny, Nz = size(grid)
    kp = KernelParameters(1:Nx, 1:(Ny - 1), 1:Nz)

    launch!(
        architecture(grid), grid, kp,
        shift_transport_to_oceananigans_convention!,
        tx, ty, Nx, Ny, Nz,
        on_architecture(architecture(grid), tx_data),
        on_architecture(architecture(grid), ty_data)
    )

    fill_halo_regions!(tx)
    return fill_halo_regions!(ty)
end

@kernel function compute_tz_from_continuity!(tz, grid, tx, ty, Nz)
    i, j = @index(Global, NTuple)

    @inbounds begin
        tz[i, j, 1] = 0.0

        for k in 1:Nz
            horizontal_div = δxᶜᵃᵃ(i, j, k, grid, tx) + δyᵃᶜᵃ(i, j, k, grid, ty)
            tz[i, j, k + 1] = tz[i, j, k] - horizontal_div
        end
    end
end

function fill_continuity_tz_from_tx_ty!(tz, grid, tx, ty)
    Nx, Ny, Nz = size(grid)
    kp = KernelParameters(1:Nx, 1:Ny)

    launch!(architecture(grid), grid, kp, compute_tz_from_continuity!, tz, grid, tx, ty, Nz)
    return fill_halo_regions!(tz)
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

    launch!(
        arch, grid, kp,
        compute_Bgrid_velocity_from_MOM_output!,
        u, v, Nx, Nz,
        on_architecture(arch, u_data), on_architecture(arch, v_data)
    )

    Oceananigans.BoundaryConditions.fill_halo_regions!(u)
    return Oceananigans.BoundaryConditions.fill_halo_regions!(v)
end

function fill_w_from_MOM_output!(w, grid, w_data)
    Nx, Ny, Nz = size(w_data)

    kp = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    arch = architecture(grid)

    launch!(
        arch, grid, kp,
        compute_w_from_MOM_output!,
        w, on_architecture(arch, w_data), Nz
    )

    return Oceananigans.BoundaryConditions.fill_halo_regions!(w)
end

function interpolate_velocities_from_Bgrid_to_Cgrid!(u, v, grid, uFF, vFF, Δxᶠᶠᶜ, Δyᶠᶠᶜ)
    Δzᶠᶠᶜ = zspacings(grid, Face(), Face(), Center())  # lazy, always current

    # Δy * Δz weighted average for the interpolation
    interp_u = (@at (Face, Center, Center) uFF * Δyᶠᶠᶜ * Δzᶠᶠᶜ) / (@at (Face, Center, Center) Δyᶠᶠᶜ * Δzᶠᶠᶜ)
    interp_v = (@at (Center, Face, Center) vFF * Δxᶠᶠᶜ * Δzᶠᶠᶜ) / (@at (Center, Face, Center) Δxᶠᶠᶜ * Δzᶠᶠᶜ)

    u .= interp_u
    v .= interp_v

    fill_halo_regions!(u)
    return fill_halo_regions!(v)
end

function compute_plot_scale(field2D)
    wet_values = field2D[.!isnan.(field2D)]
    isempty(wet_values) && return 1.0

    scale = quantile(abs.(wet_values), 0.9)
    return scale > 0 ? scale : 1.0
end

function save_single_field_plot(field2D, title, filepath)
    fig = Figure(size = (1200, 600))
    ax = Axis(fig[1, 1], title = title)
    scale = compute_plot_scale(field2D)
    hm = heatmap!(ax, field2D; colormap = :RdBu_9, colorrange = scale .* (-1, 1), nan_color = :black)
    Colorbar(fig[1, 2], hm)
    save(filepath, fig)
    return nothing
end

function save_field_k_plot(field, k, title, filepath)
    field2D = view(make_plottable_array(field), :, :, k)
    return save_single_field_plot(field2D, title, filepath)
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
η_ds = open_dataset(joinpath(inputdir, "eta_t_periodic.nc"))
dht_ds = open_dataset(joinpath(inputdir, "dht_periodic.nc"))
tx_ds = open_dataset(joinpath(inputdir, "tx_trans_periodic.nc"))
ty_ds = open_dataset(joinpath(inputdir, "ty_trans_periodic.nc"))
dht_var_name = hasproperty(dht_ds, :dht) ? :dht : error("Could not find variable `dht` in dht_periodic.nc")
wt_var_name = hasproperty(wt_ds, :wt) ? :wt : hasproperty(wt_ds, :w) ? :w : error("Could not find variable `wt` or `w` in wt_periodic.nc")

# prescribed_Δt = 1Δt
prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

u_periodic_file = joinpath(preprocessed_inputs_dir, "u_interpolated_periodic.jld2")
v_periodic_file = joinpath(preprocessed_inputs_dir, "v_interpolated_periodic.jld2")
w_periodic_file = joinpath(preprocessed_inputs_dir, "w_periodic.jld2")
η_periodic_file = joinpath(preprocessed_inputs_dir, "eta_periodic.jld2")
u_mt_periodic_file = joinpath(preprocessed_inputs_dir, "u_from_mass_transport_periodic.jld2")
v_mt_periodic_file = joinpath(preprocessed_inputs_dir, "v_from_mass_transport_periodic.jld2")
w_mt_periodic_file = joinpath(preprocessed_inputs_dir, "w_from_mass_transport_periodic.jld2")

u_constant_file = joinpath(preprocessed_inputs_dir, "u_interpolated_constant.jld2")
v_constant_file = joinpath(preprocessed_inputs_dir, "v_interpolated_constant.jld2")
w_constant_file = joinpath(preprocessed_inputs_dir, "w_constant.jld2")
η_constant_file = joinpath(preprocessed_inputs_dir, "eta_constant.jld2")
u_mt_constant_file = joinpath(preprocessed_inputs_dir, "u_from_mass_transport_constant.jld2")
v_mt_constant_file = joinpath(preprocessed_inputs_dir, "v_from_mass_transport_constant.jld2")
w_mt_constant_file = joinpath(preprocessed_inputs_dir, "w_from_mass_transport_constant.jld2")

# remove old files if they exist
rm(u_periodic_file; force = true)
rm(v_periodic_file; force = true)
rm(w_periodic_file; force = true)
rm(η_periodic_file; force = true)
rm(u_mt_periodic_file; force = true)
rm(v_mt_periodic_file; force = true)
rm(w_mt_periodic_file; force = true)
rm(u_constant_file; force = true)
rm(v_constant_file; force = true)
rm(w_constant_file; force = true)
rm(η_constant_file; force = true)
rm(u_mt_constant_file; force = true)
rm(v_mt_constant_file; force = true)
rm(w_mt_constant_file; force = true)

# Create FieldTimeSeries with OnDisk backend directly to write data as we process it
u_periodic_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_periodic_file, name = "u", time_indexing = Cyclical(stop_time))
v_periodic_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_periodic_file, name = "v", time_indexing = Cyclical(stop_time))
w_periodic_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_periodic_file, name = "w", time_indexing = Cyclical(stop_time))
u_mt_periodic_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_mt_periodic_file, name = "u", time_indexing = Cyclical(stop_time))
v_mt_periodic_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_mt_periodic_file, name = "v", time_indexing = Cyclical(stop_time))
w_mt_periodic_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_mt_periodic_file, name = "w", time_indexing = Cyclical(stop_time))
η_periodic_ts = FieldTimeSeries{Center, Center, Nothing}(grid, fts_times; backend = OnDisk(), path = η_periodic_file, name = "η", time_indexing = Cyclical(stop_time))

println("Grid spacings for B-grid to C-grid interpolation (computed once and reused)")
Δxᶠᶠᶜ = Field(xspacings(grid, Face(), Face(), Center()))
Δyᶠᶠᶜ = Field(yspacings(grid, Face(), Face(), Center()))
compute!(Δxᶠᶠᶜ)
compute!(Δyᶠᶠᶜ)
fill_halo_regions!(Δxᶠᶠᶜ)
fill_halo_regions!(Δyᶠᶠᶜ)

# Pre-allocate fields reused every month to avoid per-month allocations
η = Field{Center, Center, Nothing}(grid)
dht_diag = Field{Center, Center, Center}(grid)
Δzstar = Field(zspacings(grid, Center(), Center(), Center()))

north_ff = FPivotZipperBoundaryCondition(-1)
ff_bcs = FieldBoundaryConditions(grid, (Face(), Face(), Center()); north = north_ff)
u_Bgrid = Field((Face(), Face(), Center()), grid; boundary_conditions = ff_bcs)
v_Bgrid = Field((Face(), Face(), Center()), grid; boundary_conditions = ff_bcs)

ubcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = FPivotZipperBoundaryCondition(-1))
vbcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = FPivotZipperBoundaryCondition(-1))
u = XFaceField(grid; boundary_conditions = ubcs)
v = YFaceField(grid; boundary_conditions = vbcs)

wbcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north = FPivotZipperBoundaryCondition(1))
w = ZFaceField(grid; boundary_conditions = wbcs)

north_t = FPivotZipperBoundaryCondition(-1)
tx_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = north_t)
ty_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = north_t)
tx = XFaceField(grid; boundary_conditions = tx_bcs)
ty = YFaceField(grid; boundary_conditions = ty_bcs)

u_mt = XFaceField(grid; boundary_conditions = ubcs)
v_mt = YFaceField(grid; boundary_conditions = vbcs)
w_mt = ZFaceField(grid; boundary_conditions = wbcs)
tz_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north = FPivotZipperBoundaryCondition(1))
tz = Field{Center, Center, Face}(grid; boundary_conditions = tz_bcs)

Axᶠᶜᶜ = Field(grid_metric_operation((Face, Center, Center), Ax, grid))
Ayᶜᶠᶜ = Field(grid_metric_operation((Center, Face, Center), Ay, grid))
Azᶜᶜᶠ = Field(grid_metric_operation((Center, Center, Face), Az, grid))

# Time-average accumulators (reuse same BCs as the corresponding periodic fields)
u_acc = XFaceField(grid; boundary_conditions = ubcs)
v_acc = YFaceField(grid; boundary_conditions = vbcs)
w_acc = ZFaceField(grid; boundary_conditions = wbcs)
u_mt_acc = XFaceField(grid; boundary_conditions = ubcs)
v_mt_acc = YFaceField(grid; boundary_conditions = vbcs)
w_mt_acc = ZFaceField(grid; boundary_conditions = wbcs)
η_acc = Field{Center, Center, Nothing}(grid)

for month in 1:12
    println("month $month")

    # ── η (set first so _update_zstar_scaling! runs before the dht check) ────
    println("- η")
    η_data = readcubedata(η_ds.eta_t[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, η_data, η_data)
    set!(η, η_data)
    mask_immersed_field!(η, 0.0)
    fill_halo_regions!(η)
    set!(η_periodic_ts, η, month)

    # Update z-star grid scaling (before dht check so Δzstar reflects current η)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η, grid)

    # ── dht consistency check ────────────────────────────────────────────────
    # mask_immersed_field! sets immersed cells to NaN in both fields so that
    # filter(!isnan, ratio) selects only wet cells (Δrᶜᶜᶜ is non-zero for immersed
    # cells in PartialCellBottom grids, so Δzᶜᶜᶜ_data .> 0 would be incorrect).
    println("- dht consistency check")
    dht_data = readcubedata(getproperty(dht_ds, dht_var_name)[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, dht_data, dht_data)
    size(dht_data) == (Nx, Ny - 1, Nz) || error("Unexpected dht monthly shape $(size(dht_data)); expected ($Nx, $(Ny - 1), $Nz)")
    set_kreversed!(dht_diag, dht_data)
    mask_immersed_field!(dht_diag, NaN)

    compute!(Δzstar)                   # Δr * σⁿ with current σ after _update_zstar_scaling!
    mask_immersed_field!(Δzstar, NaN)

    wet_ratio = filter(!isnan, vec(Array(interior(dht_diag)) ./ Array(interior(Δzstar))))
    println("  - dht/Δzstar wet-cell ratio: min=$(minimum(wet_ratio)), mean=$(mean(wet_ratio)), max=$(maximum(wet_ratio))")

    # ── u and v ──────────────────────────────────────────────────────────────
    println("- u and v:")
    println("  - loading from MOM B grid")
    u_data = readcubedata(u_ds.u[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, u_data, u_data)
    v_data = readcubedata(v_ds.v[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, v_data, v_data)
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
    set!(u_periodic_ts, u, month)
    set!(v_periodic_ts, v, month)

    # ── u, v, w from mass transports ────────────────────────────────────────
    println("- u, v, w from mass transports")
    tx_data = readcubedata(tx_ds.tx_trans[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, tx_data, tx_data)
    ty_data = readcubedata(ty_ds.ty_trans[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, ty_data, ty_data)
    fill_Cgrid_transport_from_MOM_output!(tx, ty, grid, tx_data, ty_data)
    mask_immersed_field!(tx, 0.0)
    mask_immersed_field!(ty, 0.0)
    fill_halo_regions!(tx)
    fill_halo_regions!(ty)

    compute!(Axᶠᶜᶜ)
    compute!(Ayᶜᶠᶜ)
    compute!(Azᶜᶜᶠ)

    u_mt .= tx / (ρ₀ * Axᶠᶜᶜ)
    v_mt .= ty / (ρ₀ * Ayᶜᶠᶜ)
    fill_continuity_tz_from_tx_ty!(tz, grid, tx, ty)
    w_mt .= tz / (ρ₀ * Azᶜᶜᶠ)

    mask_immersed_field!(u_mt, 0.0)
    mask_immersed_field!(v_mt, 0.0)
    mask_immersed_field!(w_mt, 0.0)
    fill_halo_regions!(u_mt)
    fill_halo_regions!(v_mt)
    fill_halo_regions!(w_mt)

    set!(u_mt_periodic_ts, u_mt, month)
    set!(v_mt_periodic_ts, v_mt, month)
    set!(w_mt_periodic_ts, w_mt, month)

    # ── w ────────────────────────────────────────────────────────────────────
    println("- w")
    println("  - loading from MOM wt output")
    w_data = readcubedata(getproperty(wt_ds, wt_var_name)[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, w_data, w_data)
    println("  - to Oceananigans C grid")
    fill_w_from_MOM_output!(w, grid, w_data)
    println("  - Masking immersed w")
    mask_immersed_field!(w, 0.0)
    println("  - Filling halo regions for w")
    fill_halo_regions!(w)
    println("  - Set FieldTimeSeries for w")
    set!(w_periodic_ts, w, month)

    if make_plots
        println("  - Plot B grid u and v")
        for k in 25:25
            local u_k_dir = joinpath(bgrid_u_plot_dir, "k$(k)")
            local v_k_dir = joinpath(bgrid_v_plot_dir, "k$(k)")
            mkpath(u_k_dir)
            mkpath(v_k_dir)
            save_field_k_plot(u_Bgrid, k, "B-grid u[k=$k, month=$month]", joinpath(u_k_dir, "u_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(v_Bgrid, k, "B-grid v[k=$k, month=$month]", joinpath(v_k_dir, "v_$(k)_month$(month)_$(arch_str).png"))
        end

        println("  - Plot C grid u and v")
        for k in 25:25
            local ui_k_dir = joinpath(u_interpolated_plot_dir, "k$(k)")
            local vi_k_dir = joinpath(v_interpolated_plot_dir, "k$(k)")
            local umt_k_dir = joinpath(u_mt_plot_dir, "k$(k)")
            local vmt_k_dir = joinpath(v_mt_plot_dir, "k$(k)")
            local w_k_dir = joinpath(w_plot_dir, "k$(k)")
            local wmt_k_dir = joinpath(w_mt_plot_dir, "k$(k)")
            mkpath(ui_k_dir)
            mkpath(vi_k_dir)
            mkpath(umt_k_dir)
            mkpath(vmt_k_dir)
            mkpath(w_k_dir)
            mkpath(wmt_k_dir)

            save_field_k_plot(u, k, "C-grid u[k=$k, month=$month]", joinpath(ui_k_dir, "u_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(v, k, "C-grid v[k=$k, month=$month]", joinpath(vi_k_dir, "v_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(u_mt, k, "C-grid u from mass transports[k=$k, month=$month]", joinpath(umt_k_dir, "u_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(v_mt, k, "C-grid v from mass transports[k=$k, month=$month]", joinpath(vmt_k_dir, "v_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(w, k + 1, "C-grid w[k=$k, month=$month]", joinpath(w_k_dir, "w_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(w_mt, k + 1, "C-grid w from mass transports[k=$k, month=$month]", joinpath(wmt_k_dir, "w_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
        end

        println("- Plot η")
        plottable_η = view(make_plottable_array(η), :, :, 1)
        save_single_field_plot(plottable_η, "sea surface height[month=$month]", joinpath(η_plot_dir, "sea_surface_height_month$(month)_$(arch_str).png"))
    end

    println("- Accumulate into time-average")
    u_acc .+= u
    v_acc .+= v
    w_acc .+= w
    u_mt_acc .+= u_mt
    v_mt_acc .+= v_mt
    w_mt_acc .+= w_mt
    η_acc .+= η

end
println("Done!")

@show u_periodic_ts
@info "saved to $(u_periodic_file)"

@show v_periodic_ts
@info "saved to $(v_periodic_file)"

@show w_periodic_ts
@info "saved to $(w_periodic_file)"

@show u_mt_periodic_ts
@info "saved to $(u_mt_periodic_file)"

@show v_mt_periodic_ts
@info "saved to $(v_mt_periodic_file)"

@show w_mt_periodic_ts
@info "saved to $(w_mt_periodic_file)"

@show η_periodic_ts
@info "saved to $(η_periodic_file)"

@info "Computing time-averaged (constant) fields"
u_acc ./= 12
v_acc ./= 12
w_acc ./= 12
u_mt_acc ./= 12
v_mt_acc ./= 12
w_mt_acc ./= 12
η_acc ./= 12

@info "Filling halo regions for time-averaged (constant) fields"
fill_halo_regions!(u_acc)
fill_halo_regions!(v_acc)
fill_halo_regions!(w_acc)
fill_halo_regions!(u_mt_acc)
fill_halo_regions!(v_mt_acc)
fill_halo_regions!(w_mt_acc)
fill_halo_regions!(η_acc)

@info "Saving time-averaged (constant) fields"
jldsave(u_constant_file; u = Array(interior(u_acc)))
jldsave(v_constant_file; v = Array(interior(v_acc)))
jldsave(w_constant_file; w = Array(interior(w_acc)))
jldsave(u_mt_constant_file; u = Array(interior(u_mt_acc)))
jldsave(v_mt_constant_file; v = Array(interior(v_mt_acc)))
jldsave(w_mt_constant_file; w = Array(interior(w_mt_acc)))
jldsave(η_constant_file; η = Array(interior(η_acc, :, :, 1)))
@info "saved constant u to $(u_constant_file)"
@info "saved constant v to $(v_constant_file)"
@info "saved constant w to $(w_constant_file)"
@info "saved constant u_mt to $(u_mt_constant_file)"
@info "saved constant v_mt to $(v_mt_constant_file)"
@info "saved constant w_mt to $(w_mt_constant_file)"
@info "saved constant η to $(η_constant_file)"
