"""
Create and save the velocities and free surface to JLD2 format.

Requires: create_grid.jl to have been run first.

Usage:
    julia --project prep_velocities.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.AbstractOperations: grid_metric_operation, Ax, Ay, Az
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: zspacings
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_scaling!, surface_kernel_parameters
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
(; parentmodel, experiment_dir, monthly_dir, yearly_dir) = load_project_config()
(; VELOCITY_SOURCE) = parse_config_env()
mkpath(monthly_dir)
mkpath(yearly_dir)

build_total_transport = VELOCITY_SOURCE == "totaltransport"
@info "VELOCITY_SOURCE = $VELOCITY_SOURCE (build_total_transport = $build_total_transport)"

dht_check = lowercase(get(ENV, "DHT_CHECK", "no")) ∈ ("yes", "true", "1")
@info "dht/Δzstar sanity check: $dht_check"

make_plots = lowercase(get(ENV, "MAKE_PLOTS", "no")) ∈ ("yes", "true", "1")
@info "Plotting enabled: $make_plots"

if make_plots
    plots_dir = joinpath(experiment_dir, "plots")
    η_plot_dir = joinpath(plots_dir, "eta")
    u_mt_plot_dir = joinpath(plots_dir, "u_from_mass_transport")
    v_mt_plot_dir = joinpath(plots_dir, "v_from_mass_transport")
    w_mt_plot_dir = joinpath(plots_dir, "w_from_mass_transport")
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
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Functions + kernels for creating velocities
################################################################################

# MOM → Oceananigans convention kernels and helpers are in shared_utils/data_loading.jl:
#   set_kreversed!, fill_Cgrid_transport_from_MOM_output!,
#   fill_continuity_tz_from_tx_ty!

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
flush(stdout); flush(stderr)

@info "Reading NetCDF inputs from monthly_dir=$monthly_dir"
flush(stdout); flush(stderr)

@info "  Opening eta_t_monthly.nc"; flush(stdout); flush(stderr)
η_ds = open_dataset(joinpath(monthly_dir, "eta_t_monthly.nc"))
@info "  Opening tx_trans_monthly.nc"; flush(stdout); flush(stderr)
tx_ds = open_dataset(joinpath(monthly_dir, "tx_trans_monthly.nc"))
@info "  Opening ty_trans_monthly.nc"; flush(stdout); flush(stderr)
ty_ds = open_dataset(joinpath(monthly_dir, "ty_trans_monthly.nc"))

if build_total_transport
    tx_gm_path = joinpath(monthly_dir, "tx_trans_gm_monthly.nc")
    ty_gm_path = joinpath(monthly_dir, "ty_trans_gm_monthly.nc")
    isfile(tx_gm_path) && isfile(ty_gm_path) ||
        error("VELOCITY_SOURCE=totaltransport requires tx_trans_gm_monthly.nc and ty_trans_gm_monthly.nc in $monthly_dir")
    @info "  Opening tx_trans_gm_monthly.nc"; flush(stdout); flush(stderr)
    tx_gm_ds = open_dataset(tx_gm_path)
    @info "  Opening ty_trans_gm_monthly.nc"; flush(stdout); flush(stderr)
    ty_gm_ds = open_dataset(ty_gm_path)
end

if dht_check
    @info "  Opening dht_monthly.nc (DHT_CHECK)"; flush(stdout); flush(stderr)
    dht_ds = open_dataset(joinpath(monthly_dir, "dht_monthly.nc"))
    dht_var_name = hasproperty(dht_ds, :dht) ? :dht : error("Could not find variable `dht` in dht_monthly.nc")
end

# prescribed_Δt = 1Δt
prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

η_monthly_file = joinpath(monthly_dir, "eta_monthly.jld2")
u_mt_monthly_file = joinpath(monthly_dir, "u_from_mass_transport_monthly.jld2")
v_mt_monthly_file = joinpath(monthly_dir, "v_from_mass_transport_monthly.jld2")
w_mt_monthly_file = joinpath(monthly_dir, "w_from_mass_transport_monthly.jld2")

η_yearly_file = joinpath(yearly_dir, "eta_yearly.jld2")
u_mt_yearly_file = joinpath(yearly_dir, "u_from_mass_transport_yearly.jld2")
v_mt_yearly_file = joinpath(yearly_dir, "v_from_mass_transport_yearly.jld2")
w_mt_yearly_file = joinpath(yearly_dir, "w_from_mass_transport_yearly.jld2")

u_tt_monthly_file = joinpath(monthly_dir, "u_from_total_transport_monthly.jld2")
v_tt_monthly_file = joinpath(monthly_dir, "v_from_total_transport_monthly.jld2")
w_tt_monthly_file = joinpath(monthly_dir, "w_from_total_transport_monthly.jld2")
u_tt_yearly_file = joinpath(yearly_dir, "u_from_total_transport_yearly.jld2")
v_tt_yearly_file = joinpath(yearly_dir, "v_from_total_transport_yearly.jld2")
w_tt_yearly_file = joinpath(yearly_dir, "w_from_total_transport_yearly.jld2")

# remove old files if they exist
rm(η_monthly_file; force = true)
rm(u_mt_monthly_file; force = true)
rm(v_mt_monthly_file; force = true)
rm(w_mt_monthly_file; force = true)
rm(η_yearly_file; force = true)
rm(u_mt_yearly_file; force = true)
rm(v_mt_yearly_file; force = true)
rm(w_mt_yearly_file; force = true)
if build_total_transport
    rm(u_tt_monthly_file; force = true)
    rm(v_tt_monthly_file; force = true)
    rm(w_tt_monthly_file; force = true)
    rm(u_tt_yearly_file; force = true)
    rm(v_tt_yearly_file; force = true)
    rm(w_tt_yearly_file; force = true)
end

# Boundary conditions for velocity FTS (sign flip at north fold for u/v, no flip for w/η)
ubcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = FPivotZipperBoundaryCondition(-1))
vbcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = FPivotZipperBoundaryCondition(-1))
wbcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north = FPivotZipperBoundaryCondition(1))

# Create FieldTimeSeries with OnDisk backend directly to write data as we process it
u_mt_monthly_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_mt_monthly_file, name = "u", time_indexing = Cyclical(stop_time), boundary_conditions = ubcs)
v_mt_monthly_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_mt_monthly_file, name = "v", time_indexing = Cyclical(stop_time), boundary_conditions = vbcs)
w_mt_monthly_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_mt_monthly_file, name = "w", time_indexing = Cyclical(stop_time), boundary_conditions = wbcs)
η_monthly_ts = FieldTimeSeries{Center, Center, Nothing}(grid, fts_times; backend = OnDisk(), path = η_monthly_file, name = "η", time_indexing = Cyclical(stop_time))

if build_total_transport
    u_tt_monthly_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_tt_monthly_file, name = "u", time_indexing = Cyclical(stop_time), boundary_conditions = ubcs)
    v_tt_monthly_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_tt_monthly_file, name = "v", time_indexing = Cyclical(stop_time), boundary_conditions = vbcs)
    w_tt_monthly_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_tt_monthly_file, name = "w", time_indexing = Cyclical(stop_time), boundary_conditions = wbcs)
end

@info "All NetCDF datasets opened successfully"
flush(stdout); flush(stderr)

@info "Creating FieldTimeSeries with OnDisk backend"
flush(stdout); flush(stderr)

# Pre-allocate fields reused every month to avoid per-month allocations
η = Field{Center, Center, Nothing}(grid)
if dht_check
    dht_diag = Field{Center, Center, Center}(grid)
    Δzstar = Field(zspacings(grid, Center(), Center(), Center()))
end

tx = XFaceField(grid; boundary_conditions = ubcs)
ty = YFaceField(grid; boundary_conditions = vbcs)

u_mt = XFaceField(grid; boundary_conditions = ubcs)
v_mt = YFaceField(grid; boundary_conditions = vbcs)
w_mt = ZFaceField(grid; boundary_conditions = wbcs)
tz = Field{Center, Center, Face}(grid; boundary_conditions = wbcs)

if build_total_transport
    tx_gm = XFaceField(grid; boundary_conditions = ubcs)
    ty_gm = YFaceField(grid; boundary_conditions = vbcs)
    u_tt = XFaceField(grid; boundary_conditions = ubcs)
    v_tt = YFaceField(grid; boundary_conditions = vbcs)
    w_tt = ZFaceField(grid; boundary_conditions = wbcs)
    tz_tt = Field{Center, Center, Face}(grid; boundary_conditions = wbcs)
end

# TODO: Alternatives to precomputed area Fields:
#   1) Use AbstractOperation directly (no Field/compute!; lazy eval per iteration, negligible cost)
#   2) Use kernels with Oceananigans.Operators.Axᶠᶜᶜ/Ayᶜᶠᶜ/Azᶜᶜᶠ (fused, no allocations; more code)
AxFCC = Field(grid_metric_operation((Face, Center, Center), Ax, grid))
AyCFC = Field(grid_metric_operation((Center, Face, Center), Ay, grid))
AzCCF = Field(grid_metric_operation((Center, Center, Face), Az, grid))

# Time-average accumulators (reuse same BCs as the corresponding periodic fields)
u_mt_acc = XFaceField(grid; boundary_conditions = ubcs)
v_mt_acc = YFaceField(grid; boundary_conditions = vbcs)
w_mt_acc = ZFaceField(grid; boundary_conditions = wbcs)
η_acc = Field{Center, Center, Nothing}(grid)
if build_total_transport
    u_tt_acc = XFaceField(grid; boundary_conditions = ubcs)
    v_tt_acc = YFaceField(grid; boundary_conditions = vbcs)
    w_tt_acc = ZFaceField(grid; boundary_conditions = wbcs)
end

for month in 1:12
    println("month $month")
    flush(stdout)

    # ── η (set first so _update_zstar_scaling! runs before the dht check) ────
    println("- η"); flush(stdout)
    η_data = readcubedata(η_ds.eta_t[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, η_data, η_data)
    set!(η, η_data)
    mask_immersed_field!(η, 0.0)
    fill_halo_regions!(η)
    set!(η_monthly_ts, η, month)

    # Update z-star grid scaling (so Δzstar reflects current η for the optional dht check)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η, grid)

    # ── dht consistency check (optional; DHT_CHECK=yes) ──────────────────────
    # mask_immersed_field! sets immersed cells to NaN in both fields so that
    # filter(!isnan, ratio) selects only wet cells (Δrᶜᶜᶜ is non-zero for immersed
    # cells in PartialCellBottom grids, so Δzᶜᶜᶜ_data .> 0 would be incorrect).
    if dht_check
        println("- dht consistency check"); flush(stdout)
        dht_data = readcubedata(getproperty(dht_ds, dht_var_name)[month = At(month)]).data
        map!(x -> isnan(x) ? zero(x) : x, dht_data, dht_data)
        size(dht_data) == (Nx, Ny, Nz) || error("Unexpected dht monthly shape $(size(dht_data)); expected ($Nx, $Ny, $Nz)")
        set_kreversed!(dht_diag, dht_data)
        mask_immersed_field!(dht_diag, NaN)

        compute!(Δzstar)                   # Δr * σⁿ with current σ after _update_zstar_scaling!
        mask_immersed_field!(Δzstar, NaN)

        wet_ratio = filter(!isnan, vec(Array(interior(dht_diag)) ./ Array(interior(Δzstar))))
        println("  - dht/Δzstar wet-cell ratio: min=$(minimum(wet_ratio)), mean=$(mean(wet_ratio)), max=$(maximum(wet_ratio))")
    end

    # ── u, v, w from mass transports ────────────────────────────────────────
    println("- u, v, w from mass transports"); flush(stdout)
    tx_data = readcubedata(tx_ds.tx_trans[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, tx_data, tx_data)
    ty_data = readcubedata(ty_ds.ty_trans[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, ty_data, ty_data)
    fill_Cgrid_transport_from_MOM_output!(tx, ty, grid, tx_data, ty_data)
    mask_immersed_field!(tx, 0.0)
    mask_immersed_field!(ty, 0.0)
    fill_halo_regions!(tx)
    fill_halo_regions!(ty)

    compute!(AxFCC)
    compute!(AyCFC)
    compute!(AzCCF)

    u_mt .= tx / (ρ₀ * AxFCC)
    v_mt .= ty / (ρ₀ * AyCFC)
    fill_continuity_tz_from_tx_ty!(tz, grid, tx, ty)
    w_mt .= tz / (ρ₀ * AzCCF)

    mask_immersed_field!(u_mt, 0.0)
    mask_immersed_field!(v_mt, 0.0)
    mask_immersed_field!(w_mt, 0.0)
    fill_halo_regions!(u_mt)
    fill_halo_regions!(v_mt)
    fill_halo_regions!(w_mt)

    set!(u_mt_monthly_ts, u_mt, month)
    set!(v_mt_monthly_ts, v_mt, month)
    set!(w_mt_monthly_ts, w_mt, month)

    # ── u, v, w from total transports (resolved + GM) ──────────────────
    if build_total_transport
        println("- u, v, w from total transports (resolved + GM)"); flush(stdout)
        tx_gm_data = readcubedata(tx_gm_ds.tx_trans_gm[month = At(month)]).data
        map!(x -> isnan(x) ? zero(x) : x, tx_gm_data, tx_gm_data)
        ty_gm_data = readcubedata(ty_gm_ds.ty_trans_gm[month = At(month)]).data
        map!(x -> isnan(x) ? zero(x) : x, ty_gm_data, ty_gm_data)
        fill_Cgrid_transport_from_MOM_output!(tx_gm, ty_gm, grid, tx_gm_data, ty_gm_data)
        streamfunction_to_perlayer!(tx_gm, grid)
        streamfunction_to_perlayer!(ty_gm, grid)
        mask_immersed_field!(tx_gm, 0.0)
        mask_immersed_field!(ty_gm, 0.0)
        fill_halo_regions!(tx_gm)
        fill_halo_regions!(ty_gm)

        # Velocity from total transport = resolved + GM per-layer
        u_tt .= (tx .+ tx_gm) / (ρ₀ * AxFCC)
        v_tt .= (ty .+ ty_gm) / (ρ₀ * AyCFC)
        # Vertical velocity from continuity of total transport
        tx_gm .+= tx  # tx_gm now holds total tx
        ty_gm .+= ty  # ty_gm now holds total ty
        fill_halo_regions!(tx_gm)
        fill_halo_regions!(ty_gm)
        fill_continuity_tz_from_tx_ty!(tz_tt, grid, tx_gm, ty_gm)
        w_tt .= tz_tt / (ρ₀ * AzCCF)

        mask_immersed_field!(u_tt, 0.0)
        mask_immersed_field!(v_tt, 0.0)
        mask_immersed_field!(w_tt, 0.0)
        fill_halo_regions!(u_tt)
        fill_halo_regions!(v_tt)
        fill_halo_regions!(w_tt)

        set!(u_tt_monthly_ts, u_tt, month)
        set!(v_tt_monthly_ts, v_tt, month)
        set!(w_tt_monthly_ts, w_tt, month)
    end

    if make_plots
        println("  - Plot C grid u_mt, v_mt, w_mt")
        for k in 25:25
            local umt_k_dir = joinpath(u_mt_plot_dir, "k$(k)")
            local vmt_k_dir = joinpath(v_mt_plot_dir, "k$(k)")
            local wmt_k_dir = joinpath(w_mt_plot_dir, "k$(k)")
            mkpath(umt_k_dir)
            mkpath(vmt_k_dir)
            mkpath(wmt_k_dir)

            save_field_k_plot(u_mt, k, "C-grid u from mass transports[k=$k, month=$month]", joinpath(umt_k_dir, "u_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(v_mt, k, "C-grid v from mass transports[k=$k, month=$month]", joinpath(vmt_k_dir, "v_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
            save_field_k_plot(w_mt, k + 1, "C-grid w from mass transports[k=$k, month=$month]", joinpath(wmt_k_dir, "w_from_mass_transport_$(k)_month$(month)_$(arch_str).png"))
        end

        println("- Plot η")
        plottable_η = view(make_plottable_array(η), :, :, 1)
        save_single_field_plot(plottable_η, "sea surface height[month=$month]", joinpath(η_plot_dir, "sea_surface_height_month$(month)_$(arch_str).png"))
    end

    println("- Accumulate into time-average"); flush(stdout)
    u_mt_acc .+= u_mt
    v_mt_acc .+= v_mt
    w_mt_acc .+= w_mt
    η_acc .+= η
    if build_total_transport
        u_tt_acc .+= u_tt
        v_tt_acc .+= v_tt
        w_tt_acc .+= w_tt
    end

end
println("Done!")

@show u_mt_monthly_ts
@info "saved to $(u_mt_monthly_file)"

@show v_mt_monthly_ts
@info "saved to $(v_mt_monthly_file)"

@show w_mt_monthly_ts
@info "saved to $(w_mt_monthly_file)"

@show η_monthly_ts
@info "saved to $(η_monthly_file)"

if build_total_transport
    @show u_tt_monthly_ts
    @info "saved to $(u_tt_monthly_file)"
    @show v_tt_monthly_ts
    @info "saved to $(v_tt_monthly_file)"
    @show w_tt_monthly_ts
    @info "saved to $(w_tt_monthly_file)"
end

@info "Computing time-averaged (yearly) fields"
u_mt_acc ./= 12
v_mt_acc ./= 12
w_mt_acc ./= 12
η_acc ./= 12

@info "Filling halo regions for time-averaged (yearly) fields"
fill_halo_regions!(u_mt_acc)
fill_halo_regions!(v_mt_acc)
fill_halo_regions!(w_mt_acc)
fill_halo_regions!(η_acc)

@info "Saving time-averaged (yearly) fields"
jldsave(u_mt_yearly_file; u = Array(interior(u_mt_acc)))
jldsave(v_mt_yearly_file; v = Array(interior(v_mt_acc)))
jldsave(w_mt_yearly_file; w = Array(interior(w_mt_acc)))
jldsave(η_yearly_file; η = Array(interior(η_acc, :, :, 1)))
@info "saved yearly u_mt to $(u_mt_yearly_file)"
@info "saved yearly v_mt to $(v_mt_yearly_file)"
@info "saved yearly w_mt to $(w_mt_yearly_file)"
@info "saved yearly η to $(η_yearly_file)"

if build_total_transport
    @info "Computing time-averaged (yearly) total transport fields"
    u_tt_acc ./= 12
    v_tt_acc ./= 12
    w_tt_acc ./= 12
    fill_halo_regions!(u_tt_acc)
    fill_halo_regions!(v_tt_acc)
    fill_halo_regions!(w_tt_acc)
    jldsave(u_tt_yearly_file; u = Array(interior(u_tt_acc)))
    jldsave(v_tt_yearly_file; v = Array(interior(v_tt_acc)))
    jldsave(w_tt_yearly_file; w = Array(interior(w_tt_acc)))
    @info "saved yearly u_tt to $(u_tt_yearly_file)"
    @info "saved yearly v_tt to $(v_tt_yearly_file)"
    @info "saved yearly w_tt to $(w_tt_yearly_file)"
end
