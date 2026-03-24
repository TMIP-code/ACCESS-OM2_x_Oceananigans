"""
Preprocess closure-related fields (T, S, κV) from ACCESS-OM2 output to JLD2 format.

Creates monthly FieldTimeSeries and yearly averaged Fields for:
  - Temperature (T) and Salinity (S) — for GM-Redi buoyancy
  - Vertical diffusivity (κV) — computed from monthly MLD

Requires: create_grid.jl and periodicaverage.py to have been run first.

Usage:
    julia --project prep_closures.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.OutputReaders: Cyclical, OnDisk
using Oceananigans.Units: days, seconds
year = years = 365.25days
month = months = year / 12

using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2

include("select_architecture.jl")
include("shared_functions.jl")

# Configuration
(; parentmodel, experiment_dir, monthly_dir, yearly_dir) = load_project_config()
mkpath(monthly_dir)
mkpath(yearly_dir)

################################################################################
# Load grid from JLD2
################################################################################

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# z-reversal kernel (MOM stores k=1 at top, Oceananigans at bottom)
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

################################################################################
# Open NetCDF datasets
################################################################################

@info "Reading NetCDF inputs from monthly_dir=$monthly_dir"
flush(stdout); flush(stderr)

@info "  Opening temp_monthly.nc"; flush(stdout); flush(stderr)
temp_ds = open_dataset(joinpath(monthly_dir, "temp_monthly.nc"))
@info "  Opening salt_monthly.nc"; flush(stdout); flush(stderr)
salt_ds = open_dataset(joinpath(monthly_dir, "salt_monthly.nc"))
@info "  Opening mld_monthly.nc (for κV)"; flush(stdout); flush(stderr)
mld_monthly_ds = open_dataset(joinpath(monthly_dir, "mld_monthly.nc"))

@info "All NetCDF datasets opened successfully"
flush(stdout); flush(stderr)

################################################################################
# File paths
################################################################################

# prescribed_Δt = 1Δt
prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

temp_monthly_file = joinpath(monthly_dir, "temp_monthly.jld2")
salt_monthly_file = joinpath(monthly_dir, "salt_monthly.jld2")
κV_monthly_file = joinpath(monthly_dir, "kappa_v_monthly.jld2")
mld_monthly_file = joinpath(monthly_dir, "mld_monthly.jld2")

temp_yearly_file = joinpath(yearly_dir, "temp_yearly.jld2")
salt_yearly_file = joinpath(yearly_dir, "salt_yearly.jld2")

# remove old files if they exist
rm(temp_monthly_file; force = true)
rm(salt_monthly_file; force = true)
rm(κV_monthly_file; force = true)
rm(mld_monthly_file; force = true)
rm(temp_yearly_file; force = true)
rm(salt_yearly_file; force = true)

################################################################################
# Create FieldTimeSeries with OnDisk backend
################################################################################

@info "Creating FieldTimeSeries with OnDisk backend"
flush(stdout); flush(stderr)

temp_monthly_ts = FieldTimeSeries{Center, Center, Center}(grid, fts_times; backend = OnDisk(), path = temp_monthly_file, name = "T", time_indexing = Cyclical(stop_time))
salt_monthly_ts = FieldTimeSeries{Center, Center, Center}(grid, fts_times; backend = OnDisk(), path = salt_monthly_file, name = "S", time_indexing = Cyclical(stop_time))
κV_monthly_ts = FieldTimeSeries{Center, Center, Center}(grid, fts_times; backend = OnDisk(), path = κV_monthly_file, name = "κV", time_indexing = Cyclical(stop_time))
mld_monthly_ts = FieldTimeSeries{Center, Center, Nothing}(grid, fts_times; backend = OnDisk(), path = mld_monthly_file, name = "MLD", time_indexing = Cyclical(stop_time))

################################################################################
# Pre-allocate fields
################################################################################

temp_field = CenterField(grid)
salt_field = CenterField(grid)
temp_acc = CenterField(grid)
salt_acc = CenterField(grid)

mld_field = Field{Center, Center, Nothing}(grid)

# κV parameters (same as setup_model.jl)
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)
z_center = znodes(grid, Center(), Center(), Center())
κV_field = CenterField(grid)

################################################################################
# Month loop
################################################################################

for month in 1:12
    println("month $month")
    flush(stdout)

    # ── T (temperature) ──────────────────────────────────────────────────
    println("- T (temperature)"); flush(stdout)
    temp_data = readcubedata(temp_ds.temp[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, temp_data, temp_data)
    set_kreversed!(temp_field, temp_data)
    mask_immersed_field!(temp_field, 0.0)
    fill_halo_regions!(temp_field)
    set!(temp_monthly_ts, temp_field, month)

    # ── S (salinity) ─────────────────────────────────────────────────────
    println("- S (salinity)"); flush(stdout)
    salt_data = readcubedata(salt_ds.salt[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, salt_data, salt_data)
    set_kreversed!(salt_field, salt_data)
    mask_immersed_field!(salt_field, 0.0)
    fill_halo_regions!(salt_field)
    set!(salt_monthly_ts, salt_field, month)

    # ── MLD (2D field, positive-downward in MOM) ────────────────────────
    println("- MLD"); flush(stdout)
    mld_data = readcubedata(mld_monthly_ds.mld[month = At(month)]).data
    map!(x -> isnan(x) ? zero(x) : x, mld_data, mld_data)
    set!(mld_field, mld_data)
    mask_immersed_field!(mld_field, 0.0)
    fill_halo_regions!(mld_field)
    set!(mld_monthly_ts, mld_field, month)

    # ── κV from MLD (compute 3D diffusivity field from monthly MLD) ─────
    println("- κV from MLD"); flush(stdout)
    mld_data .= .-mld_data  # negate for z-comparison (MLD positive-downward → negative z)
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    set!(κV_field, κVML * is_mld + κVBG * .!is_mld)
    mask_immersed_field!(κV_field, 0.0)
    fill_halo_regions!(κV_field)
    set!(κV_monthly_ts, κV_field, month)

    # ── Accumulate into time-average ─────────────────────────────────────
    println("- Accumulate into time-average"); flush(stdout)
    temp_acc .+= temp_field
    salt_acc .+= salt_field
end
println("Done!")

@show temp_monthly_ts
@info "saved to $(temp_monthly_file)"

@show salt_monthly_ts
@info "saved to $(salt_monthly_file)"

@show κV_monthly_ts
@info "saved to $(κV_monthly_file)"

@show mld_monthly_ts
@info "saved to $(mld_monthly_file)"

################################################################################
# Yearly averaging for T and S
################################################################################

@info "Computing time-averaged (yearly) fields"
temp_acc ./= 12
salt_acc ./= 12

@info "Filling halo regions for time-averaged (yearly) fields"
fill_halo_regions!(temp_acc)
fill_halo_regions!(salt_acc)

@info "Saving time-averaged (yearly) fields"
jldsave(temp_yearly_file; T = Array(interior(temp_acc)))
jldsave(salt_yearly_file; S = Array(interior(salt_acc)))
@info "saved yearly T to $(temp_yearly_file)"
@info "saved yearly S to $(salt_yearly_file)"
@info "saved monthly κV to $(κV_monthly_file)"
