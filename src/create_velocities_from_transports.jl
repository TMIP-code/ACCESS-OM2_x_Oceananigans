"""
Create and save velocities from mass transports.

Inputs:
- tx_trans_periodic.nc (zonal mass transport, east-face convention)
- ty_trans_periodic.nc (meridional mass transport, north-face convention)
- eta_t_periodic.nc   (free-surface displacement)

Outputs:
- *_ts_from_mass_transports.jld2 FieldTimeSeries files

Requires: create_grid.jl to have been run first.
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.AbstractOperations: grid_metric_operation, Ax, Ay, Az
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, fill_halo_regions!
using Oceananigans.Grids: xspacings, yspacings, zspacings
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ
using Oceananigans.OutputReaders: Cyclical, OnDisk
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using TOML
using JLD2
using CairoMakie
using Statistics: quantile

const ρ₀ = 1035.0 # kg/m^3

parse_env_bool(name, default) = lowercase(strip(get(ENV, name, string(default)))) in ("1", "true", "yes", "on")

const CHECK_FIELD_INTEGRITY = parse_env_bool("CHECK_FIELD_INTEGRITY", true)
@info "CHECK_FIELD_INTEGRITY = $CHECK_FIELD_INTEGRITY"
const ENABLE_VELOCITY_PLOTTING = parse_env_bool("ENABLE_VELOCITY_PLOTTING", true)
@info "ENABLE_VELOCITY_PLOTTING = $ENABLE_VELOCITY_PLOTTING"

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
cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
	ARGS[1]
elseif haskey(ENV, "PARENTMODEL")
	ENV["PARENTMODEL"]
else
	get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end
@info "Parent model: $parentmodel"

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
	@warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
	outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
else
	outputdir = profile["outputdir"]
end
@info "Output directory: $outputdir"

mkpath(outputdir)
mkpath(joinpath(outputdir, "velocities"))
u_plot_dir = joinpath(outputdir, "velocities", "from_mass_transport", "u")
v_plot_dir = joinpath(outputdir, "velocities", "from_mass_transport", "v")
w_plot_dir = joinpath(outputdir, "velocities", "from_mass_transport", "w")
tz_plot_dir = joinpath(outputdir, "velocities", "from_mass_transport", "tz")
mkpath(u_plot_dir)
mkpath(v_plot_dir)
mkpath(w_plot_dir)
mkpath(tz_plot_dir)

include("tripolargrid_reader.jl")

################################################################################
# Load grid
################################################################################

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Functions + kernels
################################################################################

@kernel function shift_transport_to_oceananigans_convention!(tx, ty, Nx, Ny, Nz, tx_data, ty_data)
	i, j, k = @index(Global, NTuple)

	# Flip vertical indexing to match Oceananigans convention
	𝑘 = Nz + 1 - k

    #                       ┏━━━━━━━━━┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┓
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #              j = 3 ─▶ u ────────┼──────── u ────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    # 𝑗 = max(j - 1, 1) = 2 ┃         │         ┃         │         ┃
    #              j = 3 ─▶ ┣━━━━━━━━ v ━━━━━━━━╋━━━━━━━━ v ━━━━━━━━┫
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #              j = 2 ─▶ u ────────┼──────── u ────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    # 𝑗 = max(j - 1, 1) = 1 ┃         │         ┃         │         ┃
    #              j = 2 ─▶ ┣━━━━━━━━ v ━━━━━━━━╋━━━━━━━━ v ━━━━━━━━┫
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #              j = 1 ─▶ u ────────╂──────── u ────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    # 𝑗 = max(j - 1, 1) = 1 ┃         │         ┃         │         ┃
    #              j = 1 ─▶ ┗━━━━━━━━ v ━━━━━━━━┻━━━━━━━━ v ━━━━━━━━┛
	#                       ▲         ▲         ▲         ▲
    #                   i = 1         1         2         2
    # 𝑖 = mod1(i - 1, Nx) = 2                   1

	# Circular-shift i index for tx data
	tx[i, j, k] = tx_data[mod1(i - 1, Nx), j, 𝑘]

	# Shift j index for ty data
	ty[i, j + 1, k] = ty_data[i, j, 𝑘]
end

"""
Shift MOM transports from east/north face convention to Oceananigans west/south
face convention and flip vertical indexing.
"""
function Cgrid_transport_from_MOM_output(grid, tx_data, ty_data)
	north = FPivotZipperBoundaryCondition(-1)

	tx_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north)
	ty_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north)

	tx = XFaceField(grid; boundary_conditions = tx_bcs)
	ty = YFaceField(grid; boundary_conditions = ty_bcs)

	Nx, Ny, Nz = size(grid)
	kp = KernelParameters(1:Nx, 1:Ny - 1, 1:Nz) # size of FPivot grid is +1 in y

	launch!(architecture(grid), grid, kp, shift_transport_to_oceananigans_convention!,
			tx, ty, Nx, Ny, Nz,
			on_architecture(architecture(grid), tx_data),
			on_architecture(architecture(grid), ty_data))

	fill_halo_regions_checked!(tx, "tx_after_shift")
	fill_halo_regions_checked!(ty, "ty_after_shift")

	return tx, ty
end

@kernel function compute_tz_from_continuity!(tz, grid, tx, ty, Nz)
	i, j = @index(Global, NTuple)

	@inbounds tz[i, j, 1] = 0.0

	@inbounds for k in 1:Nz
		horizontal_div = δxᶜᵃᵃ(i, j, k, grid, tx) + δyᵃᶜᵃ(i, j, k, grid, ty)
		tz[i, j, k + 1] = tz[i, j, k] - horizontal_div
	end
end

function continuity_tz_from_tx_ty(grid, tx, ty)
	fill_halo_regions_checked!(tx, "tx_before_continuity")
	fill_halo_regions_checked!(ty, "ty_before_continuity")

	Nx, Ny, Nz = size(grid)
	tz = Field{Center, Center, Face}(grid)
	kp = KernelParameters(1:Nx, 1:Ny)

	launch!(architecture(grid), grid, kp, compute_tz_from_continuity!, tz, grid, tx, ty, Nz)
	fill_halo_regions_checked!(tz, "tz_after_continuity")

	return tz
end

function max_abs_diff_nonzero_entries(before, after)
	nonzero_mask = before .!= 0
	return any(nonzero_mask) ? maximum(abs.(after[nonzero_mask] .- before[nonzero_mask])) : 0.0
end

function print_array_eltype(name, arr)
	T = eltype(arr)
	println("eltype($name) = $T (Float64=$(T == Float64))")
	return nothing
end

function print_field_eltype(name, field)
	A = adapt(Array, interior(field))
	T = eltype(A)
	println("eltype($name interior) = $T (Float64=$(T == Float64))")
	return nothing
end

function print_max_abs_diff(label, lhs, rhs)
	diff = abs.(lhs .- rhs)
	println("$label max abs diff = $(maximum(diff))")
	return nothing
end

function mask_immersed_field_checked!(field, value, field_name)
	if CHECK_FIELD_INTEGRITY
		before = adapt(Array, interior(field))
		mask_immersed_field!(field, value)
		after = adapt(Array, interior(field))
		maxdiff = max_abs_diff_nonzero_entries(before, after)
		if !iszero(maxdiff)
			println("integrity [$field_name] after mask_immersed_field!: nonzero unchanged=$(iszero(maxdiff)), max abs diff=$maxdiff")
		end
	else
		mask_immersed_field!(field, value)
	end
	return field
end

function fill_halo_regions_checked!(field, field_name)
	if CHECK_FIELD_INTEGRITY
		before = adapt(Array, interior(field))
		fill_halo_regions!(field)
		after = adapt(Array, interior(field))
		maxdiff = max_abs_diff_nonzero_entries(before, after)
		if !iszero(maxdiff)
			println("integrity [$field_name] after fill_halo_regions!: nonzero unchanged=$(iszero(maxdiff)), max abs diff=$maxdiff")
		end
	else
		fill_halo_regions!(field)
	end
	return field
end

function plot_velocity_slice(field, field_name, month_idx, k_level, plot_dir, parentmodel)
	plottable = view(make_plottable_array(field), :, :, k_level)
	valid_values = plottable[.!isnan.(plottable)]
	max_velocity = isempty(valid_values) ? 1.0 : quantile(abs.(valid_values), 0.99)
	max_velocity = max(max_velocity, eps(Float64))

	fig = Figure(size = (1200, 600))
	ax = Axis(fig[1, 1], title = "$(field_name) [k=$(k_level), month=$(month_idx)]")
	hm = heatmap!(ax, plottable; colormap = :balance, colorrange = (-max_velocity, max_velocity), nan_color = :black)
	Colorbar(fig[1, 2], hm)

	plot_file = joinpath(plot_dir, "$(parentmodel)_$(field_name)_from_mass_transports_month_$(month_idx).png")
	save(plot_file, fig)

	return nothing
end

################################################################################
# Create transports/velocities
################################################################################

@info "Compute velocities from mass transport files"

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

tx_ds = open_dataset(joinpath(inputdir, "tx_trans_periodic.nc"))
ty_ds = open_dataset(joinpath(inputdir, "ty_trans_periodic.nc"))

prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

u_file = joinpath(outputdir, "$(parentmodel)_u_ts_from_mass_transports.jld2")
v_file = joinpath(outputdir, "$(parentmodel)_v_ts_from_mass_transports.jld2")
w_file = joinpath(outputdir, "$(parentmodel)_w_ts_from_mass_transports.jld2")

rm(u_file; force = true)
rm(v_file; force = true)
rm(w_file; force = true)

u_ts = FieldTimeSeries{Face, Center, Center}(grid, fts_times; backend = OnDisk(), path = u_file, name = "u", time_indexing = Cyclical(stop_time))
v_ts = FieldTimeSeries{Center, Face, Center}(grid, fts_times; backend = OnDisk(), path = v_file, name = "v", time_indexing = Cyclical(stop_time))
w_ts = FieldTimeSeries{Center, Center, Face}(grid, fts_times; backend = OnDisk(), path = w_file, name = "w", time_indexing = Cyclical(stop_time))

Axᶠᶜᶜ = Field(grid_metric_operation((Face, Center, Center), Ax, grid))
Ayᶜᶠᶜ = Field(grid_metric_operation((Center, Face, Center), Ay, grid))
Azᶜᶜᶠ = Field(grid_metric_operation((Center, Center, Face), Az, grid))
compute!(Axᶠᶜᶜ)
compute!(Ayᶜᶠᶜ)
compute!(Azᶜᶜᶠ)

Axᶠᶜᶜ_underlying = Field(grid_metric_operation((Face, Center, Center), Ax, grid.underlying_grid))
Ayᶜᶠᶜ_underlying = Field(grid_metric_operation((Center, Face, Center), Ay, grid.underlying_grid))
Azᶜᶜᶠ_underlying = Field(grid_metric_operation((Center, Center, Face), Az, grid.underlying_grid))
compute!(Axᶠᶜᶜ_underlying)
compute!(Ayᶜᶠᶜ_underlying)
compute!(Azᶜᶜᶠ_underlying)

Axᶠᶜᶜ_diff = abs.(adapt(Array, interior(Axᶠᶜᶜ)) .- adapt(Array, interior(Axᶠᶜᶜ_underlying)))
Ayᶜᶠᶜ_diff = abs.(adapt(Array, interior(Ayᶜᶠᶜ)) .- adapt(Array, interior(Ayᶜᶠᶜ_underlying)))
Azᶜᶜᶠ_diff = abs.(adapt(Array, interior(Azᶜᶜᶠ)) .- adapt(Array, interior(Azᶜᶜᶠ_underlying)))

println("Immersed area-metric check (max abs difference from underlying grid):" *
	" Axᶠᶜᶜ=$(maximum(Axᶠᶜᶜ_diff))," *
	" Ayᶜᶠᶜ=$(maximum(Ayᶜᶠᶜ_diff))," *
	" Azᶜᶜᶠ=$(maximum(Azᶜᶜᶠ_diff))")

Δxᶜᶜᶜ = Field(xspacings(grid, Center(), Center(), Center()))
Δyᶜᶜᶜ = Field(yspacings(grid, Center(), Center(), Center()))
Δzᶜᶜᶜ = Field(zspacings(grid, Center(), Center(), Center()))
Δxᶜᶜᶜ_underlying = Field(xspacings(grid.underlying_grid, Center(), Center(), Center()))
Δyᶜᶜᶜ_underlying = Field(yspacings(grid.underlying_grid, Center(), Center(), Center()))
Δzᶜᶜᶜ_underlying = Field(zspacings(grid.underlying_grid, Center(), Center(), Center()))
compute!(Δxᶜᶜᶜ)
compute!(Δyᶜᶜᶜ)
compute!(Δzᶜᶜᶜ)
compute!(Δxᶜᶜᶜ_underlying)
compute!(Δyᶜᶜᶜ_underlying)
compute!(Δzᶜᶜᶜ_underlying)

Δxᶜᶜᶜ_diff = abs.(adapt(Array, interior(Δxᶜᶜᶜ)) .- adapt(Array, interior(Δxᶜᶜᶜ_underlying)))
Δyᶜᶜᶜ_diff = abs.(adapt(Array, interior(Δyᶜᶜᶜ)) .- adapt(Array, interior(Δyᶜᶜᶜ_underlying)))
Δzᶜᶜᶜ_diff = abs.(adapt(Array, interior(Δzᶜᶜᶜ)) .- adapt(Array, interior(Δzᶜᶜᶜ_underlying)))

println("Immersed spacing check (max abs difference from underlying grid):" *
	" Δxᶜᶜᶜ=$(maximum(Δxᶜᶜᶜ_diff))," *
	" Δyᶜᶜᶜ=$(maximum(Δyᶜᶜᶜ_diff))," *
	" Δzᶜᶜᶜ=$(maximum(Δzᶜᶜᶜ_diff))")

Δyᶠᶜᶜ = Field(yspacings(grid, Face(), Center(), Center()))
Δzᶠᶜᶜ = Field(zspacings(grid, Face(), Center(), Center()))
Δyᶠᶜᶜ_underlying = Field(yspacings(grid.underlying_grid, Face(), Center(), Center()))
Δzᶠᶜᶜ_underlying = Field(zspacings(grid.underlying_grid, Face(), Center(), Center()))
compute!(Δyᶠᶜᶜ)
compute!(Δzᶠᶜᶜ)
compute!(Δyᶠᶜᶜ_underlying)
compute!(Δzᶠᶜᶜ_underlying)

Δxᶜᶠᶜ = Field(xspacings(grid, Center(), Face(), Center()))
Δzᶜᶠᶜ = Field(zspacings(grid, Center(), Face(), Center()))
Δxᶜᶠᶜ_underlying = Field(xspacings(grid.underlying_grid, Center(), Face(), Center()))
Δzᶜᶠᶜ_underlying = Field(zspacings(grid.underlying_grid, Center(), Face(), Center()))
compute!(Δxᶜᶠᶜ)
compute!(Δzᶜᶠᶜ)
compute!(Δxᶜᶠᶜ_underlying)
compute!(Δzᶜᶠᶜ_underlying)

Axᶠᶜᶜ_arr = adapt(Array, interior(Axᶠᶜᶜ))
Ayᶜᶠᶜ_arr = adapt(Array, interior(Ayᶜᶠᶜ))
Axᶠᶜᶜ_underlying_arr = adapt(Array, interior(Axᶠᶜᶜ_underlying))
Ayᶜᶠᶜ_underlying_arr = adapt(Array, interior(Ayᶜᶠᶜ_underlying))

Axᶠᶜᶜ_from_spacings = adapt(Array, interior(Δyᶠᶜᶜ)) .* adapt(Array, interior(Δzᶠᶜᶜ))
Ayᶜᶠᶜ_from_spacings = adapt(Array, interior(Δxᶜᶠᶜ)) .* adapt(Array, interior(Δzᶜᶠᶜ))
Axᶠᶜᶜ_underlying_from_spacings = adapt(Array, interior(Δyᶠᶜᶜ_underlying)) .* adapt(Array, interior(Δzᶠᶜᶜ_underlying))
Ayᶜᶠᶜ_underlying_from_spacings = adapt(Array, interior(Δxᶜᶠᶜ_underlying)) .* adapt(Array, interior(Δzᶜᶠᶜ_underlying))

print_max_abs_diff("Axᶠᶜᶜ (immersed) vs Δyᶠᶜᶜ⋅Δzᶠᶜᶜ", Axᶠᶜᶜ_arr, Axᶠᶜᶜ_from_spacings)
print_max_abs_diff("Axᶠᶜᶜ (underlying) vs Δyᶠᶜᶜ⋅Δzᶠᶜᶜ", Axᶠᶜᶜ_underlying_arr, Axᶠᶜᶜ_underlying_from_spacings)
print_max_abs_diff("Ayᶜᶠᶜ (immersed) vs Δxᶜᶠᶜ⋅Δzᶜᶠᶜ", Ayᶜᶠᶜ_arr, Ayᶜᶠᶜ_from_spacings)
print_max_abs_diff("Ayᶜᶠᶜ (underlying) vs Δxᶜᶠᶜ⋅Δzᶜᶠᶜ", Ayᶜᶠᶜ_underlying_arr, Ayᶜᶠᶜ_underlying_from_spacings)

Δzᶠᶜᶜ_arr = adapt(Array, interior(Δzᶠᶜᶜ))
Δzᶜᶜᶜ_arr = adapt(Array, interior(Δzᶜᶜᶜ))

nxf, nyf, nzf = size(Δzᶠᶜᶜ_arr)
nxc, nyc, nzc = size(Δzᶜᶜᶜ_arr)
imax = min(nxf - 1, nxc)
jmax = min(nyf, nyc)
kmax = min(nzf, nzc)

if imax >= 2 && jmax >= 1 && kmax >= 1
	Δz_face = @view Δzᶠᶜᶜ_arr[2:imax, 1:jmax, 1:kmax]
	Δz_center_min = min.(@view(Δzᶜᶜᶜ_arr[1:imax-1, 1:jmax, 1:kmax]),
						@view(Δzᶜᶜᶜ_arr[2:imax, 1:jmax, 1:kmax]))
	Δz_face_min_diff = abs.(Δz_face .- Δz_center_min)
	println("Immersed Δz face-min check max abs diff = $(maximum(Δz_face_min_diff))")
else
	println("Immersed Δz face-min check skipped due to insufficient overlap dimensions")
end

print_field_eltype("Axᶠᶜᶜ", Axᶠᶜᶜ)
print_field_eltype("Ayᶜᶠᶜ", Ayᶜᶠᶜ)
print_field_eltype("Azᶜᶜᶠ", Azᶜᶜᶠ)

for month_idx in 1:12
	@info "Processing month $month_idx / 12"

	tx_data = replace(readcubedata(tx_ds.tx_trans[month = At(month_idx)]).data, NaN => 0.0)
	ty_data = replace(readcubedata(ty_ds.ty_trans[month = At(month_idx)]).data, NaN => 0.0)
	print_array_eltype("tx_data", tx_data)
	print_array_eltype("ty_data", ty_data)

	tx, ty = Cgrid_transport_from_MOM_output(grid, tx_data, ty_data)
	mask_immersed_field_checked!(tx, 0.0, "tx")
	mask_immersed_field_checked!(ty, 0.0, "ty")
	fill_halo_regions_checked!(tx, "tx")
	fill_halo_regions_checked!(ty, "ty")
	print_field_eltype("tx", tx)
	print_field_eltype("ty", ty)

	u = XFaceField(grid)
	v = YFaceField(grid)
	u .= tx / (ρ₀ * Axᶠᶜᶜ)
	v .= ty / (ρ₀ * Ayᶜᶠᶜ)
	mask_immersed_field_checked!(u, 0.0, "u")
	mask_immersed_field_checked!(v, 0.0, "v")
	fill_halo_regions_checked!(u, "u")
	fill_halo_regions_checked!(v, "v")
	print_field_eltype("u", u)
	print_field_eltype("v", v)

	tz = continuity_tz_from_tx_ty(grid, tx, ty)
	mask_immersed_field_checked!(tz, 0.0, "tz")
	fill_halo_regions_checked!(tz, "tz")
	print_field_eltype("tz", tz)

	w = Field{Center, Center, Face}(grid)
	w .= tz / (ρ₀ * Azᶜᶜᶠ)
	mask_immersed_field_checked!(w, 0.0, "w")
	fill_halo_regions_checked!(w, "w")
	print_field_eltype("w", w)

	if ENABLE_VELOCITY_PLOTTING
		plot_velocity_slice(u, "u", month_idx, 25, u_plot_dir, parentmodel)
		plot_velocity_slice(v, "v", month_idx, 25, v_plot_dir, parentmodel)
		plot_velocity_slice(w, "w", month_idx, 25, w_plot_dir, parentmodel)
		plot_velocity_slice(tz, "tz", month_idx, 25, tz_plot_dir, parentmodel)
	end

	set!(u_ts, u, month_idx)
	set!(v_ts, v, month_idx)
	set!(w_ts, w, month_idx)
end

@info("""
Transport-derived FieldTimeSeries outputs saved to
- $(u_file)
- $(v_file)
- $(w_file)
""")