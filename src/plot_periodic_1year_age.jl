"""
Plot age diagnostic figures from the 1-year periodic simulation output.

Generates:
  - Periodicity check (vol-weighted RMS of snapshot 25 − snapshot 1)
  - 10 static PNGs from the time-averaged age (4 zonal averages + 6 depth slices)
  - 10 smooth MP4 animations (4 zonal averages + 6 depth slices, 144 frames @ 24 fps)

This is a standalone CPU script designed to be submitted as a CPU PBS job
after the GPU simulation completes.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_periodic_1year_age.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  LINEAR_SOLVER    – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY   – yes | no  (default: no)
"""

@info "Loading packages for periodic age plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: day, days, second, seconds, Time
year = years = 365.25days

using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using LinearAlgebra: dot, Diagonal
using Statistics
using TOML
using JLD2
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
else
    outputdir = profile["outputdir"]
end

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"
solver_tag = "$(LINEAR_SOLVER)_$(lumpspray_tag)"

periodic_1year_dir = joinpath(outputdir, "periodic", model_config, "1year", solver_tag)
output_filepath = joinpath(periodic_1year_dir, "age_periodic_1year.jld2")

@info "Periodic age plot configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- LINEAR_SOLVER    = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY   = $LUMP_AND_SPRAY (tag: $solver_tag)"
@info "- Output file      = $output_filepath"
flush(stdout); flush(stderr)

isfile(output_filepath) || error("Output file not found: $output_filepath")

################################################################################
# Load grid
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

################################################################################
# Load age FieldTimeSeries
################################################################################

@info "Loading age FieldTimeSeries from $output_filepath"
flush(stdout); flush(stderr)

age_fts = FieldTimeSeries(output_filepath, "age")
n_times = length(age_fts.times)
@info "Found $n_times output timesteps"
flush(stdout); flush(stderr)

################################################################################
# Compute masks, volume, and grid-derived quantities (once)
################################################################################

@info "Computing wet mask, cell volumes, and grid coordinates"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)

vol_3D = Array(interior(compute_volume(grid)))

# Grid coordinates (reused every iteration)
ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
z = znodes(grid, Center(), Center(), Center())
depth_vals = -z
lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

# Basin masks (reused every iteration)
basins = compute_ocean_basin_masks(grid, wet3D)
global_mask = trues(Nx′, Ny′)

basin_configs = [
    ("global", global_mask),
    ("atlantic", basins.ATL),
    ("pacific", basins.PAC),
    ("indian", basins.IND),
]

# Reshape 2D basin masks to 3D once (avoids reshape per zonalaverage call)
basin_masks_3D = [(name, reshape(m, size(m, 1), size(m, 2), 1)) for (name, m) in basin_configs]

# Depth slice indices (reused every iteration)
target_depths = [100, 200, 500, 1000, 2000, 3000]
depth_k_indices = [(d, find_nearest_depth_index(grid, d)) for d in target_depths]

################################################################################
# Preallocate reusable buffers
################################################################################

age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)    # age in years, NaN-masked
xw_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)     # weighted numerator
w_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)      # weighted denominator

# In-place volume-weighted zonal average using preallocated buffers
function zonalaverage!(za, xw, w, x3D, v3D, mask3D)
    @. xw = ifelse(isnan(x3D) | !mask3D, 0.0, x3D * v3D)
    @. w = ifelse(isnan(x3D) | !mask3D, 0.0, v3D)
    @views for j in axes(za, 1), k in axes(za, 2)
        num = 0.0
        den = 0.0
        for i in axes(xw, 1)
            num += xw[i, j, k]
            den += w[i, j, k]
        end
        za[j, k] = den > 0 ? num / den : NaN
    end
    return za
end

za_buf = Array{Float64}(undef, Ny′, Nz′)  # zonal average buffer

################################################################################
# Plot settings
################################################################################

plot_dir = joinpath(periodic_1year_dir, "plots")
mkpath(plot_dir)

colorrange = (0, 1600)
levels = 0:100:1600
colormap = cgrad(:viridis, length(levels) - 1, categorical = true)

@info "Colorrange for all plots: $colorrange years"
flush(stdout); flush(stderr)

################################################################################
# Periodicity check: snapshot 1 ≈ snapshot n_times
################################################################################

@info "Running periodicity check (snapshot 1 vs snapshot $n_times)"
flush(stdout); flush(stderr)

begin
    v1D = vol_3D[idx]
    inv_sumv = 1 / sum(v1D)
    age_first_1D = interior(age_fts[1])[idx]
    age_last_1D = interior(age_fts[n_times])[idx]
    diff_1D = age_last_1D .- age_first_1D
    vol_rms_drift = sqrt(dot(diff_1D, Diagonal(v1D), diff_1D) * inv_sumv) / year
    max_drift = maximum(abs, diff_1D) / year
    mean_drift = mean(abs, diff_1D) / year
    @info "Periodicity check results" vol_rms_drift_years = vol_rms_drift max_drift_years = max_drift mean_drift_years = mean_drift
    flush(stdout); flush(stderr)
end

################################################################################
# Averaged diagnostics (10 static PNGs)
################################################################################

@info "Computing time-averaged age over first $(n_times - 1) snapshots"
flush(stdout); flush(stderr)

n_avg = n_times - 1
fill!(age_buf, 0.0)
for n in 1:n_avg
    age_raw = interior(age_fts[n])
    @. age_buf += ifelse(wet3D, age_raw / year, 0.0)
end
@. age_buf = ifelse(wet3D, age_buf / n_avg, NaN)

label = "age_periodic_mean_$(ADVECTION_SCHEME)"

@info "Generating averaged diagnostic plots"
flush(stdout); flush(stderr)

# ── Zonal averages (figures 1-4) ──────────────────────────────────────

for (basin_name, mask3D) in basin_masks_3D
    zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)

    fig = Figure(; size = (800, 500))
    ax = Axis(
        fig[1, 1];
        title = "$label — $basin_name zonal average",
        xlabel = "Latitude",
        ylabel = "Depth (m)",
        backgroundcolor = :lightgray,
        xgridvisible = false,
        ygridvisible = false,
    )

    cf = contourf!(
        ax, lat_repr, depth_vals, za_buf;
        levels, colormap, nan_color = :lightgray, extendhigh = :auto, extendlow = :auto
    )
    translate!(cf, 0, 0, -100)
    ylims!(ax, maximum(depth_vals), 0)
    Colorbar(fig[1, 2], cf; label = "Age (years)")

    outputfile = joinpath(plot_dir, "$(label)_zonal_avg_$(basin_name).png")
    @info "Saving $outputfile"
    save(outputfile, fig)
end

# ── Horizontal slices (figures 5-10) ──────────────────────────────────

for (depth, k) in depth_k_indices
    actual_depth = round(depth_vals[k]; digits = 1)
    slice = @view age_buf[:, :, k]

    fig = Figure(; size = (1000, 500))
    ax = Axis(
        fig[1, 1];
        title = "$label at $depth m (k=$k, z=$actual_depth m)",
    )

    hm = heatmap!(
        ax, slice; colorrange, colormap, nan_color = :black,
        lowclip = colormap[1], highclip = colormap[end]
    )
    Colorbar(fig[1, 2], hm; label = "Age (years)")

    outputfile = joinpath(plot_dir, "$(label)_slice_$(depth)m.png")
    @info "Saving $outputfile"
    save(outputfile, fig)
end

@info "Averaged diagnostic plots complete (10 PNGs)"
flush(stdout); flush(stderr)

################################################################################
# Smooth animations (10 MP4 files)
################################################################################

@info "Generating smooth animations (144 frames @ 24 fps, 6s per video)"
flush(stdout); flush(stderr)

stop_time = age_fts.times[end]
n_frames = 144
frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

# Preallocate slice buffer (reused across all depth slice animations)
slice_buf = Matrix{Float64}(undef, Nx′, Ny′)

# ── Zonal average animations (4 videos) ──────────────────────────────

for (basin_name, mask3D) in basin_masks_3D
    @info "Animating zonal average — $basin_name"
    flush(stdout); flush(stderr)

    # Compute first frame to initialise observable
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
    za_obs = Observable(copy(za_buf))

    title_obs = Observable(@sprintf("age — %s zonal avg (t = 0.0 months)", basin_name))

    fig = Figure(; size = (800, 500))
    ax = Axis(
        fig[1, 1];
        title = title_obs,
        xlabel = "Latitude",
        ylabel = "Depth (m)",
        backgroundcolor = :lightgray,
        xgridvisible = false,
        ygridvisible = false,
    )

    cf = contourf!(
        ax, lat_repr, depth_vals, za_obs;
        levels, colormap, nan_color = :lightgray, extendhigh = :auto, extendlow = :auto
    )
    translate!(cf, 0, 0, -100)
    ylims!(ax, maximum(depth_vals), 0)
    Colorbar(fig[1, 2], cf; label = "Age (years)")

    filepath = joinpath(plot_dir, "age_periodic_$(ADVECTION_SCHEME)_zonal_avg_$(basin_name).mp4")
    record(fig, filepath, 1:n_frames; framerate = 24) do i
        age_raw = interior(age_fts[Time(frame_times[i])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
        za_obs.val .= za_buf
        notify(za_obs)
        title_obs[] = @sprintf("age — %s zonal avg (t = %.1f months)", basin_name, frame_times[i] / (year / 12))
    end

    @info "Saved $filepath"
end

# ── Depth slice animations (6 videos) ────────────────────────────────

for (depth, k) in depth_k_indices
    @info "Animating depth slice — $(depth) m"
    flush(stdout); flush(stderr)

    actual_depth = round(depth_vals[k]; digits = 1)

    # Compute first frame to initialise observable
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    slice_buf .= @view age_buf[:, :, k]
    slice_obs = Observable(copy(slice_buf))

    title_obs = Observable(@sprintf("age at %d m (k=%d, z=%.1f m, t = 0.0 months)", depth, k, actual_depth))

    fig = Figure(; size = (1000, 500))
    ax = Axis(fig[1, 1]; title = title_obs)

    hm = heatmap!(
        ax, slice_obs; colorrange, colormap, nan_color = :black,
        lowclip = colormap[1], highclip = colormap[end]
    )
    Colorbar(fig[1, 2], hm; label = "Age (years)")

    filepath = joinpath(plot_dir, "age_periodic_$(ADVECTION_SCHEME)_slice_$(depth)m.mp4")
    record(fig, filepath, 1:n_frames; framerate = 24) do i
        age_raw = interior(age_fts[Time(frame_times[i])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        slice_obs.val .= @view(age_buf[:, :, k])
        notify(slice_obs)
        title_obs[] = @sprintf("age at %d m (k=%d, z=%.1f m, t = %.1f months)", depth, k, actual_depth, frame_times[i] / (year / 12))
    end

    @info "Saved $filepath"
end

@info "plot_periodic_1year_age.jl complete — 10 averaged PNGs + 10 animations saved to $plot_dir"
flush(stdout); flush(stderr)
