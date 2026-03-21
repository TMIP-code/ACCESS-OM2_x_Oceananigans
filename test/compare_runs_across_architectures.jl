"""
Compare age output between serial (1 GPU) and distributed (multi-GPU) runs.

Loads each snapshot (part file) one at a time, computes volume-weighted
RMS norm of the difference, then plots diagnostics of the final snapshot.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
GPU_TAG=2x2 DURATION_TAG=1year julia --project test/compare_runs_across_architectures.jl
```

Environment variables:
  GPU_TAG          – partition tag to compare against serial (e.g., "2x2")
  DURATION_TAG     – output tag (default: "1year", can be "diag" for 10-step diagnostic)
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

@info "Loading packages for architecture comparison"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days

using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using Statistics
using TOML
using JLD2
using OffsetArrays
using Printf

include("../src/shared_functions.jl")

################################################################################
# Configuration
################################################################################

GPU_TAG = get(ENV, "GPU_TAG", "")
isempty(GPU_TAG) && error("GPU_TAG must be set (e.g., GPU_TAG=2x2)")
DURATION_TAG = get(ENV, "DURATION_TAG", "1year")

# Parse partition from GPU_TAG (e.g., "2x2" → px=2, py=2)
gpu_tag_parts = split(GPU_TAG, "x")
length(gpu_tag_parts) == 2 || error("GPU_TAG must be in format PxQ (e.g., 2x2), got: $GPU_TAG")
px = parse(Int, gpu_tag_parts[1])
py = parse(Int, gpu_tag_parts[2])

(; parentmodel, experiment_dir, outputdir) = load_project_config(; parentmodel_arg_index = 2)

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

serial_dir = joinpath(outputdir, "standardrun", model_config)
distributed_dir = joinpath(outputdir, "standardrun", model_config, GPU_TAG)

@info "Architecture comparison configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- GPU_TAG          = $GPU_TAG ($(px)x$(py) partition)"
@info "- DURATION_TAG     = $DURATION_TAG"
@info "- model_config     = $model_config"
@info "- Serial dir       = $serial_dir"
@info "- Distributed dir  = $distributed_dir"
flush(stdout); flush(stderr)

# Verify part files exist
serial_part1 = joinpath(serial_dir, "age_$(DURATION_TAG)_part1.jld2")
dist_part1 = joinpath(distributed_dir, "age_$(DURATION_TAG)_rank0_part1.jld2")
isfile(serial_part1) || error("Serial part file not found: $serial_part1")
isfile(dist_part1) || error("Distributed part file not found: $dist_part1")

# Auto-detect number of parts from serial directory
NPARTS = length(filter(f -> startswith(f, "age_$(DURATION_TAG)_part") && endswith(f, ".jld2"), readdir(serial_dir)))
@info "Detected $NPARTS part files for DURATION_TAG=$DURATION_TAG"

################################################################################
# Load grid, masks, volumes
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

@info "Computing wet mask and cell volumes"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx, Ny, Nz = size(wet3D)

vol = compute_volume(grid)
vol_3D = Array(interior(vol))

# Volume-weighted RMS norm (in years)
v1D = vol_3D[idx]
vol_norm = make_vol_norm(v1D, year)

################################################################################
# Per-snapshot comparison (load one part at a time)
################################################################################

@info "Comparing $NPARTS snapshots"
@info @sprintf("  %5s  %10s  %14s  %14s", "part", "time(yr)", "vol_norm(yr)", "max|diff|(yr)")
flush(stdout); flush(stderr)

# Store snapshots for plotting: parts 1, 2, and NPARTS
plot_parts = sort(unique([1, 2, NPARTS]))
snapshots = Dict{Int, @NamedTuple{age_serial_yr::Array{Float64, 3}, age_dist_yr::Array{Float64, 3}, age_diff_yr::Array{Float64, 3}, t_yr::Float64}}()

for part in 1:NPARTS
    age_serial_full, t_serial = load_serial_part(serial_dir, "age", DURATION_TAG, part)
    age_dist_raw, t_dist = load_distributed_part(distributed_dir, "age", DURATION_TAG, part, px, py, Nx, Ny, Nz)

    # Trim serial data to interior size (fold row may be included in serial output)
    age_serial_raw = @view age_serial_full[1:Nx, 1:Ny, 1:Nz]

    diff_raw = age_dist_raw .- age_serial_raw
    diff_1D = diff_raw[idx]

    vn = vol_norm(diff_1D)
    maxdiff = maximum(abs, diff_1D) / year
    t_yr = t_serial / year

    @info @sprintf("  %5d  %10.5f  %14.2e  %14.2e", part, t_yr, vn, maxdiff)

    if part ∈ plot_parts
        snapshots[part] = (
            age_serial_yr = Array(age_serial_raw) ./ year,
            age_dist_yr = age_dist_raw ./ year,
            age_diff_yr = (age_dist_raw .- Array(age_serial_raw)) ./ year,
            t_yr = t_yr,
        )
    end
end
flush(stdout); flush(stderr)

# Use final snapshot for summary statistics
age_serial_yr = snapshots[NPARTS].age_serial_yr
age_dist_yr = snapshots[NPARTS].age_dist_yr
age_diff_yr = snapshots[NPARTS].age_diff_yr

################################################################################
# Summary statistics for final snapshot (wet cells only)
################################################################################

wet_diff = age_diff_yr[wet3D]

# Relative difference (avoid division by zero)
age_reldiff = similar(age_diff_yr)
for i in eachindex(age_diff_yr)
    if wet3D[i] && abs(age_serial_yr[i]) > 1.0e-10
        age_reldiff[i] = age_diff_yr[i] / age_serial_yr[i]
    else
        age_reldiff[i] = NaN
    end
end
wet_reldiff = filter(!isnan, age_reldiff[wet3D])

@info "Final snapshot statistics (wet cells only):"
@info @sprintf("  max|diff|    = %.2e years", maximum(abs, wet_diff))
@info @sprintf("  mean|diff|   = %.2e years", mean(abs, wet_diff))
@info @sprintf("  RMS diff     = %.2e years", sqrt(mean(wet_diff .^ 2)))
if !isempty(wet_reldiff)
    @info @sprintf("  max|reldiff| = %.2e", maximum(abs, wet_reldiff))
    @info @sprintf("  mean|reldiff|= %.2e", mean(abs, wet_reldiff))
else
    @info "  reldiff: not computed (all values near zero)"
end
flush(stdout); flush(stderr)

################################################################################
# Generate diagnostic plots for stored snapshots
################################################################################

plot_output_dir = joinpath(serial_dir, "plots", "compare_$(GPU_TAG)_$(DURATION_TAG)")
mkpath(plot_output_dir)
@info "Saving comparison plots to $plot_output_dir"
flush(stdout); flush(stderr)

n_levels = 11

for part in plot_parts
    snap = snapshots[part]
    part_label = "part$(part)"
    @info "Plotting $part_label (t = $(@sprintf("%.5f", snap.t_yr)) yr)"

    compare_k_indices = 30:50

    # Serial age (reference)
    plot_age_diagnostics(
        snap.age_serial_yr, grid, wet3D, vol_3D, plot_output_dir,
        "serial_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(part_label)";
        colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
        target_k_indices = compare_k_indices,
    )

    # Distributed age
    plot_age_diagnostics(
        snap.age_dist_yr, grid, wet3D, vol_3D, plot_output_dir,
        "distributed_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(part_label)";
        colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
        target_k_indices = compare_k_indices,
    )

    # Absolute difference — colorscale based on mean|diff|
    wet_diff_part = snap.age_diff_yr[wet3D]
    mean_abs_diff = mean(abs, wet_diff_part)
    diff_scale = mean_abs_diff > 0 ? 3 * mean_abs_diff : 1.0e-10
    diff_range = (-diff_scale, diff_scale)
    diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
    plot_age_diagnostics(
        snap.age_diff_yr, grid, wet3D, vol_3D, plot_output_dir,
        "diff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(part_label)";
        colorrange = diff_range, levels = diff_levels,
        colormap = cgrad(:balance, n_levels - 1, categorical = true),
        lowclip = :blue, highclip = :red,
        target_k_indices = compare_k_indices,
    )

    # Relative difference (skip if age too small)
    age_reldiff_part = similar(snap.age_diff_yr)
    for i in eachindex(snap.age_diff_yr)
        if wet3D[i] && abs(snap.age_serial_yr[i]) > 1.0e-10
            age_reldiff_part[i] = snap.age_diff_yr[i] / snap.age_serial_yr[i]
        else
            age_reldiff_part[i] = NaN
        end
    end
    wet_reldiff_part = filter(!isnan, age_reldiff_part[wet3D])

    if !isempty(wet_reldiff_part)
        mean_abs_reldiff = mean(abs, wet_reldiff_part)
        reldiff_scale = mean_abs_reldiff > 0 ? 3 * mean_abs_reldiff : 1.0e-10
        reldiff_range = (-reldiff_scale, reldiff_scale)
        reldiff_levels = range(reldiff_range[1], reldiff_range[2]; length = n_levels)
        plot_age_diagnostics(
            age_reldiff_part, grid, wet3D, vol_3D, plot_output_dir,
            "reldiff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(part_label)";
            colorrange = reldiff_range, levels = reldiff_levels,
            colormap = cgrad(:balance, n_levels - 1, categorical = true),
            lowclip = :blue, highclip = :red,
            target_k_indices = compare_k_indices,
        )
    end

    flush(stdout); flush(stderr)
end

################################################################################
# Compare velocity and eta fields (part 2 = first timestep after t=0)
################################################################################

# Extract grid coordinates for heatmap plotting
ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
lon = Array(ug.λᶜᶜᵃ[1:Nx, 1:Ny])
lat = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])

velocity_fields = ["u", "v", "w", "eta"]
velocity_part = 2  # first timestep (part 1 = t=0, part 2 = t=Δt)

# Check if velocity files exist (they won't if distributed was GPU-only)
serial_vel_exists = isfile(joinpath(serial_dir, "u_$(DURATION_TAG)_part$(velocity_part).jld2"))
dist_vel_exists = isfile(joinpath(distributed_dir, "u_$(DURATION_TAG)_rank0_part$(velocity_part).jld2"))

if serial_vel_exists && dist_vel_exists
    @info "Comparing velocity/eta fields at part $velocity_part"
    flush(stdout); flush(stderr)

    for field_name in velocity_fields
        serial_file = joinpath(serial_dir, "$(field_name)_$(DURATION_TAG)_part$(velocity_part).jld2")
        dist_file = joinpath(distributed_dir, "$(field_name)_$(DURATION_TAG)_rank0_part$(velocity_part).jld2")
        if !isfile(serial_file) || !isfile(dist_file)
            @warn "Skipping $field_name: missing serial or distributed file"
            continue
        end

        field_serial, t_s = load_serial_part(serial_dir, field_name, DURATION_TAG, velocity_part)
        field_dist, t_d = load_distributed_part(distributed_dir, field_name, DURATION_TAG, velocity_part, px, py, Nx, Ny)
        t_yr = t_s / year

        # Trim serial to match distributed interior size
        nz_field = size(field_dist, 3)
        field_serial_trimmed = @view field_serial[1:Nx, 1:Ny, 1:nz_field]
        field_diff = field_dist .- Array(field_serial_trimmed)

        # Determine which k-slices to plot
        # For eta (2D, stored as nz=1), use k=1 for data but surface mask (k=Nz)
        if field_name == "eta"
            k_slices = [1]
            k_labels = ["2D"]
            mask_k_override = Nz  # surface mask for 2D field
        elseif field_name == "w"
            # w is at Face z-locations: Nz+1 levels. Plot top (k=nz_field) and near-surface (k=nz_field-1)
            k_slices = [nz_field, nz_field - 1]
            k_labels = ["k$(nz_field)_top", "k$(nz_field - 1)"]
            mask_k_override = nothing
        else
            # u, v: plot surface k=Nz
            k_slices = [Nz]
            k_labels = ["k$(Nz)_surface"]
            mask_k_override = nothing
        end

        for (ki, (k, klabel)) in enumerate(zip(k_slices, k_labels))
            mk = isnothing(mask_k_override) ? min(k, Nz) : mask_k_override
            # --- Serial field ---
            serial_slice = Array(field_serial_trimmed[:, :, k])
            serial_slice[.!wet3D[:, :, mk]] .= NaN

            fig = Figure(; size = (900, 500))
            ax = Axis(
                fig[1, 1];
                title = "$field_name serial (t=$(@sprintf("%.5f", t_yr)) yr, $klabel)",
                xlabel = "Longitude", ylabel = "Latitude",
                backgroundcolor = :lightgray,
            )
            hm = heatmap!(ax, 1:Nx, 1:Ny, serial_slice; nan_color = :lightgray)
            Colorbar(fig[1, 2], hm; label = field_name)
            save(joinpath(plot_output_dir, "$(field_name)_serial_$(DURATION_TAG)_$(klabel).png"), fig)

            # --- Distributed field ---
            dist_slice = field_dist[:, :, k]
            dist_slice[.!wet3D[:, :, mk]] .= NaN

            fig = Figure(; size = (900, 500))
            ax = Axis(
                fig[1, 1];
                title = "$field_name distributed $(GPU_TAG) (t=$(@sprintf("%.5f", t_yr)) yr, $klabel)",
                xlabel = "Longitude", ylabel = "Latitude",
                backgroundcolor = :lightgray,
            )
            hm = heatmap!(ax, 1:Nx, 1:Ny, dist_slice; nan_color = :lightgray)
            Colorbar(fig[1, 2], hm; label = field_name)
            save(joinpath(plot_output_dir, "$(field_name)_dist_$(GPU_TAG)_$(DURATION_TAG)_$(klabel).png"), fig)

            # --- Difference ---
            diff_slice = copy(field_diff[:, :, k])
            wet_slice = wet3D[:, :, mk]
            diff_slice[.!wet_slice] .= NaN
            wet_vals = filter(!isnan, diff_slice[wet_slice])

            if !isempty(wet_vals)
                mean_abs = mean(abs, wet_vals)
                max_abs = maximum(abs, wet_vals)
                rms = sqrt(mean(wet_vals .^ 2))
                @info @sprintf(
                    "  %4s %s: max|diff|=%.2e  mean|diff|=%.2e  RMS=%.2e",
                    field_name, klabel, max_abs, mean_abs, rms
                )

                diff_scale = mean_abs > 0 ? 3 * mean_abs : 1.0e-10
                fig = Figure(; size = (900, 500))
                ax = Axis(
                    fig[1, 1];
                    title = "$field_name diff (dist-serial) (t=$(@sprintf("%.5f", t_yr)) yr, $klabel)",
                    xlabel = "Longitude", ylabel = "Latitude",
                    backgroundcolor = :lightgray,
                )
                hm = heatmap!(
                    ax, 1:Nx, 1:Ny, diff_slice;
                    colorrange = (-diff_scale, diff_scale),
                    colormap = :balance, nan_color = :lightgray,
                    lowclip = :blue, highclip = :red,
                )
                Colorbar(fig[1, 2], hm; label = "$field_name diff")
                save(joinpath(plot_output_dir, "$(field_name)_diff_$(GPU_TAG)_$(DURATION_TAG)_$(klabel).png"), fig)
            end
        end

        flush(stdout); flush(stderr)
    end
else
    @info "Velocity/eta files not found — skipping velocity comparison"
    if !serial_vel_exists
        @info "  Missing serial: $(joinpath(serial_dir, "u_$(DURATION_TAG)_part$(velocity_part).jld2"))"
    end
    if !dist_vel_exists
        @info "  Missing distributed: $(joinpath(distributed_dir, "u_$(DURATION_TAG)_rank0_part$(velocity_part).jld2"))"
    end
end
flush(stdout); flush(stderr)

################################################################################
# Halo-inclusive plots (fold diagnostic)
################################################################################

# JLD2Writer with `with_halos=true` saves data as OffsetArrays that include halos.
# Load the raw data and convert OffsetArray → plain Array via `parent()` to plot
# the full domain including halo regions.

halo_fields = ["u", "v", "w", "age"]
halo_part = 2  # first timestep

@info "Plotting halo-inclusive fields (fold diagnostic)"
flush(stdout); flush(stderr)

"""Load raw data from JLD2Writer file, preserving halos. Returns (parent_array, time)."""
function load_with_halos(filepath, field_name)
    return jldopen(filepath, "r") do f
        iters = keys(f["timeseries/$(field_name)"])
        iter = first(filter(k -> k != "serialized", iters))
        raw = f["timeseries/$(field_name)/$iter"]
        t = f["timeseries/t/$iter"]
        # OffsetArray → plain Array (parent strips the offset indices)
        data = raw isa OffsetArrays.OffsetArray ? parent(raw) : raw
        return data, t
    end
end

for field_name in halo_fields
    serial_file = joinpath(serial_dir, "$(field_name)_$(DURATION_TAG)_part$(halo_part).jld2")
    if !isfile(serial_file)
        @warn "Skipping $field_name halo plot: no serial file"
        continue
    end

    serial_halo, t_s = load_with_halos(serial_file, field_name)
    t_yr = t_s / year

    # Determine k-slice for plotting (in parent array coordinates, halos included)
    nz_halo = size(serial_halo, 3)
    if field_name == "eta"
        k_halo = nz_halo ÷ 2 + 1  # middle of the z-with-halos dimension for 2D field
    elseif field_name == "w"
        k_halo = nz_halo - 7  # near-surface face (avoid top halo)
    else
        k_halo = nz_halo - 7  # near-surface center (avoid top halo)
    end

    # Plot serial with halos
    nx_h, ny_h = size(serial_halo, 1), size(serial_halo, 2)
    @info "  $field_name serial: parent size=$(size(serial_halo)), plotting k=$k_halo"
    fig = Figure(; size = (1000, 600))
    ax = Axis(
        fig[1, 1];
        title = "$field_name serial WITH HALOS (t=$(@sprintf("%.5f", t_yr)) yr, k=$k_halo, size=$(size(serial_halo)))",
        xlabel = "i (with halos)", ylabel = "j (with halos)",
        backgroundcolor = :lightgray,
    )
    hm = heatmap!(ax, 1:nx_h, 1:ny_h, serial_halo[:, :, k_halo]; nan_color = :lightgray)
    Colorbar(fig[1, 2], hm; label = field_name)
    save(joinpath(plot_output_dir, "$(field_name)_serial_halos_$(DURATION_TAG)_k$(k_halo).png"), fig)

    # Plot each distributed rank with halos
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = ry * px + rx
        dist_file = joinpath(distributed_dir, "$(field_name)_$(DURATION_TAG)_rank$(r)_part$(halo_part).jld2")
        if !isfile(dist_file)
            @warn "Missing distributed file for rank $r: $dist_file"
            continue
        end
        rank_halo, _ = load_with_halos(dist_file, field_name)
        nxr, nyr = size(rank_halo, 1), size(rank_halo, 2)
        k_r = min(k_halo, size(rank_halo, 3))
        @info "  $field_name rank $r: parent size=$(size(rank_halo)), plotting k=$k_r"
        fig = Figure(; size = (1000, 600))
        ax = Axis(
            fig[1, 1];
            title = "$field_name rank $r ($(rx),$(ry)) WITH HALOS (k=$k_r, size=$(size(rank_halo)))",
            xlabel = "i (with halos)", ylabel = "j (with halos)",
            backgroundcolor = :lightgray,
        )
        hm = heatmap!(ax, 1:nxr, 1:nyr, rank_halo[:, :, k_r]; nan_color = :lightgray)
        Colorbar(fig[1, 2], hm; label = field_name)
        save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_halos_$(DURATION_TAG)_k$(k_r).png"), fig)
    end

    @info "  $field_name: halo plots saved (serial + $(px * py) ranks)"
    flush(stdout); flush(stderr)
end

@info "compare_runs_across_architectures.jl complete (GPU_TAG=$GPU_TAG, DURATION_TAG=$DURATION_TAG)"
flush(stdout); flush(stderr)
