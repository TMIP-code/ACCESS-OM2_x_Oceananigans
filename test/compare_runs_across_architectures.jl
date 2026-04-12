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
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

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

# Verify JLD2Writer files exist (single file per variable)
serial_age_file = joinpath(serial_dir, "age_$(DURATION_TAG).jld2")
dist_age_file = joinpath(distributed_dir, "age_$(DURATION_TAG)_rank0.jld2")
isfile(serial_age_file) || error("Serial file not found: $serial_age_file")
isfile(dist_age_file) || error("Distributed file not found: $dist_age_file")

# Auto-detect iterations from serial file
iter_keys = list_iterations(serial_dir, "age", DURATION_TAG)
NITERS = length(iter_keys)
@info "Detected $NITERS iterations in $(serial_age_file)"

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
# Per-snapshot comparison
################################################################################

@info "Comparing $NITERS snapshots"
@info @sprintf("  %5s  %10s  %14s  %14s", "iter", "time(yr)", "vol_norm(yr)", "max|diff|(yr)")
flush(stdout); flush(stderr)

# Store snapshots for plotting: first, second, and last
plot_indices = sort(unique([1, 2, NITERS]))
snapshots = Dict{Int, @NamedTuple{age_serial_yr::Array{Float64, 3}, age_dist_yr::Array{Float64, 3}, age_diff_yr::Array{Float64, 3}, t_yr::Float64}}()

# Infer halo size from serial data: parent_size = interior + 2*H
age_sample, _ = load_serial_snapshot(serial_dir, "age", DURATION_TAG, iter_keys[1])
Hx = (size(age_sample, 1) - Nx) ÷ 2
Hy = (size(age_sample, 2) - Ny) ÷ 2
Hz = (size(age_sample, 3) - Nz) ÷ 2
@info "Inferred halo size: Hx=$Hx, Hy=$Hy, Hz=$Hz"

"""Extract interior from halo-inclusive parent data.
For 2D fields (nz ≤ 2Hz), z-halos are not present so use full z range."""
function extract_interior(data, Hx, Hy, Hz, Nx, Ny, Nz)
    nz_raw = size(data, 3)
    z_range = nz_raw > 2Hz ? ((Hz + 1):(Hz + Nz)) : (1:Nz)
    return data[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), z_range]
end

for (idx_i, iter_key) in enumerate(iter_keys)
    age_serial_full, t_serial = load_serial_snapshot(serial_dir, "age", DURATION_TAG, iter_key)
    age_dist_raw, t_dist = load_distributed_snapshot(distributed_dir, "age", DURATION_TAG, iter_key, px, py, Nx, Ny; halo = (Hx, Hy, Hz))

    # Extract interior from halo-inclusive serial data
    age_serial_raw = extract_interior(age_serial_full, Hx, Hy, Hz, Nx, Ny, Nz)

    diff_raw = age_dist_raw .- age_serial_raw
    diff_1D = diff_raw[idx]

    vn = vol_norm(diff_1D)
    maxdiff = maximum(abs, diff_1D) / year
    t_yr = t_serial / year

    @info @sprintf("  %5s  %10.5f  %14.2e  %14.2e", iter_key, t_yr, vn, maxdiff)

    if idx_i ∈ plot_indices
        snapshots[idx_i] = (
            age_serial_yr = Array(age_serial_raw) ./ year,
            age_dist_yr = age_dist_raw ./ year,
            age_diff_yr = (age_dist_raw .- Array(age_serial_raw)) ./ year,
            t_yr = t_yr,
        )
    end
end
flush(stdout); flush(stderr)

# Use final snapshot for summary statistics
age_serial_yr = snapshots[NITERS].age_serial_yr
age_dist_yr = snapshots[NITERS].age_dist_yr
age_diff_yr = snapshots[NITERS].age_diff_yr

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

for idx_i in plot_indices
    snap = snapshots[idx_i]
    iter_label = "iter$(iter_keys[idx_i])"
    @info "Plotting $iter_label (t = $(@sprintf("%.5f", snap.t_yr)) yr)"

    compare_k_indices = 30:50

    # Serial age (reference)
    plot_age_diagnostics(
        snap.age_serial_yr, grid, wet3D, vol_3D, plot_output_dir,
        "serial_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(iter_label)";
        colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
        target_k_indices = compare_k_indices,
    )

    # Distributed age
    plot_age_diagnostics(
        snap.age_dist_yr, grid, wet3D, vol_3D, plot_output_dir,
        "distributed_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(iter_label)";
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
        "diff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(iter_label)";
        colorrange = diff_range, levels = diff_levels,
        colormap = cgrad(:balance, n_levels - 1, categorical = true),
        lowclip = :blue, highclip = :red,
        target_k_indices = compare_k_indices,
        colorbar_label = "Δage (years)",
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
            "reldiff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)_$(iter_label)";
            colorrange = reldiff_range, levels = reldiff_levels,
            colormap = cgrad(:balance, n_levels - 1, categorical = true),
            lowclip = :blue, highclip = :red,
            target_k_indices = compare_k_indices,
            colorbar_label = "Δage / age",
        )
    end

    flush(stdout); flush(stderr)
end

################################################################################
# Compare velocity and eta fields (iter 1 = first timestep after t=0)
################################################################################

# Extract grid coordinates for heatmap plotting
ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
lon = Array(ug.λᶜᶜᵃ[1:Nx, 1:Ny])
lat = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])

velocity_fields = ["u", "v", "w", "eta"]
vel_iter_key = NITERS >= 2 ? iter_keys[2] : iter_keys[1]  # first timestep after t=0

# Check if velocity files exist (they won't if distributed was GPU-only)
serial_vel_exists = isfile(joinpath(serial_dir, "u_$(DURATION_TAG).jld2"))
dist_vel_exists = isfile(joinpath(distributed_dir, "u_$(DURATION_TAG)_rank0.jld2"))

if serial_vel_exists && dist_vel_exists
    @info "Comparing velocity/eta fields at iter $vel_iter_key"
    flush(stdout); flush(stderr)

    for field_name in velocity_fields
        serial_file = joinpath(serial_dir, "$(field_name)_$(DURATION_TAG).jld2")
        dist_file = joinpath(distributed_dir, "$(field_name)_$(DURATION_TAG)_rank0.jld2")
        if !isfile(serial_file) || !isfile(dist_file)
            @warn "Skipping $field_name: missing serial or distributed file"
            continue
        end

        field_serial_full, t_s = load_serial_snapshot(serial_dir, field_name, DURATION_TAG, vel_iter_key)
        field_dist_full, t_d = load_distributed_snapshot(distributed_dir, field_name, DURATION_TAG, vel_iter_key, px, py, Nx, Ny; halo = (Hx, Hy, Hz))
        t_yr = t_s / year

        # Extract interior from halo-inclusive serial data
        nz_field = size(field_dist_full, 3)
        field_serial = extract_interior(field_serial_full, Hx, Hy, Hz, Nx, Ny, nz_field)
        field_dist = field_dist_full[1:Nx, 1:Ny, 1:nz_field]
        field_diff = field_dist .- field_serial

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
            serial_slice = Array(field_serial[:, :, k])
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

                # Use tight fixed range for w to see interior structure (auto range is dominated by halo outliers)
                diff_scale = field_name == "w" ? 1.0e-10 : (mean_abs > 0 ? 3 * mean_abs : 1.0e-10)
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

halo_fields = ["u", "v", "w", "eta", "age"]
halo_iter_key = vel_iter_key  # same iteration as velocity comparison

@info "Plotting halo-inclusive fields (fold diagnostic) at iter $halo_iter_key"
flush(stdout); flush(stderr)

"""Load raw data from JLD2Writer file at a specific iteration. Returns (parent_array, time)."""
function load_with_halos(filepath, field_name, iter_key)
    return jldopen(filepath, "r") do f
        raw = f["timeseries/$(field_name)/$iter_key"]
        t = f["timeseries/t/$iter_key"]
        return raw, t
    end
end

for field_name in halo_fields
    serial_file = joinpath(serial_dir, "$(field_name)_$(DURATION_TAG).jld2")
    if !isfile(serial_file)
        @warn "Skipping $field_name halo plot: no serial file"
        continue
    end

    serial_halo, t_s = load_with_halos(serial_file, field_name, halo_iter_key)
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
    serial_slice = serial_halo[:, :, k_halo]
    if field_name in ("u", "v", "w")
        vmax = maximum(abs, filter(!isnan, serial_slice))
        vmax = vmax > 0 ? vmax : 1.0e-10
        hm = heatmap!(
            ax, 1:nx_h, 1:ny_h, serial_slice;
            colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
        )
    else
        hm = heatmap!(ax, 1:nx_h, 1:ny_h, serial_slice; nan_color = :lightgray)
    end
    Colorbar(fig[1, 2], hm; label = field_name)
    save(joinpath(plot_output_dir, "$(field_name)_serial_halos_$(DURATION_TAG)_k$(k_halo).png"), fig)

    # Load all rank data
    rank_halos = Dict{Int, Array}()
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry  # Oceananigans rank ordering: rank = i * Ry + j
        dist_file = joinpath(distributed_dir, "$(field_name)_$(DURATION_TAG)_rank$(r).jld2")
        if !isfile(dist_file)
            @warn "Missing distributed file for rank $r: $dist_file"
            continue
        end
        rank_halo, _ = load_with_halos(dist_file, field_name, halo_iter_key)
        rank_halos[r] = rank_halo
    end

    # Infer halo size from serial parent vs interior.
    # Serial parent size = interior_size + 2*H, where interior_size depends on field location.
    # For Center-y (u, w, age): interior Ny = Ny (300). For Face-y (v): interior Ny = Ny+1 (301).
    # We compute Hx, Hy from the serial data directly.
    serial_interior_nx = Nx  # always Nx for Periodic x
    serial_interior_ny = ny_h > Ny ? ny_h : Ny  # will compute below
    # Get Hx from x: parent_nx = Nx + 2*Hx
    Hx = (nx_h - Nx) ÷ 2
    # Get Hy from y: need to know the interior ny for this field's y-location
    # u (Face,Center): interior ny = Ny; v (Center,Face): interior ny = Ny+1;
    # w (Center,Center,Face): interior ny = Ny; age/eta (Center,Center): interior ny = Ny
    field_interior_ny = ny_h - 2 * ((nx_h - Nx) ÷ 2)  # assume Hy == Hx (both are 7)
    Hy = Hx  # halos are symmetric

    # Determine interior sizes per rank from parent sizes
    rank_interior_nx = zeros(Int, px)
    rank_interior_ny = zeros(Int, py)
    for rx in 0:(px - 1)
        r0 = rx * py  # first rank in this x-column
        if haskey(rank_halos, r0)
            rank_interior_nx[rx + 1] = size(rank_halos[r0], 1) - 2 * Hx
        end
    end
    for ry in 0:(py - 1)
        r0 = ry  # first rank in this y-row (rx=0)
        if haskey(rank_halos, r0)
            rank_interior_ny[ry + 1] = size(rank_halos[r0], 2) - 2 * Hy
        end
    end
    x_offsets = cumsum([0; rank_interior_nx[1:(end - 1)]])
    y_offsets = cumsum([0; rank_interior_ny[1:(end - 1)]])

    # Use symmetric blue-red colormap for velocity fields
    is_velocity = field_name in ("u", "v", "w")

    # Plot each rank with halos + diff against serial
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry  # Oceananigans rank ordering: rank = i * Ry + j
        haskey(rank_halos, r) || continue
        rank_halo = rank_halos[r]
        nxr, nyr = size(rank_halo, 1), size(rank_halo, 2)
        k_r = min(k_halo, size(rank_halo, 3))
        @info "  $field_name rank $r ($(rx),$(ry)): parent size=$(size(rank_halo)), plotting k=$k_r"

        # Rank value plot
        rank_slice = rank_halo[:, :, k_r]
        fig = Figure(; size = (1000, 600))
        ax = Axis(
            fig[1, 1];
            title = "$field_name rank $r ($(rx),$(ry)) WITH HALOS (k=$k_r, size=$(size(rank_halo)))",
            xlabel = "i (with halos)", ylabel = "j (with halos)",
            backgroundcolor = :lightgray,
        )
        if is_velocity
            vmax = maximum(abs, filter(!isnan, rank_slice))
            vmax = vmax > 0 ? vmax : 1.0e-10
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, rank_slice;
                colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
            )
        else
            hm = heatmap!(ax, 1:nxr, 1:nyr, rank_slice; nan_color = :lightgray)
        end
        Colorbar(fig[1, 2], hm; label = field_name)
        save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_halos_$(DURATION_TAG)_k$(k_r).png"), fig)

        # Diff: slice serial parent to match rank's parent (including halos).
        # Rank parent[1..nxr] maps to serial parent[x_offsets[rx]+1 .. x_offsets[rx]+nxr].
        i_start = x_offsets[rx + 1] + 1
        j_start = y_offsets[ry + 1] + 1
        i_end = i_start + nxr - 1
        j_end = j_start + nyr - 1

        if i_end <= nx_h && j_end <= ny_h
            serial_slice = serial_halo[i_start:i_end, j_start:j_end, k_r]
            diff_slice = rank_slice .- serial_slice
            max_abs = maximum(abs, diff_slice)
            mean_abs = mean(abs, diff_slice)
            @info "  $field_name rank $r diff: max|diff|=$(@sprintf("%.2e", max_abs)) mean|diff|=$(@sprintf("%.2e", mean_abs))"

            diff_scale = field_name == "w" ? 1.0e-10 : (max_abs > 0 ? max_abs : 1.0e-10)
            fig = Figure(; size = (1000, 600))
            ax = Axis(
                fig[1, 1];
                title = "$field_name rank $r DIFF (dist-serial) WITH HALOS (k=$k_r, max=$(@sprintf("%.2e", max_abs)))",
                xlabel = "i (with halos)", ylabel = "j (with halos)",
                backgroundcolor = :lightgray,
            )
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, diff_slice;
                colorrange = (-diff_scale, diff_scale),
                colormap = :balance, nan_color = :lightgray,
                lowclip = :blue, highclip = :red,
            )
            Colorbar(fig[1, 2], hm; label = "$field_name diff")
            save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_diff_halos_$(DURATION_TAG)_k$(k_r).png"), fig)
        else
            @warn "  $field_name rank $r: serial slice out of bounds ($(i_start):$(i_end), $(j_start):$(j_end)) vs serial size $(nx_h)×$(ny_h)"
        end
    end

    @info "  $field_name: halo plots saved (serial + $(px * py) ranks, with diffs)"
    flush(stdout); flush(stderr)
end

################################################################################
# Manual callback field plots (true model halos, bypassing JLD2Writer)
################################################################################

# The JLD2Writer wraps fields in anonymous ComputedFields whose fill_halo_regions!
# uses default sign=+1 BCs (the wrapper isn't named :u/:v so it can't dispatch to
# sign=-1 in regularize_field_boundary_conditions). The manual callback saves
# Array(parent(field.data)) directly, capturing the TRUE model field halos.

manual_fields = ["u", "v", "w", "eta"]

@info "Plotting manual callback fields (true model halos)"
flush(stdout); flush(stderr)

for field_name in manual_fields
    serial_manual = joinpath(serial_dir, "$(field_name)_manual_$(DURATION_TAG).jld2")
    if !isfile(serial_manual)
        @info "  $field_name: no manual callback file — skipping"
        continue
    end

    # Load serial manual data (first non-zero iteration)
    serial_data, t_s = jldopen(serial_manual, "r") do f
        iters = collect(keys(f["timeseries/$(field_name)"]))
        iter = first(sort(iters; by = k -> parse(Int, k)))
        f["timeseries/$(field_name)/$iter"], f["timeseries/t/$iter"]
    end
    t_yr = t_s / year

    nz_m = size(serial_data, 3)
    k_m = field_name == "eta" ? nz_m ÷ 2 + 1 : nz_m - 7
    nx_m, ny_m = size(serial_data, 1), size(serial_data, 2)

    @info "  $field_name manual serial: size=$(size(serial_data)), plotting k=$k_m"

    # Serial manual plot
    serial_slice = serial_data[:, :, k_m]
    fig = Figure(; size = (1000, 600))
    ax = Axis(
        fig[1, 1];
        title = "$field_name MANUAL serial (t=$(@sprintf("%.5f", t_yr)) yr, k=$k_m)",
        xlabel = "i (with halos)", ylabel = "j (with halos)",
        backgroundcolor = :lightgray,
    )
    if field_name in ("u", "v", "w")
        vmax = maximum(abs, filter(!isnan, serial_slice))
        vmax = vmax > 0 ? vmax : 1.0e-10
        hm = heatmap!(
            ax, 1:nx_m, 1:ny_m, serial_slice;
            colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
        )
    else
        hm = heatmap!(ax, 1:nx_m, 1:ny_m, serial_slice; nan_color = :lightgray)
    end
    Colorbar(fig[1, 2], hm; label = field_name)
    save(joinpath(plot_output_dir, "$(field_name)_serial_manualcallback_$(DURATION_TAG)_k$(k_m).png"), fig)

    # Collect per-rank interior sizes for diff slicing
    rank_int_nxs = zeros(Int, px)
    rank_int_nys = zeros(Int, py)
    for rx2 in 0:(px - 1)
        r2 = rx2 * py
        f2 = joinpath(distributed_dir, "$(field_name)_manual_$(DURATION_TAG)_rank$(r2).jld2")
        if isfile(f2)
            jldopen(f2, "r") do f
                iters = collect(keys(f["timeseries/$(field_name)"]))
                iter = first(sort(iters; by = k -> parse(Int, k)))
                rank_int_nxs[rx2 + 1] = size(f["timeseries/$(field_name)/$iter"], 1) - 2Hx
            end
        end
    end
    for ry2 in 0:(py - 1)
        f2 = joinpath(distributed_dir, "$(field_name)_manual_$(DURATION_TAG)_rank$(ry2).jld2")
        if isfile(f2)
            jldopen(f2, "r") do f
                iters = collect(keys(f["timeseries/$(field_name)"]))
                iter = first(sort(iters; by = k -> parse(Int, k)))
                rank_int_nys[ry2 + 1] = size(f["timeseries/$(field_name)/$iter"], 2) - 2Hy
            end
        end
    end

    # Plot each distributed rank + diff
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry
        dist_manual = joinpath(distributed_dir, "$(field_name)_manual_$(DURATION_TAG)_rank$(r).jld2")
        if !isfile(dist_manual)
            @warn "  $field_name manual rank $r: file not found"
            continue
        end

        rank_data, _ = jldopen(dist_manual, "r") do f
            iters = collect(keys(f["timeseries/$(field_name)"]))
            iter = first(sort(iters; by = k -> parse(Int, k)))
            f["timeseries/$(field_name)/$iter"], f["timeseries/t/$iter"]
        end

        nxr, nyr = size(rank_data, 1), size(rank_data, 2)
        k_r = min(k_m, size(rank_data, 3))

        # Rank value plot
        rank_slice = rank_data[:, :, k_r]
        fig = Figure(; size = (1000, 600))
        ax = Axis(
            fig[1, 1];
            title = "$field_name MANUAL rank $r ($(rx),$(ry)) (k=$k_r)",
            xlabel = "i (with halos)", ylabel = "j (with halos)",
            backgroundcolor = :lightgray,
        )
        if field_name in ("u", "v", "w")
            vmax = maximum(abs, filter(!isnan, rank_slice))
            vmax = vmax > 0 ? vmax : 1.0e-10
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, rank_slice;
                colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
            )
        else
            hm = heatmap!(ax, 1:nxr, 1:nyr, rank_slice; nan_color = :lightgray)
        end
        Colorbar(fig[1, 2], hm; label = field_name)
        save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_manualcallback_$(DURATION_TAG)_k$(k_r).png"), fig)

        # Diff against serial
        i_start = sum(rank_int_nxs[1:rx]) + 1
        j_start = sum(rank_int_nys[1:ry]) + 1
        i_end = i_start + nxr - 1
        j_end = j_start + nyr - 1

        if i_end <= nx_m && j_end <= ny_m
            serial_match = serial_data[i_start:i_end, j_start:j_end, k_r]
            diff_slice = rank_slice .- serial_match
            max_abs = maximum(abs, diff_slice)
            mean_abs = mean(abs, diff_slice)
            @info "  $field_name manual rank $r diff: max|diff|=$(@sprintf("%.2e", max_abs)) mean|diff|=$(@sprintf("%.2e", mean_abs))"

            diff_scale = field_name == "w" ? 1.0e-10 : (max_abs > 0 ? max_abs : 1.0e-10)
            fig = Figure(; size = (1000, 600))
            ax = Axis(
                fig[1, 1];
                title = "$field_name MANUAL rank $r DIFF (k=$k_r, max=$(@sprintf("%.2e", max_abs)))",
                xlabel = "i (with halos)", ylabel = "j (with halos)",
                backgroundcolor = :lightgray,
            )
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, diff_slice;
                colorrange = (-diff_scale, diff_scale),
                colormap = :balance, nan_color = :lightgray,
                lowclip = :blue, highclip = :red,
            )
            Colorbar(fig[1, 2], hm; label = "$field_name diff")
            save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_diff_manualcallback_$(DURATION_TAG)_k$(k_r).png"), fig)
        else
            @warn "  $field_name manual rank $r: serial slice out of bounds"
        end
    end

    @info "  $field_name: manual callback plots saved"
    flush(stdout); flush(stderr)
end

################################################################################
# FTS snapshot plots (raw halos from load_fts, bypassing TSI interpolation)
################################################################################

fts_fields = ["u", "v", "eta"]

@info "Plotting FTS snapshot fields (raw halos from load_fts)"
flush(stdout); flush(stderr)

for field_name in fts_fields
    serial_fts_file = joinpath(serial_dir, "$(field_name)_fts_$(DURATION_TAG).jld2")
    if !isfile(serial_fts_file)
        @info "  $field_name: no FTS file — skipping"
        continue
    end

    serial_data, t_s = jldopen(serial_fts_file, "r") do f
        iters = collect(keys(f["timeseries/$(field_name)"]))
        iter = first(sort(iters; by = k -> parse(Int, k)))
        f["timeseries/$(field_name)/$iter"], f["timeseries/t/$iter"]
    end
    t_yr = t_s / year

    nz_f = size(serial_data, 3)
    k_f = field_name == "eta" ? nz_f ÷ 2 + 1 : nz_f - 7
    nx_f, ny_f = size(serial_data, 1), size(serial_data, 2)

    @info "  $field_name FTS serial: size=$(size(serial_data)), plotting k=$k_f"

    # Serial FTS plot
    serial_slice = serial_data[:, :, k_f]
    fig = Figure(; size = (1000, 600))
    ax = Axis(
        fig[1, 1];
        title = "$field_name FTS serial (k=$k_f, size=$(size(serial_data)))",
        xlabel = "i (with halos)", ylabel = "j (with halos)",
        backgroundcolor = :lightgray,
    )
    if field_name in ("u", "v")
        vmax = maximum(abs, filter(!isnan, serial_slice))
        vmax = vmax > 0 ? vmax : 1.0e-10
        hm = heatmap!(
            ax, 1:nx_f, 1:ny_f, serial_slice;
            colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
        )
    else
        hm = heatmap!(ax, 1:nx_f, 1:ny_f, serial_slice; nan_color = :lightgray)
    end
    Colorbar(fig[1, 2], hm; label = field_name)
    save(joinpath(plot_output_dir, "$(field_name)_serial_fts_$(DURATION_TAG)_k$(k_f).png"), fig)

    # Distributed rank FTS plots + diff
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry
        dist_fts_file = joinpath(distributed_dir, "$(field_name)_fts_$(DURATION_TAG)_rank$(r).jld2")
        if !isfile(dist_fts_file)
            @warn "  $field_name FTS rank $r: file not found"
            continue
        end

        rank_data, _ = jldopen(dist_fts_file, "r") do f
            iters = collect(keys(f["timeseries/$(field_name)"]))
            iter = first(sort(iters; by = k -> parse(Int, k)))
            f["timeseries/$(field_name)/$iter"], f["timeseries/t/$iter"]
        end

        nxr, nyr = size(rank_data, 1), size(rank_data, 2)
        k_r = min(k_f, size(rank_data, 3))

        # Rank value plot
        rank_slice = rank_data[:, :, k_r]
        fig = Figure(; size = (1000, 600))
        ax = Axis(
            fig[1, 1];
            title = "$field_name FTS rank $r ($(rx),$(ry)) (k=$k_r)",
            xlabel = "i (with halos)", ylabel = "j (with halos)",
            backgroundcolor = :lightgray,
        )
        if field_name in ("u", "v")
            vmax = maximum(abs, filter(!isnan, rank_slice))
            vmax = vmax > 0 ? vmax : 1.0e-10
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, rank_slice;
                colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray
            )
        else
            hm = heatmap!(ax, 1:nxr, 1:nyr, rank_slice; nan_color = :lightgray)
        end
        Colorbar(fig[1, 2], hm; label = field_name)
        save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_fts_$(DURATION_TAG)_k$(k_r).png"), fig)

        # Diff against serial (slice serial to match rank)
        rank_int_nx = nxr - 2Hx
        rank_int_ny = nyr - 2Hy
        # Compute rank interior offsets
        rank_nxs = zeros(Int, px)
        rank_nys = zeros(Int, py)
        for rx2 in 0:(px - 1)
            f2 = joinpath(distributed_dir, "$(field_name)_fts_$(DURATION_TAG)_rank$(rx2 * py).jld2")
            isfile(f2) && jldopen(f2, "r") do f
                it = first(sort(filter(k -> k != "serialized", collect(keys(f["timeseries/$(field_name)"]))); by = k -> parse(Int, k)))
                rank_nxs[rx2 + 1] = size(f["timeseries/$(field_name)/$it"], 1) - 2Hx
            end
        end
        for ry2 in 0:(py - 1)
            f2 = joinpath(distributed_dir, "$(field_name)_fts_$(DURATION_TAG)_rank$(ry2).jld2")
            isfile(f2) && jldopen(f2, "r") do f
                it = first(sort(filter(k -> k != "serialized", collect(keys(f["timeseries/$(field_name)"]))); by = k -> parse(Int, k)))
                rank_nys[ry2 + 1] = size(f["timeseries/$(field_name)/$it"], 2) - 2Hy
            end
        end
        i_start = sum(rank_nxs[1:rx]) + 1
        j_start = sum(rank_nys[1:ry]) + 1
        i_end = i_start + nxr - 1
        j_end = j_start + nyr - 1

        if i_end <= nx_f && j_end <= ny_f
            serial_match = serial_data[i_start:i_end, j_start:j_end, k_r]
            diff_slice = rank_slice .- serial_match
            max_abs = maximum(abs, diff_slice)
            mean_abs = mean(abs, diff_slice)
            @info "  $field_name FTS rank $r diff: max|diff|=$(@sprintf("%.2e", max_abs)) mean|diff|=$(@sprintf("%.2e", mean_abs))"

            diff_scale = field_name == "w" ? 1.0e-10 : (max_abs > 0 ? max_abs : 1.0e-10)
            fig = Figure(; size = (1000, 600))
            ax = Axis(
                fig[1, 1];
                title = "$field_name FTS rank $r DIFF (k=$k_r, max=$(@sprintf("%.2e", max_abs)))",
                xlabel = "i (with halos)", ylabel = "j (with halos)",
                backgroundcolor = :lightgray,
            )
            hm = heatmap!(
                ax, 1:nxr, 1:nyr, diff_slice;
                colorrange = (-diff_scale, diff_scale),
                colormap = :balance, nan_color = :lightgray,
                lowclip = :blue, highclip = :red,
            )
            Colorbar(fig[1, 2], hm; label = "$field_name diff")
            save(joinpath(plot_output_dir, "$(field_name)_rank$(r)_diff_fts_$(DURATION_TAG)_k$(k_r).png"), fig)
        end
    end

    @info "  $field_name: FTS plots saved"
    flush(stdout); flush(stderr)
end

################################################################################
# Bottom height plots
################################################################################

serial_bottom_file = joinpath(serial_dir, "bottom_$(DURATION_TAG).jld2")
dist_bottom_file = joinpath(distributed_dir, "bottom_$(DURATION_TAG)_rank0.jld2")

if isfile(serial_bottom_file)
    @info "Plotting bottom height"
    serial_bottom = jldopen(serial_bottom_file, "r") do f
        f["bottom"]
    end
    nx_b, ny_b = size(serial_bottom, 1), size(serial_bottom, 2)
    @info "  Serial bottom: size=$(size(serial_bottom))"

    fig = Figure(; size = (1000, 600))
    ax = Axis(
        fig[1, 1];
        title = "Bottom height serial (size=$(size(serial_bottom)))",
        xlabel = "i (with halos)", ylabel = "j (with halos)",
        backgroundcolor = :lightgray,
    )
    hm = heatmap!(ax, 1:nx_b, 1:ny_b, serial_bottom[:, :, 1]; nan_color = :lightgray)
    Colorbar(fig[1, 2], hm; label = "bottom (m)")
    save(joinpath(plot_output_dir, "bottom_serial_$(DURATION_TAG).png"), fig)

    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry
        df = joinpath(distributed_dir, "bottom_$(DURATION_TAG)_rank$(r).jld2")
        if !isfile(df)
            continue
        end
        rank_bottom = jldopen(df, "r") do f
            f["bottom"]
        end
        nxr, nyr = size(rank_bottom, 1), size(rank_bottom, 2)

        fig = Figure(; size = (1000, 600))
        ax = Axis(
            fig[1, 1];
            title = "Bottom height rank $r ($(rx),$(ry)) (size=$(size(rank_bottom)))",
            xlabel = "i (with halos)", ylabel = "j (with halos)",
            backgroundcolor = :lightgray,
        )
        hm = heatmap!(ax, 1:nxr, 1:nyr, rank_bottom[:, :, 1]; nan_color = :lightgray)
        Colorbar(fig[1, 2], hm; label = "bottom (m)")
        save(joinpath(plot_output_dir, "bottom_rank$(r)_$(DURATION_TAG).png"), fig)
    end
    @info "  Bottom plots saved"
    flush(stdout); flush(stderr)
else
    @info "No bottom height file — skipping"
end

################################################################################
# z-star field comparison (sigma_cc, dt_sigma, eta_n) per iteration
################################################################################
# These are saved by save_zstar_fields in setup_simulation.jl as per-iteration
# snapshots of the ug.z MutableVerticalDiscretization state. We compare them
# at every iteration to detect when distributed and serial first diverge.

zstar_fields = ["sigma_cc", "dt_sigma", "eta_n"]

@info "Comparing z-star callback fields per iteration (serial vs distributed)"
flush(stdout); flush(stderr)

for zname in zstar_fields
    serial_zfile = joinpath(serial_dir, "$(zname)_$(DURATION_TAG).jld2")
    if !isfile(serial_zfile)
        @info "  $zname: no serial file — skipping"
        continue
    end

    # Read all iterations from the serial file
    serial_iters, serial_data = jldopen(serial_zfile, "r") do f
        iters = sort(parse.(Int, collect(keys(f["timeseries/$zname"]))))
        data = [f["timeseries/$zname/$it"] for it in iters]
        (iters, data)
    end

    @info "  $zname: $(length(serial_iters)) iterations found"

    # For each rank, load its data and compare to serial at each iteration
    any_mismatch = false
    for ry in 0:(py - 1), rx in 0:(px - 1)
        r = rx * py + ry
        rank_zfile = joinpath(distributed_dir, "$(zname)_$(DURATION_TAG)_rank$(r).jld2")
        if !isfile(rank_zfile)
            @info "    rank $r: no file — skipping"
            continue
        end

        rank_iters, rank_data = jldopen(rank_zfile, "r") do f
            iters = sort(parse.(Int, collect(keys(f["timeseries/$zname"]))))
            data = [f["timeseries/$zname/$it"] for it in iters]
            (iters, data)
        end

        # Compare at shared iterations
        common = intersect(serial_iters, rank_iters)
        first_div = -1
        max_diff_over_all = 0.0
        for it in common
            si = findfirst(==(it), serial_iters)
            ri = findfirst(==(it), rank_iters)
            sd = serial_data[si]
            rd = rank_data[ri]

            # If shapes match, compare directly (global-sized z-star field).
            # Otherwise slice the serial using offsets derived from even
            # partitioning (Ny/py per rank in y, Nx/px per rank in x).
            local sd_slice
            if size(sd) == size(rd)
                sd_slice = sd
            else
                nx_s, ny_s = size(sd, 1), size(sd, 2)
                nx_r, ny_r = size(rd, 1), size(rd, 2)
                # Compute offsets from even partition of interior cells
                # (each rank gets Ny÷py cells, offset = ry * Ny÷py)
                Hx_z = (nx_s - Nx) ÷ 2
                Hy_z = Hx_z  # halos are symmetric
                x_off = rx * (Nx ÷ px)
                y_off = ry * (Ny ÷ py)
                x_start = 1 + x_off
                y_start = 1 + y_off
                x_end = x_start + nx_r - 1
                y_end = y_start + ny_r - 1
                if x_end > nx_s || y_end > ny_s
                    @warn "    $zname iter $it rank $r slice out of bounds — skipping"
                    continue
                end
                sd_slice = ndims(sd) == 3 ? sd[x_start:x_end, y_start:y_end, :] : sd[x_start:x_end, y_start:y_end]
            end

            diff = rd .- sd_slice
            absdiff = abs.(filter(!isnan, diff))
            max_abs = isempty(absdiff) ? 0.0 : maximum(absdiff)
            if max_abs > max_diff_over_all
                max_diff_over_all = max_abs
            end
            if max_abs > 0 && first_div == -1
                first_div = it
            end
        end

        if first_div == -1
            @info "    rank $r: all $(length(common)) iterations bit-identical (max|diff|=0)"
        else
            any_mismatch = true
            @info "    rank $r: FIRST DIVERGENCE at iter $first_div, overall max|diff|=$(@sprintf("%.3e", max_diff_over_all)) over $(length(common)) iterations"
        end

        # Plot spatial diff at the LAST shared iteration
        if !isempty(common)
            last_it = last(common)
            si_last = findfirst(==(last_it), serial_iters)
            ri_last = findfirst(==(last_it), rank_iters)
            sd_last = serial_data[si_last]
            rd_last = rank_data[ri_last]

            local sd_slice_last
            if size(sd_last) == size(rd_last)
                sd_slice_last = sd_last
            else
                nx_s2, ny_s2 = size(sd_last, 1), size(sd_last, 2)
                nx_r2, ny_r2 = size(rd_last, 1), size(rd_last, 2)
                x_off2 = rx * (Nx ÷ px)
                y_off2 = ry * (Ny ÷ py)
                x_s2 = 1 + x_off2
                y_s2 = 1 + y_off2
                x_e2 = x_s2 + nx_r2 - 1
                y_e2 = y_s2 + ny_r2 - 1
                sd_slice_last = ndims(sd_last) == 3 ? sd_last[x_s2:x_e2, y_s2:y_e2, :] : sd_last[x_s2:x_e2, y_s2:y_e2]
            end

            if size(sd_slice_last) == size(rd_last)
                diff_last = rd_last .- sd_slice_last
                # Pick a z-slice (these are 2D+1 fields; take the only z-slice)
                k_z = 1
                diff_2d = ndims(diff_last) == 3 ? diff_last[:, :, k_z] : diff_last
                rd_2d = ndims(rd_last) == 3 ? rd_last[:, :, k_z] : rd_last
                sd_2d = ndims(sd_slice_last) == 3 ? sd_slice_last[:, :, k_z] : sd_slice_last
                nxp, nyp = size(diff_2d)

                absdiff_last = abs.(filter(!isnan, diff_2d))
                max_abs_last = isempty(absdiff_last) ? 0.0 : maximum(absdiff_last)
                mean_serial = mean(abs, filter(!isnan, sd_2d))
                mean_dist = mean(abs, filter(!isnan, rd_2d))

                @info @sprintf(
                    "    rank %d %s iter %d: max|diff|=%.3e mean|serial|=%.3e mean|dist|=%.3e",
                    r, zname, last_it, max_abs_last, mean_serial, mean_dist,
                )

                diff_scale_z = max_abs_last > 0 ? max_abs_last : 1.0e-10
                fig = Figure(; size = (1000, 600))
                ax = Axis(
                    fig[1, 1];
                    title = "$zname rank $r DIFF iter $last_it (max=$(@sprintf("%.2e", max_abs_last)))",
                    xlabel = "i (with halos)", ylabel = "j (with halos)",
                    backgroundcolor = :lightgray,
                )
                hm = heatmap!(
                    ax, 1:nxp, 1:nyp, diff_2d;
                    colorrange = (-diff_scale_z, diff_scale_z),
                    colormap = :balance, nan_color = :lightgray,
                    lowclip = :blue, highclip = :red,
                )
                Colorbar(fig[1, 2], hm; label = "$zname diff")
                save(joinpath(plot_output_dir, "$(zname)_rank$(r)_diff_$(DURATION_TAG)_iter$(last_it).png"), fig)

                # Also plot the rank's raw value for reference
                fig2 = Figure(; size = (1000, 600))
                ax2 = Axis(
                    fig2[1, 1];
                    title = "$zname rank $r (iter $last_it)",
                    xlabel = "i (with halos)", ylabel = "j (with halos)",
                    backgroundcolor = :lightgray,
                )
                hm2 = heatmap!(ax2, 1:nxp, 1:nyp, rd_2d; nan_color = :lightgray)
                Colorbar(fig2[1, 2], hm2; label = zname)
                save(joinpath(plot_output_dir, "$(zname)_rank$(r)_$(DURATION_TAG)_iter$(last_it).png"), fig2)
            end
        end
    end

    flush(stdout); flush(stderr)
end

@info "compare_runs_across_architectures.jl complete (GPU_TAG=$GPU_TAG, DURATION_TAG=$DURATION_TAG)"
flush(stdout); flush(stderr)
