################################################################################
# Analysis utilities and plotting
#
# Extracted from shared_functions.jl — plotting prep helpers, ocean basin masks,
# zonal averages, horizontal slices, age diagnostic plots, and animation.
################################################################################

################################################################################
# Shared axis tick/label helpers
################################################################################

"""
    latticklabel(lat) -> String

Format a latitude value as "30°S", "0°", "30°N", etc.
"""
function latticklabel(lat)
    lat = isinteger(lat) ? Int(lat) : lat
    return if lat == 0
        "0°"
    elseif lat > 0
        "$(lat)°N"
    else
        "$(-lat)°S"
    end
end

const DEPTH_YLIM = (6000, 0)
const DEPTH_YTICKS = 0:1000:6000
const DEPTH_YTICKLABELS = string.(DEPTH_YTICKS)

const MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Grids: on_architecture, znodes
using Statistics: mean, median, quantile
using Printf: @sprintf

#taken from Makie ext (not sure how to load these)
function drop_singleton_indices(N)
    if N == 1
        return 1
    else
        return Colon()
    end
end
function make_plottable_array(f)
    compute!(f)
    mask_immersed_field!(f, NaN)

    Nx, Ny, Nz = size(f)

    ii = drop_singleton_indices(Nx)
    jj = drop_singleton_indices(Ny)
    kk = drop_singleton_indices(Nz)

    fi_cpu = Array(interior(f, ii, jj, kk))

    if architecture(f) isa CPU
        fi_cpu = deepcopy(fi_cpu) # so we can re-zero peripheral nodes
    end

    mask_immersed_field!(f)

    return fi_cpu
end


################################################################################
# Analysis utilities: zonal averages and horizontal slices
################################################################################

"""
    compute_ocean_basin_masks(grid, wet3D) -> (; ATL, PAC, IND)

Compute Atlantic, Pacific, and Indian ocean basin masks using OceanBasins.jl.
Returns a named tuple of 2D Bool arrays sized (Nx', Ny') where (Nx', Ny')
are the interior dimensions from `wet3D` (excludes tripolar fold point).

Requires `OCEANS, isatlantic, ispacific, isindian` from OceanBasins in scope.
"""
function compute_ocean_basin_masks(grid, wet3D)
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′ = size(wet3D)[1:2]
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    lon = Array(ug.λᶜᶜᵃ[1:Nx′, 1:Ny′])

    flat_lat = vec(lat)
    flat_lon = vec(lon)
    ATL = reshape(isatlantic(flat_lat, flat_lon, OCEANS), size(lat))
    PAC = reshape(ispacific(flat_lat, flat_lon, OCEANS), size(lat))
    IND = reshape(isindian(flat_lat, flat_lon, OCEANS), size(lat))

    return (; ATL, PAC, IND)
end

"""
    zonalaverage(x3D, v3D, mask)

Volume-weighted zonal average (average along dimension 1).
`mask` is a 2D or 3D boolean array; 2D masks broadcast over depth.
NaN values in `x3D` are excluded. Returns a (Ny, Nz) matrix.
"""
function zonalaverage(x3D, v3D, mask)
    m = ndims(mask) == 2 ? reshape(mask, size(mask, 1), size(mask, 2), 1) : mask
    xw = @. ifelse(isnan(x3D) | !m, 0.0, x3D * v3D)
    w = @. ifelse(isnan(x3D) | !m, 0.0, v3D)
    num = dropdims(sum(xw; dims = 1); dims = 1)
    den = dropdims(sum(w; dims = 1); dims = 1)
    return @. ifelse(den > 0, num / den, NaN)
end

"""
    zonalaverage!(za, xw, w, x3D, v3D, mask3D)

In-place volume-weighted zonal average using preallocated buffers.
`mask3D` must be 3D (reshape 2D masks before calling).
Writes result into `za` (Ny, Nz) and uses `xw`, `w` as (Nx, Ny, Nz) scratch space.
"""
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

"""
    find_nearest_depth_index(grid, target_depth)

Return the k-index of the vertical level nearest to `target_depth` (m, positive downward).
"""
function find_nearest_depth_index(grid, target_depth)
    z = znodes(grid, Center(), Center(), Center())
    _, k = findmin(abs.(z .+ target_depth))
    return k
end


################################################################################
# Age diagnostic plots (10 figures)
#
# Requires CairoMakie and OceanBasins symbols in the calling script's scope.
################################################################################

"""
    plot_age_diagnostics(age_3D, grid, wet3D, vol_3D, output_dir, label;
                         colorrange=(0, 1500), levels=0:100:1500, colormap=:viridis)

Generate 10 diagnostic figures and save as PNG:
  1-4: Zonal average (global, Atlantic, Pacific, Indian) — contourf (lat vs depth)
  5-10: Horizontal slices at 100, 200, 500, 1000, 2000, 3000 m — heatmap

Arguments:
- `age_3D`:     (Nx', Ny', Nz') array (seconds) with 0 for dry cells
- `grid`:       ImmersedBoundaryGrid (tripolar)
- `wet3D`:      (Nx', Ny', Nz') Bool mask
- `vol_3D`:     (Nx', Ny', Nz') volume array (m^3)
- `output_dir`: directory for saving PNGs
- `label`:      filename prefix (e.g. "steady_age_full")
"""
function plot_age_diagnostics(
        age_3D, grid, wet3D, vol_3D, output_dir, label;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        lowclip = colormap[1],
        highclip = colormap[end],
        target_depths = [100, 200, 500, 1000, 2000, 3000],
        target_k_indices = Int[],
        colorbar_label = "Age (years)",
        title_prefix = label,
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    # Replace dry cells with NaN for plotting (and convert seconds → years)
    age_plot = copy(age_3D)
    age_plot[.!wet3D] .= NaN
    age_plot ./= year

    # Extract grid coordinates
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, Nz′ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z  # positive downward

    # Representative latitude for y-axis of zonal plots (mean along i)
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    # Compute basin masks
    basins = compute_ocean_basin_masks(grid, wet3D)
    global_mask = trues(Nx′, Ny′)

    # ── Zonal averages (figures 1-4) ──────────────────────────────────────

    basin_configs = [
        ("global", global_mask),
        ("atlantic", basins.ATL),
        ("pacific", basins.PAC),
        ("indian", basins.IND),
    ]

    for (basin_name, basin_mask) in basin_configs
        za = zonalaverage(age_plot, vol_3D, basin_mask)

        fig = Figure(; size = (800, 500))
        ax = Axis(
            fig[1, 1];
            title = "$title_prefix — $basin_name zonal average",
            xlabel = "Latitude",
            ylabel = "Depth (m)",
            backgroundcolor = :lightgray,
            xgridvisible = false,
            ygridvisible = false,
        )

        cf = contourf!(ax, lat_repr, depth_vals, za; levels, colormap, nan_color = :lightgray, extendhigh = :auto, extendlow = :auto)
        translate!(cf, 0, 0, -100)
        xlims!(ax, -90, 90)
        ylims!(ax, maximum(depth_vals), 0)
        Colorbar(fig[1, 2], cf; label = colorbar_label)

        outputfile = joinpath(output_dir, "$(label)_zonal_avg_$(basin_name).png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    # ── Horizontal slices (lon/lat maps via plotmap!) ────────────────────

    # Build the curvilinear gridmetrics once (shared by every depth slice).
    gridmetrics = gridmetrics_from_grid(grid, Nx′, Ny′)

    # Collect (k, file_tag, title_tag) tuples from target_depths and target_k_indices
    slice_specs = Tuple{Int, String, String}[]
    for depth in target_depths
        k = find_nearest_depth_index(grid, depth)
        actual_depth = round(depth_vals[k]; digits = 1)
        push!(slice_specs, (k, "$(depth)m", "$depth m (k=$k, z=$actual_depth m)"))
    end
    for k in target_k_indices
        actual_depth = round(depth_vals[k]; digits = 1)
        push!(slice_specs, (k, "k$(k)", "k=$k (z=$actual_depth m)"))
    end

    for (k, file_tag, title_tag) in slice_specs
        slice = age_plot[:, :, k]

        fig = Figure(; size = (1000, 500))
        ax = map_axis(fig[1, 1]; title = "$title_prefix at $title_tag")

        hm = plotmap!(ax, slice, gridmetrics; colorrange, colormap, lowclip, highclip)
        add_coastlines!(ax)
        Colorbar(fig[1, 2], hm; label = colorbar_label)

        outputfile = joinpath(output_dir, "$(label)_slice_$(file_tag).png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    @info "Age diagnostic plots saved to $output_dir"
    flush(stdout); flush(stderr)
    return nothing
end

"""
    plot_basin_zonal_panel(age_3D, grid, wet3D, vol_3D, output_dir, label;
                           colorrange=(0, 1500), levels=0:100:1500, colormap=...,
                           colorbar_label="Age (years)", title_prefix=label, lat_pad=5)

Single 1×3 figure of the Atlantic / Pacific / Indian volume-weighted zonal
averages (depth vs latitude), with each panel's width made proportional to its
basin's wet-cell latitude extent (the panels H/I/J pattern from the AIBECS
`2b_plot_makie.jl` how-to). Panels share linked depth axes and one horizontal
colorbar. Saves `{label}_zonal_avg_basins.png`.

Mirrors `plot_age_diagnostics` arguments; reuses `compute_ocean_basin_masks`
and `zonalaverage`.
"""
function plot_basin_zonal_panel(
        age_3D, grid, wet3D, vol_3D, output_dir, label;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        colorbar_label = "Age (years)",
        title_prefix = label,
        lat_pad = 5,
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    age_plot = copy(age_3D)
    age_plot[.!wet3D] .= NaN
    age_plot ./= year

    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, Nz′ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z  # positive downward
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    basins = compute_ocean_basin_masks(grid, wet3D)
    basin_configs = [
        ("Atlantic", basins.ATL),
        ("Pacific", basins.PAC),
        ("Indian", basins.IND),
    ]

    zas = [zonalaverage(age_plot, vol_3D, m) for (_, m) in basin_configs]

    # Wet-cell latitude range per basin (crop + size each column proportionally).
    function wetlatrange(za, pad)
        haswet = [any(!isnan, view(za, j, :)) for j in eachindex(lat_repr)]
        wetidx = findall(haswet)
        isempty(wetidx) && return (-90.0, 90.0)
        return (
            max(-90.0, lat_repr[first(wetidx)] - pad),
            min(90.0, lat_repr[last(wetidx)] + pad),
        )
    end
    xlims = [wetlatrange(za, lat_pad) for za in zas]

    maxdepth_round = ceil(maximum(depth_vals) / 1000) * 1000
    lat_tickvals = -90:30:90
    lat_ticks = (collect(lat_tickvals), latticklabel.(lat_tickvals))

    fig = Figure(; size = (1300, 520))
    gc = fig[1, 1] = GridLayout()

    axs = Axis[]
    local cf_ref = nothing
    for (col, ((basin_name, _), za, xl)) in enumerate(zip(basin_configs, zas, xlims))
        yposition = col == length(basin_configs) ? :right : :left
        ax = Axis(
            gc[1, col];
            backgroundcolor = :lightgray,
            xgridvisible = false, ygridvisible = false,
            xticksmirrored = true, yticksmirrored = true,
            xticks = lat_ticks,
            yaxisposition = yposition,
            yreversed = true,
            limits = (xl[1], xl[2], 0, maxdepth_round),
            ylabel = col == 1 ? "Depth (m)" : "",
        )
        cf = contourf!(
            ax, lat_repr, depth_vals, za;
            levels, colormap, nan_color = :lightgray,
            extendhigh = :auto, extendlow = :auto,
        )
        translate!(cf, 0, 0, -100)
        cf_ref = cf
        text!(
            ax, 1, 0; text = basin_name, space = :relative,
            align = (:right, :bottom), offset = (-5, 5), font = :italic,
        )
        push!(axs, ax)
    end
    linkyaxes!(axs...)
    # Only the first (left) and last (right) panels carry depth labels.
    for ax in axs[2:(end - 1)]
        hideydecorations!(ax; ticklabels = true, label = true, ticks = false, grid = false)
    end

    # Size each column proportional to its wet-cell latitude range.
    for (col, xl) in enumerate(xlims)
        colsize!(gc, col, Auto(xl[2] - xl[1]))
    end

    cbar = Colorbar(
        gc[2, 1:length(basin_configs)], cf_ref;
        vertical = false, flipaxis = false, label = colorbar_label,
    )
    cbar.width = Relative(0.5)

    Label(
        gc[0, 1:length(basin_configs)], "$title_prefix — basin zonal averages";
        fontsize = 16, tellwidth = false,
    )

    outputfile = joinpath(output_dir, "$(label)_zonal_avg_basins.png")
    @info "Saving $outputfile"
    save(outputfile, fig)
    flush(stdout); flush(stderr)
    return outputfile
end


################################################################################
# Age animation helpers (zonal averages + depth slices)
#
# Requires CairoMakie and OceanBasins symbols in the calling script's scope,
# as well as Oceananigans (Units, Grids, ImmersedBoundaries).
################################################################################

"""
    animate_zonal_averages(age_fts, grid, wet3D, vol_3D, output_dir, prefix;
                           colorrange, levels, colormap, n_frames=144, framerate=24)

Animate 4 zonal-average MP4s (global, Atlantic, Pacific, Indian) from a
`FieldTimeSeries`.  Each frame interpolates to the corresponding time, converts
to years, and computes volume-weighted zonal averages per basin.
"""
function animate_zonal_averages(
        age_fts, grid, wet3D, vol_3D, output_dir, prefix;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        n_frames = 144,
        framerate = 24,
        tracer_title::AbstractString = "age",
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, Nz′ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    basins = compute_ocean_basin_masks(grid, wet3D)
    global_mask = trues(Nx′, Ny′)
    basin_configs = [
        ("global", global_mask),
        ("atlantic", basins.ATL),
        ("pacific", basins.PAC),
        ("indian", basins.IND),
    ]

    stop_time = age_fts.times[end]
    frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

    age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    xw_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    w_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    za_buf = Array{Float64}(undef, Ny′, Nz′)

    # Build figure once; update observables per basin
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    first_mask = reshape(basin_configs[1][2], size(basin_configs[1][2], 1), size(basin_configs[1][2], 2), 1)
    zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, first_mask)
    za_obs = Observable(copy(za_buf))
    title_obs = Observable("")

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
    xlims!(ax, -90, 90)
    ylims!(ax, maximum(depth_vals), 0)
    Colorbar(fig[1, 2], cf; label = "Age (years)")

    for (basin_name, basin_mask) in basin_configs
        @info "Animating zonal average — $basin_name"
        flush(stdout); flush(stderr)

        mask3D = reshape(basin_mask, size(basin_mask, 1), size(basin_mask, 2), 1)

        # Reset to first frame for this basin
        age_raw = interior(age_fts[Time(frame_times[1])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
        za_obs.val .= za_buf
        notify(za_obs)
        title_obs[] = @sprintf("%s — %s zonal avg (t = 0.0 months)", tracer_title, basin_name)

        filepath = joinpath(output_dir, "$(prefix)_zonal_avg_$(basin_name).mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
            za_obs.val .= za_buf
            notify(za_obs)
            title_obs[] = @sprintf("%s — %s zonal avg (t = %.1f months)", tracer_title, basin_name, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
end

"""
    animate_depth_slices(age_fts, grid, wet3D, output_dir, prefix;
                         colorrange, levels, colormap, n_frames=144, framerate=24)

Animate 6 depth-slice MP4s (100, 200, 500, 1000, 2000, 3000 m) from a
`FieldTimeSeries`.  Each frame interpolates to the corresponding time and
extracts the nearest depth level.
"""
function animate_depth_slices(
        age_fts, grid, wet3D, output_dir, prefix;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        lowclip = colormap[1],
        highclip = colormap[end],
        n_frames = 144,
        framerate = 24,
        tracer_title::AbstractString = "age",
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    Nx′, Ny′, Nz′ = size(wet3D)
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z

    target_depths = [100, 200, 500, 1000, 2000, 3000]
    depth_k_indices = [(d, find_nearest_depth_index(grid, d)) for d in target_depths]

    stop_time = age_fts.times[end]
    frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

    age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)

    # Build the curvilinear gridmetrics once (shared by every frame/depth).
    gridmetrics = gridmetrics_from_grid(grid, Nx′, Ny′)

    # Build figure once; update observables per depth
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    slice_obs = Observable(age_buf[:, :, depth_k_indices[1][2]])
    title_obs = Observable("")

    fig = Figure(; size = (1000, 500))
    ax = map_axis(fig[1, 1]; title = title_obs)

    hm = plotmap!(ax, slice_obs, gridmetrics; colorrange, colormap, lowclip, highclip)
    add_coastlines!(ax)
    Colorbar(fig[1, 2], hm; label = "Age (years)")

    for (depth, k) in depth_k_indices
        @info "Animating depth slice — $(depth) m"
        flush(stdout); flush(stderr)

        actual_depth = round(depth_vals[k]; digits = 1)

        # Reset to first frame for this depth
        age_raw = interior(age_fts[Time(frame_times[1])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        slice_obs[] = @view(age_buf[:, :, k])
        title_obs[] = @sprintf("%s at %d m (k=%d, z=%.1f m, t = 0.0 months)", tracer_title, depth, k, actual_depth)

        filepath = joinpath(output_dir, "$(prefix)_slice_$(depth)m.mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            slice_obs[] = @view(age_buf[:, :, k])
            title_obs[] = @sprintf("%s at %d m (k=%d, z=%.1f m, t = %.1f months)", tracer_title, depth, k, actual_depth, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
end

################################################################################
# A vs B comparison plots (used by docs/IAF_NK_age_comparison_plan.md driver)
################################################################################

"""
    _safe_colorrange(values; default=(0.0, 1.0), low_q=0.02, high_q=0.98)

Return a non-degenerate `(cmin, cmax)` tuple suitable for Makie. Strips NaNs;
expands by `eps` if `cmin == cmax` (Makie would otherwise divide by zero and
produce all-NaN scaled colors); falls back to `default` if no valid values.
"""
function _safe_colorrange(values; default = (0.0, 1.0), low_q = 0.02, high_q = 0.98)
    valid = filter(isfinite, values)
    isempty(valid) && return default
    cmin = Float64(quantile(valid, low_q))
    cmax = Float64(quantile(valid, high_q))
    if !(isfinite(cmin) && isfinite(cmax)) || cmax - cmin < 1.0e-12
        # All values collapse to one point — expand minimally so Makie's
        # (value - cmin) / (cmax - cmin) normalization doesn't divide by zero.
        c = isfinite(cmin) ? cmin : 0.0
        return (c - 0.5, c + 0.5)
    end
    return (cmin, cmax)
end

"""
    _safe_diff_range(values; default=(-1.0, 1.0), scale=0.9)

Symmetric range `±scale·maxabs(values)`, NaN-stripped and non-degenerate.
"""
function _safe_diff_range(values; default = (-1.0, 1.0), scale = 0.9)
    valid = filter(isfinite, values)
    isempty(valid) && return default
    m = scale * maximum(abs, valid)
    isfinite(m) && m > 1.0e-12 || return default
    return (-m, m)
end

"""
    plot_age_comparison_slice(A_3D, B_3D, grid, wet3D, out_dir;
                              label_A, label_B, depth_m,
                              colorrange=:auto, diff_range=:auto,
                              colormap=:viridis,
                              value_label="Age (years)", diff_label="Δ Age (years)",
                              filename=nothing)

A | B | (B−A) horizontal slice at `depth_m`. A and B share a colourbar; the
diff uses `:balance` with a symmetric range. Dry cells (via `wet3D`) are
plotted as `nan_color`.

`colorrange=:auto` ⇒ 2nd–98th percentile of valid A/B values; pass a tuple
to override. `diff_range=:auto` ⇒ ±0.9·maxabs(B−A).
"""
function plot_age_comparison_slice(
        A_3D, B_3D, grid, wet3D, out_dir;
        label_A, label_B,
        depth_m,
        colorrange = :auto,
        diff_range = :auto,
        colormap = :viridis,
        value_label = "Age (years)",
        diff_label = "Δ Age (years)",
        filename = nothing,
    )
    mkpath(out_dir)

    k = find_nearest_depth_index(grid, depth_m)
    A_slice = Float64.(A_3D[:, :, k])
    B_slice = Float64.(B_3D[:, :, k])
    wet_k = wet3D[:, :, k]
    A_slice[.!wet_k] .= NaN
    B_slice[.!wet_k] .= NaN
    diff_slice = B_slice .- A_slice

    cr = colorrange === :auto ? _safe_colorrange(vcat(vec(A_slice), vec(B_slice))) : colorrange
    dr = diff_range === :auto ? _safe_diff_range(vec(diff_slice)) : diff_range

    Nx′, Ny′, _ = size(wet3D)
    gridmetrics = gridmetrics_from_grid(grid, Nx′, Ny′)

    fig = Figure(; size = (1700, 500))
    ax_A = map_axis(fig[1, 1]; title = label_A)
    ax_B = map_axis(fig[1, 2]; title = label_B)
    ax_D = map_axis(fig[1, 4]; title = "$label_B − $label_A")
    hm_A = plotmap!(ax_A, A_slice, gridmetrics; colorrange = cr, colormap)
    plotmap!(ax_B, B_slice, gridmetrics; colorrange = cr, colormap)
    hm_D = plotmap!(ax_D, diff_slice, gridmetrics; colorrange = dr, colormap = :balance)
    add_coastlines!(ax_A)
    add_coastlines!(ax_B)
    add_coastlines!(ax_D)
    Colorbar(fig[1, 3], hm_A; label = value_label)
    Colorbar(fig[1, 5], hm_D; label = diff_label)

    fname = filename === nothing ? "slice_$(Int(round(depth_m)))m.png" : filename
    outpath = joinpath(out_dir, fname)
    @info "Saving $outpath"
    save(outpath, fig)
    return outpath
end

"""
    plot_age_comparison_zonal(A_3D, B_3D, grid, wet3D, vol_3D,
                              basin_mask, basin_label, out_dir;
                              label_A, label_B,
                              colorrange=:auto, diff_range=:auto,
                              colormap=:viridis,
                              value_label="Age (years)", diff_label="Δ Age (years)",
                              filename=nothing)

A | B | (B−A) basin zonal-average (vol-weighted) plotted as heatmap of
(lat, depth). Auto ranges use the zonal-averaged data (not pre-zonal).
"""
function plot_age_comparison_zonal(
        A_3D, B_3D, grid, wet3D, vol_3D,
        basin_mask, basin_label, out_dir;
        label_A, label_B,
        colorrange = :auto,
        diff_range = :auto,
        colormap = :viridis,
        value_label = "Age (years)",
        diff_label = "Δ Age (years)",
        filename = nothing,
    )
    mkpath(out_dir)

    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, _ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    # Mask dry cells to NaN before averaging
    A_masked = Float64.(A_3D)
    B_masked = Float64.(B_3D)
    A_masked[.!wet3D] .= NaN
    B_masked[.!wet3D] .= NaN

    za_A = zonalaverage(A_masked, vol_3D, basin_mask)
    za_B = zonalaverage(B_masked, vol_3D, basin_mask)
    za_D = za_B .- za_A

    cr = colorrange === :auto ? _safe_colorrange(vcat(vec(za_A), vec(za_B))) : colorrange
    dr = diff_range === :auto ? _safe_diff_range(vec(za_D)) : diff_range

    fig = Figure(; size = (1700, 500))
    axkw = (
        xlabel = "Latitude", ylabel = "Depth (m)",
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
    )
    ax_A = Axis(fig[1, 1]; title = "$label_A — $basin_label", axkw...)
    ax_B = Axis(fig[1, 2]; title = "$label_B — $basin_label", axkw...)
    ax_D = Axis(fig[1, 4]; title = "$label_B − $label_A — $basin_label", axkw...)

    hm_A = heatmap!(ax_A, lat_repr, depth_vals, za_A; colorrange = cr, colormap, nan_color = :lightgray)
    heatmap!(ax_B, lat_repr, depth_vals, za_B; colorrange = cr, colormap, nan_color = :lightgray)
    hm_D = heatmap!(ax_D, lat_repr, depth_vals, za_D; colorrange = dr, colormap = :balance, nan_color = :lightgray)
    for ax in (ax_A, ax_B, ax_D)
        xlims!(ax, -90, 90)
        ylims!(ax, maximum(depth_vals), 0)
    end
    Colorbar(fig[1, 3], hm_A; label = value_label)
    Colorbar(fig[1, 5], hm_D; label = diff_label)

    fname = filename === nothing ? "zonal_$(basin_label).png" : filename
    outpath = joinpath(out_dir, fname)
    @info "Saving $outpath"
    save(outpath, fig)
    return outpath
end

"""
    compute_basin_profile(age_3D, vol_3D, basin_mask, grid; wet3D=nothing)
        -> (depths, profile_years)

Volume-weighted vertical profile of `age_3D` restricted to `basin_mask`.
`basin_mask` may be 2D (broadcast over depth) or 3D. Returns depths in metres
(positive downward) and the vol-weighted profile per depth level.
"""
function compute_basin_profile(age_3D, vol_3D, basin_mask, grid; wet3D = nothing)
    Nx′, Ny′, Nz′ = size(age_3D)
    m3 = ndims(basin_mask) == 2 ? reshape(basin_mask, Nx′, Ny′, 1) : basin_mask
    num = zeros(Float64, Nz′)
    den = zeros(Float64, Nz′)
    @inbounds for k in 1:Nz′, j in 1:Ny′, i in 1:Nx′
        mij = m3[i, j, ndims(basin_mask) == 2 ? 1 : k]
        mij || continue
        !isnothing(wet3D) && !wet3D[i, j, k] && continue
        a = age_3D[i, j, k]
        isnan(a) && continue
        v = vol_3D[i, j, k]
        num[k] += a * v
        den[k] += v
    end
    prof = @. ifelse(den > 0, num / den, NaN)
    z = znodes(grid, Center(), Center(), Center())
    return -z, prof
end

"""
    plot_age_profiles_basins(profiles, out_dir;
                             filename="profiles_basins.png",
                             value_label="Age (years)")

Overlay basin profiles from multiple pipelines on a single 1×4 figure
(Global | ATL | PAC | IND).

`profiles` is a Vector of NamedTuples with fields
`(; label, age_3D, grid, wet3D, vol_3D)` — one per pipeline. Each pipeline
may have its own grid/vol (e.g. Phase 2 overlays native OM2-1 with native
OM2-025), provided depths are comparable.
"""
function plot_age_profiles_basins(
        profiles, out_dir;
        filename = "profiles_basins.png",
        value_label = "Age (years)",
    )
    mkpath(out_dir)

    basin_names = ["Global", "ATL", "PAC", "IND"]
    fig = Figure(; size = (1600, 500))
    axes = [
        Axis(
                fig[1, i];
                title = basin_names[i],
                xlabel = value_label,
                ylabel = i == 1 ? "Depth (m)" : "",
            ) for i in 1:4
    ]

    for p in profiles
        bm = compute_ocean_basin_masks(p.grid, p.wet3D)
        Nx′, Ny′ = size(p.wet3D)[1:2]
        masks = [trues(Nx′, Ny′), bm.ATL, bm.PAC, bm.IND]
        for (i, m) in enumerate(masks)
            depths, prof = compute_basin_profile(p.age_3D, p.vol_3D, m, p.grid; wet3D = p.wet3D)
            lines!(axes[i], prof, depths; label = p.label, linewidth = 2)
        end
    end

    axislegend(axes[1]; position = :rb)
    for ax in axes
        ylims!(ax, 6000, 0)
    end

    outpath = joinpath(out_dir, filename)
    @info "Saving $outpath"
    save(outpath, fig)
    return outpath
end


################################################################################
# Seasonality helpers
################################################################################

"""
    seasonal_range(fts::FieldTimeSeries, wet3D, grid;
                   max_yr_range=10_000.0, label="") -> Array{Float64,3}

Per-cell `max − min` of `interior(fts[t])` across `t = 1:Nt`, in years
(divides by 365.25·86400). Streams snapshots so it works with `OnDisk()` FTS.
Dry cells are set to 0. Errors via the project-wide age-field sanity check if
any wet cell's range is non-finite or exceeds `max_yr_range` — that's an
upstream-solver instability symptom, not something to silently sanitize.
"""
function seasonal_range(fts, wet3D, grid; max_yr_range = 10_000.0, label = "")
    year_s = 365.25 * 86400
    Nt = length(fts.times)
    first_snap = Array(interior(fts[1]))
    age_min = Float64.(first_snap) ./ year_s
    age_max = copy(age_min)
    for t in 2:Nt
        snap = Array(interior(fts[t]))
        @inbounds for i in eachindex(age_min)
            v = Float64(snap[i]) / year_s
            v < age_min[i] && (age_min[i] = v)
            v > age_max[i] && (age_max[i] = v)
        end
    end
    res = age_max .- age_min
    @. res = ifelse(wet3D, res, 0.0)
    check_age_field(res, wet3D, grid; kind = "seasonal range", min_yr = 0.0, max_yr = max_yr_range, label)
    return res
end

"""
    check_age_field(field, wet3D, grid; kind, min_yr, max_yr, label)

Sanity check for a 3D age (or seasonal-range) field. Errors with a
diagnostic message if any wet cell is non-finite or falls outside
`[min_yr, max_yr]` years.

Ages above ~10,000 years or non-finite values are upstream-solver
pathologies (NK divergence, advection-driven blowup, …) and must be
resolved in the simulation pipeline — not papered over in the plotting
layer. The error reports the worst cell's value and its (lat, lon,
depth) so the user can localise the problem, plus the standard
remediation list (smaller Δt / lower TIMESTEP_MULT, more stable
timestepper, advection-scheme change, or upstream bug).
"""
function check_age_field(field, wet3D, grid; kind, min_yr, max_yr, label)
    bad = wet3D .& (.!isfinite.(field) .| (field .< min_yr) .| (field .> max_yr))
    n_bad = count(bad)
    n_bad == 0 && return field

    abs_bad = ifelse.(bad, abs.(replace(field, NaN => 0.0, Inf => 1.0e30, -Inf => 1.0e30)), 0.0)
    worst_idx = argmax(abs_bad)
    i, j, k = Tuple(worst_idx)
    worst_val = field[worst_idx]
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon = Array(ug.λᶜᶜᵃ)[i, j]
    lat = Array(ug.φᶜᶜᵃ)[i, j]
    z = znodes(grid, Center(), Center(), Center())
    depth = -z[k]
    n_wet = count(wet3D)

    error(
        """
        $kind for pipeline `$label` is unphysical: $n_bad / $n_wet wet cells
        ($(round(n_bad / max(n_wet, 1) * 100; digits = 3))%) have non-finite values or fall
        outside [$min_yr, $max_yr] yr.

        Worst cell:  value = $worst_val yr  at (i, j, k) = ($i, $j, $k),
                     lat = $(round(lat; digits = 2))°,  lon = $(round(lon; digits = 2))°,
                     depth ≈ $(round(depth; digits = 1)) m.

        Ages here should saturate at a few thousand years; values > 10,000 yr
        or non-finite mean the upstream 1-year NK simulation diverged.
        Remediations:
          * Re-run the periodic NK solve with a smaller timestep
            (lower TIMESTEP_MULT or smaller Δt).
          * Try a more stable timestepper (SRK3 → SRK5) or advection scheme
            (centered2 → WENO).
          * Inspect the per-iterate trace for a NaN/Inf onset.

        Fix the upstream pipeline and re-run; do not bypass this check by
        clipping values in the plotting layer.
        """,
    )
end


"""
    colormap_for_field(name)

Return a colormap symbol appropriate for the given field name.
"""
function colormap_for_field(name)
    if name in ("T", "temp", "temperature")
        return :coolwarm
    elseif name in ("S", "salt", "salinity")
        return :haline
    elseif name in ("MLD", "mld")
        return :magma
    elseif contains(name, "kappaV") || contains(name, "κ")
        return :cividis
    elseif name in ("eta", "η")
        return :PRGn
    elseif name in ("u", "v", "w")
        return :balance
    else
        return :viridis
    end
end

"""
    mk_piecewise_linear(vs) -> ReversibleScale

Create a piecewise-linear scale that maps non-uniform levels `vs` to
uniform integer indices 0, 1, ..., n-1. Used for log-like colorbar spacing.
From https://discourse.julialang.org/t/makie-nonlinear-color-levels-in-colorbar/118056/5
"""
function mk_piecewise_linear(vs)
    @assert length(vs) > 1
    function is_increasing(ss)
        prev = ss[1]
        for s in ss[2:end]
            (s ≤ prev) && return false
            prev = s
        end
        return true
    end
    @assert is_increasing(vs)
    d1 = vs[2] - vs[1]
    d2 = vs[end] - vs[end - 1]
    un = size(vs, 1) - 1
    function piecewise_linear(v)
        return if isnan(v)
            NaN
        elseif v <= vs[1]
            (v - vs[1]) / d1
        elseif v ≥ vs[end]
            (v - vs[end]) / d2 + un
        else
            i = findfirst(q -> v < q, vs) - 1
            d = vs[i + 1] - vs[i]
            (v - vs[i]) / d + i - 1
        end
    end
    function its_inverse(u)
        return if u ≤ 0
            u * d1 + vs[1]
        elseif u ≥ un
            (u - un) * d2 + vs[end]
        else
            iu = floor(Int, u)
            i = iu + 1
            d = vs[i + 1] - vs[i]
            (u - iu) * d + vs[i]
        end
    end
    return ReversibleScale(piecewise_linear, its_inverse)
end

"""
    animate_surface_fields(field_specs, output_dir, prefix;
                           n_frames=144, framerate=24, show_halos=true)

Animate surface maps (top z-level for 3D fields, or full 2D fields) from
FieldTimeSeries inputs. Includes halo regions when `show_halos=true`.

`field_specs` is a vector of `(name, fts, k_index)` tuples where:
- `name::String` — label for the field (e.g. "u", "T", "MLD")
- `fts::FieldTimeSeries` — the monthly FTS (Cyclical, InMemory)
- `k_index` — z-index for the surface slice: `nothing` for 2D fields (η, MLD),
  or an integer for 3D fields (typically `Nz` for the top level)
"""
function animate_surface_fields(
        field_specs, output_dir, prefix;
        n_frames = 144,
        framerate = 24,
        show_halos = true,
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    for (name, fts, k_index) in field_specs
        @info "Animating surface field — $name"
        flush(stdout); flush(stderr)

        stop_time = fts.times[end]
        frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

        # Extract first frame to set up figure
        field_snap = fts[Time(frame_times[1])]
        if show_halos
            raw = Array(parent(field_snap))
            if !isnothing(k_index)
                slice = raw[:, :, k_index]
            else
                slice = dropdims(raw; dims = 3)
            end
        else
            raw = Array(interior(field_snap))
            if !isnothing(k_index)
                slice = raw[:, :, k_index]
            else
                slice = dropdims(raw; dims = 3)
            end
        end

        # Replace exact zeros in immersed cells with NaN for cleaner plots
        slice_f64 = Float64.(slice)
        replace!(slice_f64, 0.0 => NaN)

        slice_obs = Observable(copy(slice_f64))
        title_obs = Observable(@sprintf("%s surface (t = 0.0 months)", name))

        fig = Figure(; size = (1000, 500))
        ax = Axis(
            fig[1, 1]; title = title_obs,
            xlabel = show_halos ? "i (with halos)" : "i",
            ylabel = show_halos ? "j (with halos)" : "j"
        )

        if name in ("MLD", "mld")
            # Log-like piecewise-linear colorscale for MLD
            mld_levels = [0, 50, 100, 200, 500, 1000, 2000]
            colorscale = mk_piecewise_linear(mld_levels)
            colorrange = extrema(mld_levels)
            colormap = cgrad(:thermal, length(mld_levels); categorical = true)
            highclip = colormap[end]
            colormap = cgrad(colormap[1:(end - 1)]; categorical = true)
            hm = heatmap!(
                ax, slice_obs;
                colorrange, colormap, colorscale, highclip, nan_color = :black
            )
            Colorbar(
                fig[1, 2];
                limits = (1, length(mld_levels)),
                colormap, highclip,
                ticks = (1:length(mld_levels), string.(mld_levels)),
                label = "MLD (m)"
            )
        elseif name in ("MLK", "mlk")
            # MLD plotted with grid z-face levels as piecewise-linear colorscale
            zf = znodes(fts.grid, Face(); with_halos = false)
            mlk_levels = sort(abs.(collect(zf)))
            # Remove duplicate zeros if present
            unique!(mlk_levels)
            colorscale = mk_piecewise_linear(mlk_levels)
            colorrange = extrema(mlk_levels)
            n_levels = length(mlk_levels)
            colormap = cgrad(:rainbow_bgyrm_35_85_c71_n256, n_levels; categorical = true)
            highclip = colormap[end]
            colormap = cgrad(colormap[1:(end - 1)]; categorical = true)
            hm = heatmap!(
                ax, slice_obs;
                colorrange, colormap, colorscale, highclip, nan_color = :black
            )
            # Subsample ticks to avoid crowding (~10 ticks)
            tick_stride = max(1, n_levels ÷ 10)
            tick_indices = 1:tick_stride:n_levels
            Colorbar(
                fig[1, 2];
                limits = (1, n_levels),
                colormap, highclip,
                ticks = (collect(tick_indices), string.(Int.(round.(mlk_levels[tick_indices])))),
                label = "MLD on z-faces (m)"
            )
        else
            # Auto-detect color range from the first frame (excluding NaN)
            valid = filter(!isnan, vec(slice_f64))
            if isempty(valid)
                cmin, cmax = -1.0, 1.0
            else
                cmin = Float64(quantile(valid, 0.02))
                cmax = Float64(quantile(valid, 0.98))
            end
            if cmin ≈ cmax
                cmin -= 1.0
                cmax += 1.0
            end
            colormap = colormap_for_field(name)
            hm = heatmap!(ax, slice_obs; colorrange = (cmin, cmax), colormap, nan_color = :black)
            Colorbar(fig[1, 2], hm; label = name)
        end

        filepath = joinpath(output_dir, "$(prefix)_surface_$(name).mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            field_snap = fts[Time(frame_times[i])]
            if show_halos
                raw = Array(parent(field_snap))
                if !isnothing(k_index)
                    sl = raw[:, :, k_index]
                else
                    sl = dropdims(raw; dims = 3)
                end
            else
                raw = Array(interior(field_snap))
                if !isnothing(k_index)
                    sl = raw[:, :, k_index]
                else
                    sl = dropdims(raw; dims = 3)
                end
            end
            sl_f64 = Float64.(sl)
            replace!(sl_f64, 0.0 => NaN)
            slice_obs[] = sl_f64
            title_obs[] = @sprintf("%s surface (t = %.1f months)", name, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
end
