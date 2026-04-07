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

    fi = interior(f, ii, jj, kk)
    fi_cpu = on_architecture(CPU(), fi)

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
- `age_3D`:     (Nx', Ny', Nz') array (years) with 0 for dry cells
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

    # Replace dry cells with NaN for plotting
    age_plot = copy(age_3D)
    age_plot[.!wet3D] .= NaN

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

    # ── Horizontal slices ───────────────────────────────────────────────

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
        ax = Axis(
            fig[1, 1];
            title = "$title_prefix at $title_tag",
        )

        hm = heatmap!(ax, slice; colorrange, colormap, nan_color = :black, lowclip, highclip)
        Colorbar(fig[1, 2], hm; label = colorbar_label)

        outputfile = joinpath(output_dir, "$(label)_slice_$(file_tag).png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    @info "Age diagnostic plots saved to $output_dir"
    flush(stdout); flush(stderr)
    return nothing
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
        title_obs[] = @sprintf("age — %s zonal avg (t = 0.0 months)", basin_name)

        filepath = joinpath(output_dir, "$(prefix)_zonal_avg_$(basin_name).mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
            za_obs.val .= za_buf
            notify(za_obs)
            title_obs[] = @sprintf("age — %s zonal avg (t = %.1f months)", basin_name, frame_times[i] / (year / 12))
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

    # Build figure once; update observables per depth
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    slice_obs = Observable(age_buf[:, :, depth_k_indices[1][2]])
    title_obs = Observable("")

    fig = Figure(; size = (1000, 500))
    ax = Axis(fig[1, 1]; title = title_obs)

    hm = heatmap!(
        ax, slice_obs; colorrange, colormap, nan_color = :black,
        lowclip, highclip
    )
    Colorbar(fig[1, 2], hm; label = "Age (years)")

    for (depth, k) in depth_k_indices
        @info "Animating depth slice — $(depth) m"
        flush(stdout); flush(stderr)

        actual_depth = round(depth_vals[k]; digits = 1)

        # Reset to first frame for this depth
        age_raw = interior(age_fts[Time(frame_times[1])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        slice_obs[] = @view(age_buf[:, :, k])
        title_obs[] = @sprintf("age at %d m (k=%d, z=%.1f m, t = 0.0 months)", depth, k, actual_depth)

        filepath = joinpath(output_dir, "$(prefix)_slice_$(depth)m.mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            slice_obs[] = @view(age_buf[:, :, k])
            title_obs[] = @sprintf("age at %d m (k=%d, z=%.1f m, t = %.1f months)", depth, k, actual_depth, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
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
        return if v <= vs[1]
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
            if k_index !== nothing
                slice = raw[:, :, k_index]
            else
                slice = dropdims(raw; dims = 3)
            end
        else
            raw = Array(interior(field_snap))
            if k_index !== nothing
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
                if k_index !== nothing
                    sl = raw[:, :, k_index]
                else
                    sl = dropdims(raw; dims = 3)
                end
            else
                raw = Array(interior(field_snap))
                if k_index !== nothing
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
