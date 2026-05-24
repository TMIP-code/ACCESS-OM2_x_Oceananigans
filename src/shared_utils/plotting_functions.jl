"""
Shared plotting helpers for curvilinear (tripolar) maps.

Ported with minimal edits from
[ACCESS-TMIP/src/plotting_functions.jl](/home/561/bp3051/Projects/TMIP/ACCESS-TMIP/src/plotting_functions.jl)
— same MOM5 tripolar grid family that ACCESS-OM2-1 uses, so the helpers
apply directly. The `GeoMakie.land()` / `LibGEOS` / `GeometryOps` land-poly
section of the upstream file has been dropped; callers add coastlines via
`lines!(ax, GeoMakie.coastlines(), color = :black, linewidth = 0.85)` if
they want them.

Public API:
  plotmap!(ax, x2D, gridmetrics; …)
  lonticklabel, latticklabel, xtickformat, ytickformat, loninsamewindow
  divergingcbarticklabel, divergingcbarticklabelformat
  mk_piecewise_linear, myhidexdecorations!, myhideydecorations!
  withwhitelow, withwhitecenter, dataforbicolorband

`gridmetrics` is a NamedTuple with fields `lon, lat, lon_vertices,
lat_vertices`. Centres are 2-D arrays of size (Nx, Ny); vertices are
4-D arrays of size (4, Nx, Ny) holding the 4 corners of each cell.

Loading this file assumes `CairoMakie` and `GeometryBasics` are already in
scope; callers `using CairoMakie, GeometryBasics` before `include`ing.
"""

################################################################################
# Tick label / axis formatters
################################################################################

function lonticklabel(lon)
    lon = mod(lon + 180, 360) - 180
    lon = isinteger(lon) ? Int(lon) : lon
    return if lon == 0
        "0°"
    elseif (lon ≈ 180) || (lon ≈ -180)
        "180°"
    elseif lon > 0
        "$(string(lon))°E"
    else
        "$(string(-lon))°W"
    end
end
xtickformat(x) = lonticklabel.(x)

function latticklabel(lat)
    lat = isinteger(lat) ? Int(lat) : lat
    return if lat == 0
        "0°"
    elseif lat > 0
        "$(string(lat))°N"
    else
        "$(string(-lat))°S"
    end
end
ytickformat(y) = latticklabel.(y)

function divergingcbarticklabel(x)
    isinteger(x) && (x = Int(x))
    return if x == 0
        "0"
    elseif x > 0
        "+" * string(x)
    else
        "−" * string(-x)
    end
end
divergingcbarticklabelformat(x) = divergingcbarticklabel.(x)

loninsamewindow(l1, l2) = mod(l1 - l2 + 180, 360) + l2 - 180

################################################################################
# Piecewise-linear scale (ReversibleScale)
################################################################################

"""
    mk_piecewise_linear(vs) → ReversibleScale

To be used as `colorscale` of Makie's `contourf` / `heatmap` / `mesh!`.
Piecewise-linear mapping such that `forward.(vs) == [0, 1, …, n-1]`. Outside
`[v1, vn]`, the mapping linearly extrapolates. `vs` must be strictly
increasing.

Source: https://discourse.julialang.org/t/makie-nonlinear-color-levels-in-colorbar/118056/5
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
        return if v ≤ vs[1]
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

################################################################################
# Colour-scheme tweaks
################################################################################

function withwhitelow(cs)
    cs = deepcopy(cs)
    cs.colors[1] = Makie.Colors.colorant"white"
    return cs
end

function withwhitecenter(cs)
    cs = deepcopy(cs)
    cs.colors[end ÷ 2 + 1] = Makie.Colors.colorant"white"
    return cs
end

################################################################################
# Bicolour-band helper for zonal-integral diff panel
################################################################################

"""
    dataforbicolorband(x, y1, y2)

Re-sample two y-series (`y1`, `y2`) sharing a common x-axis so that
`band!`-style fills can be split by sign at the crossing points. Returns
`(x_sorted, y1_sorted, y2_sorted)` augmented with the (x, y) coordinates
of each `y1 ↔ y2` self-intersection.
"""
function dataforbicolorband(x, y1, y2)
    idx, points = self_intersections(Point2f.([x; reverse(x)], [y1; reverse(y2)]))
    allx = [x; [p.data[1] for p in points]]
    ally1 = [y1; [p.data[2] for p in points]]
    ally2 = [y2; [p.data[2] for p in points]]
    isort = sortperm(allx)
    return allx[isort], ally1[isort], ally2[isort]
end

################################################################################
# Decoration hiders
################################################################################

function myhidexdecorations!(ax, condition)
    return hidexdecorations!(
        ax;
        label = condition, ticklabels = condition,
        ticks = condition, grid = false,
    )
end

function myhideydecorations!(ax, condition)
    return hideydecorations!(
        ax;
        label = condition, ticklabels = condition,
        ticks = condition, grid = false,
    )
end

################################################################################
# plotmap!: per-cell quad mesh on a curvilinear grid
################################################################################

"""
    plotmap!(ax, x2D, gridmetrics; colorrange, colormap,
             levels = nothing, highclip = automatic, lowclip = automatic,
             colorscale = identity, lon_window_start = 20)

Draw a per-cell quad mesh of `x2D` (size `(Nx, Ny)`) onto `ax` using cell
vertices from `gridmetrics.lon_vertices, gridmetrics.lat_vertices` (each of
shape `(4, Nx, Ny)`). Cells are shifted so that the longitude window starts
at `lon_window_start` (default 20°E), giving a Pacific-centred view with
the Atlantic cut over Africa — matches the upstream ACCESS-TMIP / Pasquier
2024 panels and avoids dateline-wrap artifacts.

NaN values are rendered via the axis `backgroundcolor` — set
`backgroundcolor = :lightgray` on the Axis to get an "ocean over land" feel.
"""
function plotmap!(
        ax, x2D, gridmetrics;
        colorrange, colormap,
        levels = nothing,
        highclip = automatic, lowclip = automatic,
        colorscale = identity,
        lon_window_start = 20,
    )

    lonv = gridmetrics.lon_vertices
    latv = gridmetrics.lat_vertices
    lon = gridmetrics.lon

    # Wrap so that the cell centres sit in [lon_window_start, lon_window_start + 360),
    # then pull each vertex into the same window relative to its centre.
    lon_centred = mod.(lon .- lon_window_start, 360) .+ lon_window_start
    lonv_centred = loninsamewindow.(lonv, reshape(lon_centred, (1, size(lon_centred)...)))

    # Build quads. (Note: vec(x2D) iterates linearly in i,j, matching the
    # i,j enumeration over lonv[:, i, j] below.)
    quad_points = vcat(
        [
            Point2{Float64}.(lonv_centred[:, i, j], latv[:, i, j])
                for i in axes(lonv_centred, 2), j in axes(lonv_centred, 3)
        ]...,
    )
    quad_faces = vcat(
        [
            begin
                    j = (i - 1) * 4 + 1
                    [j j + 1 j + 2; j + 2 j + 3 j]
                end for i in 1:(length(quad_points) ÷ 4)
        ]...,
    )
    colors_per_point = vcat(fill.(vec(x2D), 4)...)

    plt = mesh!(
        ax, quad_points, quad_faces;
        color = colors_per_point,
        shading = NoShading,
        colormap, colorrange,
        rasterize = 2,
        highclip, lowclip, colorscale,
    )
    xlims!(ax, (lon_window_start, lon_window_start + 360))
    ylims!(ax, (-90, 90))

    # Push the mesh below other plot elements (coastlines, contours, etc.)
    translate!(plt, 0, 0, -100)

    return plt
end
