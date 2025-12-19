"""
Extend the supergrid data with one row.
"""
function extend_supergrid(; x, y, dx, dy, area, nx, nxp, ny, nyp)
    ny += 2
    nyp += 2
    x = hcat(x, fill(NaN, nxp, 2))
    y = hcat(y, fill(NaN, nxp, 2))
    dy = hcat(dy, fill(NaN, nxp, 2))
    dx = hcat(dx, fill(NaN, nx, 2))
    area = hcat(area, fill(NaN, nx, 2))
    for i in 1:nxp, j in 1:2
        x[i, nyp - 2 + j] = x[nxp - i + 1, nyp - 2 - j]
        y[i, nyp - 2 + j] = y[nxp - i + 1, nyp - 2 - j]
        dy[i, ny - 2 + j] = dy[nxp - i + 1, ny - 1 - j]
    end
    for i in 1:nx, j in 1:2
        dx[i, nyp - 2 + j] = dx[nx - i + 1, nyp - 2 - j]
        area[i, ny - 2 + j] = area[nx - i + 1, ny - 1 - j]
    end
    return (; x, y, dx, dy, area, nx, nxp, ny, nyp)
end


"""
Generate dummy supergrid data for testing.
"""
function dummy_supergrid(; nx = 8, ny = 8)
    nxp = nx + 1
    nyp = ny + 1
    x = rand(nxp, nyp)
    y = rand(nxp, nyp)
    dx = rand(nx, nyp)
    dy = rand(nxp, ny)
    area = rand(nx, ny)
    return (; x, y, dx, dy, area, nx, nxp, ny, nyp)
end

"""
Plot the extended supergrid with markers at key points.

```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using CairoMakie
include("src/tripolargrid_reader.jl")
supergrid = dummy_supergrid()
extended_supergrid = extend_supergrid(; supergrid...)
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1]; aspect = DataAspect())
plot_extended_supergrid!(ax, extended_supergrid...)
@show figname = "output/extended_supergrid_plot.png"
save(figname, fig)
```
"""
function plot_extended_supergrid!(ax, x, y, dx, dy, area, nx, nxp, ny, nyp)
    ax.xticks = 1:2:nxp
    ax.yticks = 1:2:nyp
    ax.xminorticks = 1:nxp
    ax.yminorticks = 1:nyp
    ax.xminorgridvisible = true
    ax.yminorgridvisible = true
    ax.limits = ((0.7, nxp + 0.3), (0.7, nyp + 0.3))
    # North pole fold / seam
    lines!(ax, [1, nxp], [nyp - 2, nyp - 2]; color = :red, linewidth = 2)
    scatter!(ax, 1, nyp - 2; marker = :circle, color = :red)
    text!(ax, 1, nyp - 2; text = "P1", color = :red, offset = (+3, +3), align = (:left, :bottom))
    scatter!(ax, nx ÷ 2 + 1, nyp - 2; marker = :circle, color = :red)
    text!(ax, nx ÷ 2 + 1, nyp - 2; text = "P2", color = :red, offset = (+3, +3), align = (:left, :bottom))
    scatter!(ax, nxp, nyp - 2; marker = :circle, color = :red)
    text!(ax, nxp, nyp - 2; text = "P1′", color = :red, offset = (-3, +3), align = (:right, :bottom))
    # x y markers
    xymarkers = [
        :circle :utriangle
        :star5 :rect
    ]
    for i in 1:2, j in 1:2
        imarkers = [i, nxp - i + 1]
        jmarkers = [nyp - 2 + j, nyp - 2 - j]
        color = [x[imarkers[1], jmarkers[1]], x[imarkers[2], jmarkers[2]]]
        options = (; colormap = :jet, color, colorrange = (0, 1), strokecolor = :black, strokewidth = 2, markersize = 20)
        scatter!(ax, imarkers, jmarkers; marker = xymarkers[i, j], options...)
    end
    return
end
