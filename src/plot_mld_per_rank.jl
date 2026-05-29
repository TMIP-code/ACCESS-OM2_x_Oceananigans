"""
Plot per-rank partitioned MLD against the global serial MLD, with halos, for
every monthly snapshot. Outputs 12 PNGs (one per month) and an mp4 cycling
through them.

Usage:
    PARENT_MODEL=ACCESS-OM2-01 PARTITION=1x4 julia --project src/plot_mld_per_rank.jl
"""

using JLD2
using CairoMakie
using Printf

include("shared_functions.jl")

(; parentmodel, experiment_dir, monthly_dir, outputdir) = load_project_config()

PARTITION = get(ENV, "PARTITION", "1x4")
px, py = parse.(Int, split(PARTITION, "x"))
nranks = px * py

partition_dir = joinpath(dirname(monthly_dir), "partitions", PARTITION)
isdir(partition_dir) || error("Partition dir not found: $partition_dir")

plot_dir = joinpath(outputdir, "diagnostics", "mld_per_rank_$(PARTITION)")
mkpath(plot_dir)

@info "Config: parentmodel=$parentmodel PARTITION=$PARTITION ($nranks ranks)"
@info "monthly_dir   = $monthly_dir"
@info "partition_dir = $partition_dir"
@info "plot_dir      = $plot_dir"

################################################################################
# Load per-rank MLD (12 snapshots, 2D each with halos)
################################################################################

n_months = 12

rank_data = Vector{Vector{Matrix{Float64}}}(undef, nranks)  # rank_data[r][m] -> 2-D array
rank_meta = Vector{NamedTuple}(undef, nranks)
for r in 0:(nranks - 1)
    path = joinpath(partition_dir, "mld_monthly_rank$(r).jld2")
    isfile(path) || error("Rank file not found: $path")
    snaps, meta = jldopen(path, "r") do f
        snaps_local = [Array(f["data/$i"])[:, :, 1] for i in 1:n_months]
        meta_local = (
            local_Nx = f["local_Nx"], local_Ny = f["local_Ny"],
            Hx = f["Hx"], Hy = f["Hy"],
        )
        (snaps_local, meta_local)
    end
    rank_data[r + 1] = snaps
    rank_meta[r + 1] = meta
    @info "rank $r: size=$(size(snaps[1])) meta=$meta"
end

################################################################################
# Determine a common color scale (99th percentile of a subsample of values to
# avoid sorting the full ~470M-element vector across all ranks × months)
################################################################################

function subsample_quantile(rank_data, q; stride = 50)
    samples = Float64[]
    sizehint!(samples, 10^6)
    for snaps in rank_data, snap in snaps
        for v in @view snap[1:stride:end]
            isfinite(v) && v > 0 && push!(samples, v)
        end
    end
    isempty(samples) && return (0.0, 1.0)
    sort!(samples)
    return (samples[1], samples[clamp(round(Int, q * length(samples)), 1, length(samples))])
end

vmin, vmax = subsample_quantile(rank_data, 0.99)
@info "Color range: [$(@sprintf("%.2g", vmin)), $(@sprintf("%.2g", vmax))]"

################################################################################
# Plot one figure per month: nranks vertically stacked rank panels (rank 0 at
# the bottom, matching the physical south→north layout). x decorations are
# hidden on all but the bottom panel since they share i.
################################################################################

# Rank r → figure row (rank 0 at the bottom = nranks)
panel_row(r) = nranks - r

function plot_month(month_idx, save_path)
    fig = Figure(; size = (1800, 900), backgroundcolor = :white)
    Label(
        fig[0, 1:2], "MLD (m) — month $month_idx (1968–1977 climatology), partition=$PARTITION";
        fontsize = 18, font = :bold
    )
    hms = []
    for r in 0:(nranks - 1)
        data = rank_data[r + 1][month_idx]
        nx, ny = size(data)
        row = panel_row(r)
        Label(
            fig[row, 0], "rank $r — $(nx)×$(ny)";
            rotation = π / 2, tellheight = false, fontsize = 13, font = :bold
        )
        ax = Axis(
            fig[row, 1];
            xlabel = "i (incl. halos)", ylabel = "j (incl. halos)",
        )
        hm = heatmap!(
            ax, 1:nx, 1:ny, data;
            colorrange = (vmin, vmax), colormap = :viridis, nan_color = :lightgray,
        )
        push!(hms, hm)
        Hx = rank_meta[r + 1].Hx
        Hy = rank_meta[r + 1].Hy
        lines!(
            ax, [Hx + 0.5, nx - Hx + 0.5, nx - Hx + 0.5, Hx + 0.5, Hx + 0.5],
            [Hy + 0.5, Hy + 0.5, ny - Hy + 0.5, ny - Hy + 0.5, Hy + 0.5];
            color = :red, linewidth = 1, linestyle = :dash
        )
        # Bottom panel (rank 0) is the only one that keeps x decorations
        r != 0 && hidexdecorations!(ax; grid = false)
    end
    Colorbar(fig[1:nranks, 2], hms[1]; label = "MLD (m)")
    save(save_path, fig)
    return fig
end

png_paths = String[]
for m in 1:n_months
    path = joinpath(plot_dir, @sprintf("mld_month%02d.png", m))
    plot_month(m, path)
    push!(png_paths, path)
    @info "Saved $(basename(path))"
end

################################################################################
# Animation: record CairoMakie scene cycling through months
################################################################################

mp4_path = joinpath(plot_dir, "mld_per_rank.mp4")
@info "Recording mp4: $mp4_path"

obs_month = Observable(1)
rank_obs = [@lift(rank_data[r + 1][$obs_month]) for r in 0:(nranks - 1)]

fig_anim = Figure(; size = (1800, 900), backgroundcolor = :white)
title_obs = @lift("MLD (m) — month $($obs_month) (1968–1977 climatology), partition=$PARTITION")
Label(fig_anim[0, 1:2], title_obs; fontsize = 18, font = :bold)
hms_anim = []
for r in 0:(nranks - 1)
    data0 = rank_data[r + 1][1]
    nx, ny = size(data0)
    row = panel_row(r)
    Label(
        fig_anim[row, 0], "rank $r — $(nx)×$(ny)";
        rotation = π / 2, tellheight = false, fontsize = 13, font = :bold
    )
    ax = Axis(
        fig_anim[row, 1];
        xlabel = "i", ylabel = "j",
    )
    hm = heatmap!(
        ax, 1:nx, 1:ny, rank_obs[r + 1];
        colorrange = (vmin, vmax), colormap = :viridis, nan_color = :lightgray
    )
    push!(hms_anim, hm)
    Hx = rank_meta[r + 1].Hx
    Hy = rank_meta[r + 1].Hy
    lines!(
        ax, [Hx + 0.5, nx - Hx + 0.5, nx - Hx + 0.5, Hx + 0.5, Hx + 0.5],
        [Hy + 0.5, Hy + 0.5, ny - Hy + 0.5, ny - Hy + 0.5, Hy + 0.5];
        color = :red, linewidth = 1, linestyle = :dash
    )
    r != 0 && hidexdecorations!(ax; grid = false)
end
Colorbar(fig_anim[1:nranks, 2], hms_anim[1]; label = "MLD (m)")

record(fig_anim, mp4_path, 1:n_months; framerate = 2) do m
    obs_month[] = m
end
@info "Saved $mp4_path"
@info "Done. PNGs + mp4 in $plot_dir"
