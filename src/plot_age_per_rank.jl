"""
Per-rank diagnostic of |age| at t=1yr for instability investigation.

Loads `age_1year_rank{0..N-1}.jld2` from the standardrun directory matching
MODEL_CONFIG and PARTITION env vars (so e.g. the failed MONTHLY_KAPPAV=yes
run lands here). For each rank:

- Prints (i, j, k) of the global argmax(|age|) and the max value.
- Plots a column-max projection of `log10(|age| + 1)` per rank
  (collapses depth → shows where the worst cells live in (i, j)).
- Plots a horizontal slice of `log10(|age| + 1)` at the global-worst k.

Layout matches `plot_mld_per_rank.jl`: 4×1 stacked rows with rank 0 at the
bottom, rotated rank labels on the left, x decorations hidden on all but
the bottom panel.

Usage:
    MODEL_CONFIG=... PARTITION=1x4 PARENT_MODEL=ACCESS-OM2-01 \\
        julia --project src/plot_age_per_rank.jl
"""

using JLD2
using CairoMakie
using Printf

include("shared_functions.jl")

(; parentmodel, experiment_dir, outputdir) = load_project_config()

PARTITION = get(ENV, "PARTITION", "1x4")
px, py = parse.(Int, split(PARTITION, "x"))
nranks = px * py

MODEL_CONFIG = get(ENV, "MODEL_CONFIG", "")
isempty(MODEL_CONFIG) && error("MODEL_CONFIG must be set (e.g. cgridtransports_wdiagnosed_centered2_AB2_mkappaV_LBS_DTx2)")

data_dir = joinpath(outputdir, "standardrun", MODEL_CONFIG, PARTITION)
isdir(data_dir) || error("Data dir not found: $data_dir")

plot_dir = joinpath(outputdir, "diagnostics", "age_per_rank_$(MODEL_CONFIG)_$(PARTITION)")
mkpath(plot_dir)

@info "Config: parentmodel=$parentmodel PARTITION=$PARTITION MODEL_CONFIG=$MODEL_CONFIG"
@info "data_dir = $data_dir"
@info "plot_dir = $plot_dir"

################################################################################
# Load final-snapshot age field per rank
################################################################################

# Rank r → figure row (rank 0 at the bottom = nranks)
panel_row(r) = nranks - r

# Per-rank final-snapshot age. Each entry is a 3-D array (nx, ny, nz) with halos.
rank_age = Vector{Array{Float32, 3}}(undef, nranks)
rank_argmax = Vector{NamedTuple}(undef, nranks)
for r in 0:(nranks - 1)
    path = joinpath(data_dir, "age_1year_rank$(r).jld2")
    isfile(path) || error("Rank file not found: $path")
    age = jldopen(path, "r") do f
        iters = sort(parse.(Int, filter(k -> k != "serialized", keys(f["timeseries/age"]))))
        final_iter = iters[end]
        t_yr = f["timeseries/t/$(final_iter)"] / 31_557_600
        @info "rank $r: final iter=$final_iter (t=$(round(t_yr, sigdigits = 5)) yr)"
        Array(f["timeseries/age/$(final_iter)"])
    end
    rank_age[r + 1] = age

    abs_age = abs.(age)
    max_val, max_lin = findmax(abs_age)
    i, j, k = Tuple(CartesianIndices(abs_age)[max_lin])
    rank_argmax[r + 1] = (; i, j, k, max_val = Float64(max_val))
    @info @sprintf(
        "  rank %d size=%s  argmax(|age|) = (i=%d, j=%d, k=%d)  max=%.3e yr",
        r, size(age), i, j, k, max_val
    )
end

# Worst rank / global k for horizontal slice
worst_r = argmax([rm.max_val for rm in rank_argmax]) - 1
k_worst = rank_argmax[worst_r + 1].k
@info "Worst rank = $worst_r (max=$(rank_argmax[worst_r + 1].max_val) yr), using k=$k_worst for slice plot"

################################################################################
# Plot helpers
################################################################################

# log10(|x| + 1) so zero/clean cells map to 0 and the saturated cells span the
# full dynamic range.
logabs(x) = log10(abs(Float64(x)) + 1)

function plot_stack(get_panel_data::Function, fig_title, save_path; cbar_label)
    fig = Figure(; size = (1800, 900), backgroundcolor = :white)
    Label(fig[0, 1:2], fig_title; fontsize = 18, font = :bold)
    hms = []
    # Common color range across all 4 ranks for direct comparison. Compute
    # vmax from finite values only (rank 1 may have Inf cells that would
    # otherwise saturate everything), then clip Inf/NaN cells to vmax so
    # they render at the top of the colorbar instead of crashing CairoMakie.
    panel_data_raw = [get_panel_data(r) for r in 0:(nranks - 1)]
    vmax = 0.0
    for d in panel_data_raw, v in d
        isfinite(v) && v > vmax && (vmax = v)
    end
    vmax = vmax > 0 ? vmax : 1.0
    vmin = 0.0
    panel_data = [map(v -> isfinite(v) ? Float64(v) : vmax, d) for d in panel_data_raw]
    for r in 0:(nranks - 1)
        data = panel_data[r + 1]
        nx, ny = size(data)
        row = panel_row(r)
        Label(
            fig[row, 0],
            "rank $r — $(nx)×$(ny)  max(|age|)=$(@sprintf("%.2e", rank_argmax[r + 1].max_val)) yr";
            rotation = π / 2, tellheight = false, fontsize = 12, font = :bold,
        )
        ax = Axis(
            fig[row, 1];
            xlabel = "i (incl. halos)", ylabel = "j (incl. halos)",
        )
        hm = heatmap!(
            ax, 1:nx, 1:ny, data;
            colorrange = (vmin, vmax), colormap = :inferno, nan_color = :lightgray,
        )
        push!(hms, hm)
        # Mark the argmax cell of THIS rank with a red ×
        am = rank_argmax[r + 1]
        scatter!(ax, [am.i], [am.j]; color = :cyan, marker = :xcross, markersize = 14, strokewidth = 2)
        r != 0 && hidexdecorations!(ax; grid = false)
    end
    Colorbar(fig[1:nranks, 2], hms[1]; label = cbar_label)
    save(save_path, fig)
    return fig
end

################################################################################
# Plot 1: column-max log10(|age| + 1) per rank — collapses depth
################################################################################

col_max_path = joinpath(plot_dir, "age_log10_colmax.png")
plot_stack(
    r -> dropdims(maximum(logabs.(rank_age[r + 1]); dims = 3); dims = 3),
    "log10(|age|+1) — column maximum over k (run: $MODEL_CONFIG, $PARTITION)",
    col_max_path;
    cbar_label = "log10(|age|+1)  [yr]",
)
@info "Saved $col_max_path"

################################################################################
# Plot 2: horizontal slice at the global-worst k
################################################################################

slice_path = joinpath(plot_dir, @sprintf("age_log10_k%03d.png", k_worst))
plot_stack(
    r -> logabs.(@view rank_age[r + 1][:, :, k_worst]),
    "log10(|age|+1) — k=$k_worst (worst-rank k, run: $MODEL_CONFIG, $PARTITION)",
    slice_path;
    cbar_label = "log10(|age|+1)  [yr]",
)
@info "Saved $slice_path"

@info "Done. PNGs in $plot_dir"
