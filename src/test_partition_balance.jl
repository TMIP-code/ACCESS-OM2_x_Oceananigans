"""
Diagnostic for the wet-load partition: read an existing `grid.jld2`, run
each LB method (`:surface`, `:cell`, `:cell_obsolete`) at each of a few
y-partitions, and report per-rank loads under all three metrics.

The "true" load is the per-rank count of non-immersed 3D cells obtained
by calling Oceananigans' `immersed_cell(i, j, k, grid)` on the actual
`ImmersedBoundaryGrid + PartialCellBottom` — the same thing the
simulation uses to decide which cells are stepped. Comparing each
method's slab boundaries against this ground truth tells us whether
`:cell` (new) is better balanced than `:cell_obsolete` (what was used
to build the existing `_LB` partitions).

Just reads `preprocessed_inputs/{PM}/{EXP}/grid.jld2` — no MPI, no GPU.

Usage:
    PARENT_MODEL=ACCESS-OM2-1   julia --project src/test_partition_balance.jl
    PARENT_MODEL=ACCESS-OM2-025 julia --project src/test_partition_balance.jl
    PARENT_MODEL=ACCESS-OM2-01  julia --project src/test_partition_balance.jl

Optional: PARTITIONS=1x2,1x4,1x8 (default), METHODS=surface,cell,cell_obsolete (default).
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, immersed_cell
using JLD2
using Printf
using Statistics: mean
using CairoMakie

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment, experiment_dir) = load_project_config()
grid_file = joinpath(experiment_dir, "grid.jld2")
isfile(grid_file) || error("grid.jld2 not found at $grid_file")

# Plots are TW-independent (depend only on the grid), so write them at
# outputs/{PM}/{EXP}/partition_balance — not under outputdir, which gets
# routed under test/TR..._MLD... when MLD_TIME_WINDOW is set in env.
plot_dir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel, experiment, "partition_balance"))
mkpath(plot_dir)

partitions_str = get(ENV, "PARTITIONS", "1x2,1x4,1x8")
methods_str = get(ENV, "METHODS", "surface,cell,mix")

py_list = [parse(Int, split(p, "x")[2]) for p in split(partitions_str, ",")]
method_list = [Symbol(strip(m)) for m in split(methods_str, ",")]

@info "Configuration"
@info "  PARENT_MODEL = $parentmodel"
@info "  grid_file    = $grid_file"
@info "  py_list      = $py_list"
@info "  method_list  = $method_list"
flush(stdout); flush(stderr)

################################################################################
# Load grid once
################################################################################

@info "Loading ImmersedBoundaryGrid (CPU) ..."
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())
ug = grid.underlying_grid
Nx, Ny, Nz = size(ug)
Hy = ug.Hy
@info "Grid size = ($Nx, $Ny, $Nz), Hy = $Hy"
flush(stdout); flush(stderr)

################################################################################
# Ground-truth per-cell wet mask via `immersed_cell` (compute once, reuse)
################################################################################

@info "Computing ground-truth wet mask via immersed_cell ..."
flush(stdout); flush(stderr)
wet3D = Array{Bool}(undef, Nx, Ny, Nz)
@inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
    wet3D[i, j, k] = !immersed_cell(i, j, k, grid)
end

# Per-y-row reductions of the ground truth
true_cells_per_row = [count(view(wet3D, :, j, :)) for j in 1:Ny]   # 3D cells
true_columns_per_row = [count(any(view(wet3D, i, j, :)) for i in 1:Nx) for j in 1:Ny]

total_wet_cells = sum(true_cells_per_row)
total_wet_cols = sum(true_columns_per_row)
@info "Total true wet cells   = $total_wet_cells"
@info "Total true wet columns = $total_wet_cols"
flush(stdout); flush(stderr)

################################################################################
# Helpers
################################################################################

"""Equal-split y partition (matches `Partition(px, py)` default)."""
function equal_y_sizes(Ny::Int, py::Int)
    base = Ny ÷ py
    rem = Ny - base * py
    # Mirror Oceananigans' partitioning: first `rem` ranks get one extra row.
    return Tuple(r ≤ rem ? base + 1 : base for r in 1:py)
end

"""Compute slab y-row ranges from a sizes tuple."""
function slab_ranges(sizes)
    ranges = Vector{UnitRange{Int}}(undef, length(sizes))
    j = 1
    for (r, n) in enumerate(sizes)
        ranges[r] = j:(j + n - 1)
        j += n
    end
    return ranges
end

"""For a partition, compute per-rank (slab_size, true_cells, true_columns)."""
function per_rank_loads(sizes)
    ranges = slab_ranges(sizes)
    return [
        (
                slab = length(rng),
                cells = sum(true_cells_per_row[rng]),
                columns = sum(true_columns_per_row[rng]),
            )
            for rng in ranges
    ]
end

imbalance_ratio(xs) = maximum(xs) / max(1, minimum(xs))
imbalance_pct(xs) = 100 * (maximum(xs) - mean(xs)) / mean(xs)

# Inlined greedy splitter — same algorithm as `compute_lb_y_sizes` but
# operates on a pre-computed `wet[j]` instead of re-constructing the IBG.
# Avoids redundant grid loads when comparing methods.
function greedy_y_split(wet::AbstractVector{<:Real}, nranks_y::Int; min_size::Int = 0)
    Ny_local = length(wet)
    total_wet = sum(wet)
    target = total_wet / nranks_y
    local_Ny = zeros(Int, nranks_y)
    cum = zero(eltype(wet))
    j = 1
    for r in 1:(nranks_y - 1)
        slab = zero(eltype(wet))
        while j ≤ Ny_local && cum + slab < target * r
            slab += wet[j]
            local_Ny[r] += 1
            j += 1
        end
        cum += slab
    end
    local_Ny[end] = Ny_local - sum(local_Ny[1:(end - 1)])
    if min_size > 0
        for r in 1:nranks_y
            while local_Ny[r] < min_size
                donor = argmax(local_Ny)
                donor == r && error("greedy_y_split: cannot satisfy min_size=$min_size")
                local_Ny[donor] -= 1
                local_Ny[r] += 1
            end
        end
    end
    @assert sum(local_Ny) == Ny_local
    return Tuple(local_Ny)
end

# Per-y-row load arrays for every LB method (computed once, reused).
# `:cell` is just `true_cells_per_row` (via immersed_cell).
# `:surface` uses bottom < 0 (same as production `:surface`).
# `:mix` is the equal-weighted normalised mix of cells & columns
# (both via immersed_cell — matches production `:mix`).
# `:cell_obsolete` uses z_center > bottom (back-comparison only).
bottom = load(grid_file, "bottom")
z_faces = load(grid_file, "z_faces")
z_centers = @. 0.5 * (z_faces[1:Nz] + z_faces[2:(Nz + 1)])
surface_per_row = [count(<(0), view(bottom, :, j)) for j in 1:Ny]
cellobs_per_row = [sum(count(>(bottom[i, j]), z_centers) for i in 1:Nx) for j in 1:Ny]
let Tc = sum(true_cells_per_row), Tk = sum(true_columns_per_row)
    global mix_per_row = Float64[
        true_cells_per_row[j] / Tc + true_columns_per_row[j] / Tk
            for j in 1:Ny
    ]
end
load_per_row = Dict(
    :cell => true_cells_per_row,
    :surface => surface_per_row,
    :mix => mix_per_row,
    :cell_obsolete => cellobs_per_row,
)

################################################################################
# Run all (py, method) combinations
################################################################################

println()
println("="^96)
println("Model: $parentmodel    Nx=$Nx  Ny=$Ny  Nz=$Nz")
println("Ground truth (via immersed_cell): total wet cells = $total_wet_cells, total wet columns = $total_wet_cols")
println("="^96)

# Plot-only methods: drop `:cell_obsolete` from the visual but keep it in tables.
plot_methods = filter(!=(:cell_obsolete), method_list)

# Distinct categorical color per RANK (up to 8 ranks). Sampled from
# Makie's :tab10 palette so adjacent ranks are easy to tell apart.
rank_colors = Makie.to_colormap(:tab10)[1:8]

for py in py_list
    println()
    println("─"^96)
    println("Partition 1×$py")
    println("─"^96)

    # Schemes: equal split + each LB method
    schemes = NamedTuple[]
    push!(schemes, (name = "equal", sizes = equal_y_sizes(Ny, py)))
    for m in method_list
        sz = greedy_y_split(load_per_row[m], py; min_size = Hy + 2)
        push!(schemes, (name = String(m), sizes = sz))
    end

    # Header
    @printf("%-16s  %-12s  | per-rank true wet cells (×10⁶)     | per-rank true wet columns (×10³)    | slab Ny\n", "scheme", "imb%(cells)")
    println("─"^96)

    per_scheme = NamedTuple[]
    for s in schemes
        per = per_rank_loads(s.sizes)
        cells_vec = [p.cells   for p in per]
        cols_vec = [p.columns for p in per]
        slab_vec = [p.slab    for p in per]
        ratio_cells = imbalance_ratio(cells_vec)
        imb_cells_pct = imbalance_pct(cells_vec)

        cells_str = join([@sprintf("%6.2f", c / 1.0e6) for c in cells_vec], " ")
        cols_str = join([@sprintf("%5.1f", c / 1.0e3) for c in cols_vec], " ")
        slab_str = join([@sprintf("%4d", n) for n in slab_vec], " ")

        @printf(
            "%-16s  +%5.1f%% (×%.3f) | %s  | %s  | %s\n",
            s.name, imb_cells_pct, ratio_cells, cells_str, cols_str, slab_str,
        )

        push!(
            per_scheme, (
                name = s.name, sizes = s.sizes,
                cells = cells_vec, cols = cols_vec, slab = slab_vec,
                ratio = ratio_cells, imb_pct = imb_cells_pct,
            )
        )
    end

    # Filter schemes shown in the figure (cell_obsolete excluded — too close
    # to :cell to be useful visually, and clutters the legend).
    plot_schemes = filter(s -> Symbol(s.name) == :equal || Symbol(s.name) in plot_methods, per_scheme)

    ############################################################################
    # Plot: trace (top) + rank coverage ribbons + per-rank bar charts.
    ############################################################################

    fig = Figure(size = (1200, 980), fontsize = 13)
    Label(
        fig[0, 1:3],
        "$parentmodel  partition 1×$py  (Nx=$Nx, Ny=$Ny, Nz=$Nz, total wet cells = $(round(Int, total_wet_cells / 1.0e6))M)";
        fontsize = 16, font = :bold,
    )

    # Row 1: per-y-row trace — wet cells (left axis) + wet columns
    # (right axis, twin). Different scales (2D vs 3D), so we use two
    # axes sharing the same panel — see
    # https://docs.makie.org/stable/reference/blocks/axis#Creating-a-twin-axis
    cells_color = :black
    cols_color = :crimson
    ax1 = Axis(
        fig[1, 1:2];
        title = "Per y-row: wet 3D cells (black) and wet surface columns (red)",
        xlabel = "j (south → north)",
        ylabel = "wet cells per row", ylabelcolor = cells_color,
        yticklabelcolor = cells_color,
        limits = (nothing, nothing, 0, nothing),
    )
    ax1_twin = Axis(
        fig[1, 1:2];
        ylabel = "wet columns per row", ylabelcolor = cols_color,
        yticklabelcolor = cols_color,
        yaxisposition = :right,
        limits = (nothing, nothing, 0, nothing),
    )
    hidespines!(ax1_twin)
    hidexdecorations!(ax1_twin)
    linkxaxes!(ax1, ax1_twin)
    lines!(ax1, 1:Ny, true_cells_per_row; color = cells_color, linewidth = 1)
    lines!(ax1_twin, 1:Ny, true_columns_per_row; color = cols_color, linewidth = 1)

    # Row 2: rank coverage ribbons — one row per scheme, each y-row's slab
    # drawn as a thick colored segment along x. Colors index ranks.
    ax2 = Axis(
        fig[2, 1:2];
        title = "Rank coverage along j (one row per scheme; colors = rank index)",
        xlabel = "j (south → north)", ylabel = "",
        yticks = (1:length(plot_schemes), [s.name for s in plot_schemes]),
    )
    hidexdecorations!(ax2, ticklabels = false, ticks = false, label = false)
    for (row, s) in enumerate(plot_schemes)
        boundaries = [0; cumsum(collect(s.sizes))]
        for r in 1:length(s.sizes)
            j0 = boundaries[r] + 1
            j1 = boundaries[r + 1]
            lines!(
                ax2, [j0, j1], [row, row];
                color = rank_colors[mod1(r, length(rank_colors))],
                linewidth = 14,
            )
        end
    end
    # Match the x-range to ax1 for visual alignment
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0.5, Ny + 0.5)
    ylims!(ax2, 0.4, length(plot_schemes) + 0.6)

    # Rank legend (south → north)
    rank_leg = [
        PolyElement(color = rank_colors[mod1(r, length(rank_colors))], strokewidth = 0)
            for r in 1:py
    ]
    Legend(
        fig[2, 3], rank_leg, ["rank $(r - 1)" for r in 1:py];
        title = "rank (0 = south)", framevisible = false, tellwidth = true,
    )

    # Row 3: grouped bar chart of per-rank wet cells (3D)
    ax3 = Axis(
        fig[3, 1];
        title = "Per-rank wet cells (3D)", xlabel = "rank (0 = south)", ylabel = "wet cells (×10⁶)",
        xticks = (0:(py - 1), string.(0:(py - 1))),
    )
    nschemes_p = length(plot_schemes)
    width = 0.85 / nschemes_p
    scheme_fill = Dict("equal" => :gray60, "surface" => :steelblue, "cell" => :seagreen, "mix" => :purple, "cell_obsolete" => :darkorange)
    for (k, s) in enumerate(plot_schemes)
        xs = (0:(py - 1)) .+ (k - (nschemes_p + 1) / 2) * width
        barplot!(
            ax3, xs, s.cells ./ 1.0e6;
            width = width * 0.95, color = scheme_fill[s.name], label = s.name,
        )
    end
    hlines!(ax3, [total_wet_cells / py / 1.0e6]; color = :red, linestyle = :dash, linewidth = 1.5)

    # Row 3: grouped bar chart of per-rank wet columns (2D — per-column work)
    ax4 = Axis(
        fig[3, 2];
        title = "Per-rank wet columns (2D — per-column work, e.g. implicit vertical diffusion)",
        xlabel = "rank (0 = south)", ylabel = "wet columns (×10³)",
        xticks = (0:(py - 1), string.(0:(py - 1))),
    )
    for (k, s) in enumerate(plot_schemes)
        xs = (0:(py - 1)) .+ (k - (nschemes_p + 1) / 2) * width
        barplot!(
            ax4, xs, s.cols ./ 1.0e3;
            width = width * 0.95, color = scheme_fill[s.name], label = s.name,
        )
    end
    hlines!(ax4, [total_wet_cols / py / 1.0e3]; color = :red, linestyle = :dash, linewidth = 1.5)

    Legend(
        fig[3, 3], ax3; title = "scheme", framevisible = false, tellwidth = true,
    )

    # Vertical row-size hints
    rowsize!(fig.layout, 1, Relative(0.4))
    rowsize!(fig.layout, 2, Relative(0.18))
    rowsize!(fig.layout, 3, Relative(0.42))

    out_png = joinpath(plot_dir, "$(parentmodel)_1x$(py).png")
    save(out_png, fig)
    @info "Saved plot: $out_png"
    flush(stdout); flush(stderr)
end

println()
println("="^96)
println("Key:")
println("  imb%(cells)  = (max - mean) / mean × 100, on true wet cells (ground truth via immersed_cell).")
println("  ×ratio       = max/min ratio of true wet cells across ranks. 1.000 = perfect balance.")
println("  cells / cols = per-rank counts (rank 0 → py-1 = south → north; rank 0 owns j=1, the southernmost slab).")
println("  Plots saved under: $plot_dir")
println("="^96)
flush(stdout); flush(stderr)
