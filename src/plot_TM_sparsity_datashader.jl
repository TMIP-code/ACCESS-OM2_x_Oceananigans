"""
Datashader sparsity-pattern plot of a single transport matrix.

Two panels:
  (1) the full matrix nonzero PATTERN (column index x, row index y, row 1 at top);
  (2) a zoom onto the first two layers k ∈ {1, 2} (k=1 is the bottom layer, k=Nz
      the surface). The matrix is wet-cell-compacted with column-major ordering
      (idx = findall(wet3D), k slowest-varying), so those cells form a contiguous
      head block [1 .. kend] at the lowest matrix indices.

Rendered with datashader so it scales to the full ~tens-of-millions of nonzeros
without OOMing (the inline Makie `spy!` was dropped from create_matrix.jl in
fe094dc for exactly that reason). One matrix at a time — NOT a two-matrix
comparison (see plot_TM_datashader.jl for that).

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER, TM_LABEL (which subdir's M.jld2; default "const")
"""

@info "Loading packages for TM sparsity datashader plot"
flush(stdout); flush(stderr)

using SparseArrays
using JLD2
using Printf
using Oceananigans
using CairoMakie
using CairoMakie.Makie.StructArrays

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config()

model_config = require_env("MODEL_CONFIG")
label = get(ENV, "TM_LABEL", "const")

matrices_dir = joinpath(outputdir, "TM", model_config)
plots_dir = joinpath(matrices_dir, "plots")
mkpath(plots_dir)

@info "TM sparsity datashader plot: $label"
@info "- PARENT_MODEL = $parentmodel"
@info "- model_config = $model_config"
@info "- plots_dir    = $plots_dir"
flush(stdout); flush(stderr)

################################################################################
# Load matrix
################################################################################

file = joinpath(matrices_dir, label, "M.jld2")
isfile(file) || error("Matrix not found: $file")

M = load(file, "M")
m, n = size(M)
@info "Loaded $label: $(size(M)), nnz=$(nnz(M))"
flush(stdout); flush(stderr)

################################################################################
# Wet-cell index map → matrix-index range of the first-2-layers block (k ∈ {1,2})
################################################################################

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
(; wet3D, Nidx) = compute_wet_mask(grid)
Nz′ = size(wet3D, 3)
Nidx == n || @warn "wet-cell count Nidx=$Nidx ≠ matrix dim n=$n — block slice may be off"

# Matrix index n maps to the n-th wet cell of idx = findall(wet3D), which is
# column-major (k slowest-varying), so the first-2-layer (k ∈ {1, 2}) wet cells
# are the contiguous HEAD [1 .. kend]. wet cells per vertical layer = sum over the
# horizontal dims; the first two layers' counts give kend. (NB: in this grid k=1
# is the BOTTOM layer, k=Nz the surface.)
layer_counts = vec(sum(wet3D; dims = (1, 2)))      # wet cells in each layer k
kend = layer_counts[1] + layer_counts[2]
@info "Wet cells per layer (first 5): $(layer_counts[1:min(5, Nz′)])"
@info "First-2-layers block (k ∈ {1, 2}): $kend cells → matrix indices 1 .. $kend"
flush(stdout); flush(stderr)

################################################################################
# Expand (row, column) index of every stored nonzero (CSC, no value copy)
################################################################################

rows = rowvals(M)                 # row index of each stored entry (length nnz)
cols = Vector{Int}(undef, length(rows))
@inbounds for j in 1:n
    for k in nzrange(M, j)
        cols[k] = j
    end
end
@info "Expanded $(length(rows)) nonzero coordinates"
flush(stdout); flush(stderr)

points = StructArray{Point2f}((Float32.(cols), Float32.(rows)))

# Subset for the zoom panel: entries whose row AND column lie in the first-2-layers block.
in_block = (rows .≤ kend) .& (cols .≤ kend)
zoom_points = StructArray{Point2f}((Float32.(cols[in_block]), Float32.(rows[in_block])))
@info "First-2-layers-block nonzeros: $(count(in_block))"
flush(stdout); flush(stderr)

################################################################################
# Build two-panel datashader sparsity plot (x = column, y = row, row 1 at top)
################################################################################

datashader_colormap = cgrad([:white; collect(cgrad(:managua))])

fig = Figure(; size = (1500, 820))
Label(fig[0, 1:2], "$label TM sparsity ($parentmodel)  —  $model_config  —  $(m)×$(n), nnz=$(nnz(M))"; fontsize = 16)

ax1 = Axis(
    fig[1, 1];
    title = "full matrix",
    xlabel = "column index", ylabel = "row index",
    yreversed = true, aspect = DataAspect(),
)
ds1 = datashader!(ax1, points; colormap = datashader_colormap, async = false)
limits!(ax1, 0.5, n + 0.5, 0.5, m + 0.5)
# Outline the zoom region on the full panel.
lines!(
    ax1,
    [0.5, kend + 0.5, kend + 0.5, 0.5, 0.5],
    [0.5, 0.5, kend + 0.5, kend + 0.5, 0.5];
    color = (:red, 0.8), linewidth = 1.5,
)
Colorbar(fig[2, 1], ds1; label = "Density of nonzeros", vertical = false, flipaxis = false)

ax2 = Axis(
    fig[1, 2];
    title = "zoom: first 2 layers k ∈ {1, 2}  (indices 1..$kend)",
    xlabel = "column index", ylabel = "row index",
    yreversed = true, aspect = DataAspect(),
)
ds2 = datashader!(ax2, zoom_points; colormap = datashader_colormap, async = false)
limits!(ax2, 0.5, kend + 0.5, 0.5, kend + 0.5)
Colorbar(fig[2, 2], ds2; label = "Density of nonzeros", vertical = false, flipaxis = false)

outfile = joinpath(plots_dir, "sparsity_datashader_$(label).png")
save(outfile, fig)
@info "Saved $outfile"

flush(stdout); flush(stderr)
@info "TM sparsity datashader plot complete"
