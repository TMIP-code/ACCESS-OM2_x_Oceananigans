"""
Scatter diagnostic plots for transport matrices.

Compares the nonzero values of the averaged transport matrices (avg12a, avg12b,
avg24) against the constant (time-mean) matrix using simple scatter plots with
1:1 line.

See also: `plot_TM_datashader.jl` for density plots.

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER
"""

@info "Loading packages for TM scatter plots"
flush(stdout); flush(stderr)

using SparseArrays
using JLD2
using Printf
using CairoMakie
using CairoMakie.Makie.StructArrays

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, outputdir) = load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

matrices_dir = joinpath(outputdir, "TM", model_config)
plots_dir = joinpath(matrices_dir, "plots")
mkpath(plots_dir)

@info "TM scatter plot configuration"
@info "- PARENT_MODEL = $parentmodel"
@info "- model_config = $model_config"
@info "- matrices_dir = $matrices_dir"
@info "- plots_dir    = $plots_dir"
flush(stdout); flush(stderr)

################################################################################
# Load const matrix (kept in memory throughout)
################################################################################

const_file = joinpath(matrices_dir, "const", "M.jld2")
isfile(const_file) || error("const matrix not found: $const_file")
M_const = load(const_file, "M")
@info "Loaded const: $(size(M_const)), nnz=$(nnz(M_const))"
flush(stdout); flush(stderr)

x_const = M_const.nzval

# Pre-compute constant-side data (allocated once, reused across all iterations)
x_const_f32 = Float32.(x_const)

################################################################################
# Set up reusable figure with Observables (pre-allocated at correct size)
################################################################################

# Pre-allocate point buffer; x-column shares existing array (no copy)
scatter_y = similar(x_const_f32)
scatter_points = StructArray{Point2f}((x_const_f32, scatter_y))

scatter_data = Observable(scatter_points)
scatter_fig = Figure(; size = (700, 650))
scatter_ax = Axis(
    scatter_fig[1, 1];
    title = Observable(""),
    xlabel = "const nzval",
    ylabel = "",
    aspect = DataAspect()
)
scatter!(scatter_ax, scatter_data; markersize = 1, color = (:black, 0.1), rasterize = true)
ablines!(scatter_ax, 0, 1; color = :red, linewidth = 1)

################################################################################
# Loop over averaged matrices (load one at a time)
################################################################################

avg_labels = ["avg12a", "avg12b", "avg24"]
any_plotted = false

for label in avg_labels
    f = joinpath(matrices_dir, label, "M.jld2")
    if !isfile(f)
        @warn "Matrix not found, skipping: $f"
        continue
    end

    M_avg = load(f, "M")
    @info "Loaded $label: $(size(M_avg)), nnz=$(nnz(M_avg))"

    # Check sparsity pattern match
    if M_avg.colptr != M_const.colptr || M_avg.rowval != M_const.rowval
        @error "Sparsity pattern mismatch between const and $label — skipping"
        continue
    end

    y_avg = M_avg.nzval

    max_abs_diff = maximum(abs(y_avg[i] - x_const[i]) for i in eachindex(y_avg, x_const))
    @info "$label vs const: max|diff| = $(@sprintf("%.4e", max_abs_diff)), nnz = $(length(x_const))"

    # Update scatter in-place (x-column unchanged)
    scatter_y .= y_avg
    notify(scatter_data)
    scatter_ax.title[] = "$label vs const nzval ($parentmodel, $model_config)"
    scatter_ax.ylabel[] = "$label nzval"
    autolimits!(scatter_ax)

    outfile = joinpath(plots_dir, "nzval_scatter_$(label)_vs_const.png")
    save(outfile, scatter_fig)
    @info "Saved $outfile"

    global any_plotted = true

    # Free avg matrix before loading the next one
    M_avg = nothing
    GC.gc()
    flush(stdout); flush(stderr)
end

if !any_plotted
    @warn "No averaged matrices found — nothing to plot"
end

flush(stdout); flush(stderr)
@info "TM scatter plots complete"
