"""
Datashader density plot comparing two transport matrices.

Compares the nonzero values of two transport matrices (specified by
`TM_LABEL_X` and `TM_LABEL_Y` env vars) using a datashader density plot
on signed-log (pseudolog10) axes.

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER, TM_LABEL_X, TM_LABEL_Y
"""

@info "Loading packages for TM datashader plot"
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

label_x = get(ENV, "TM_LABEL_X", "const")
label_y = get(ENV, "TM_LABEL_Y", "avg24")

matrices_dir = joinpath(outputdir, "TM", model_config)
plots_dir = joinpath(matrices_dir, "plots")
mkpath(plots_dir)

@info "TM datashader plot: $label_y vs $label_x"
@info "- PARENT_MODEL = $parentmodel"
@info "- model_config = $model_config"
@info "- matrices_dir = $matrices_dir"
@info "- plots_dir    = $plots_dir"
flush(stdout); flush(stderr)

################################################################################
# Load matrices
################################################################################

file_x = joinpath(matrices_dir, label_x, "M.jld2")
file_y = joinpath(matrices_dir, label_y, "M.jld2")
isfile(file_x) || error("Matrix not found: $file_x")
isfile(file_y) || error("Matrix not found: $file_y")

M_x = load(file_x, "M")
@info "Loaded $label_x: $(size(M_x)), nnz=$(nnz(M_x))"
flush(stdout); flush(stderr)

M_y = load(file_y, "M")
@info "Loaded $label_y: $(size(M_y)), nnz=$(nnz(M_y))"
flush(stdout); flush(stderr)

# Check sparsity pattern match
if M_x.colptr != M_y.colptr || M_x.rowval != M_y.rowval
    error("Sparsity pattern mismatch between $label_x and $label_y")
end

nzval_x = M_x.nzval
nzval_y = M_y.nzval

max_abs_diff = maximum(abs(nzval_y[i] - nzval_x[i]) for i in eachindex(nzval_y, nzval_x))
@info "$label_y vs $label_x: max|diff| = $(@sprintf("%.4e", max_abs_diff)), nnz = $(length(nzval_x))"
flush(stdout); flush(stderr)

################################################################################
# Datashader scale and ticks (signed-log)
################################################################################

myscale(x) = Makie.pseudolog10(1.0e7x)
powersoften = -6:-2
ticks = [-reverse(exp10.(powersoften)); 0; exp10.(powersoften)]
signedstr(x) = x > 0 ? "+$x" : "−$(-x)"
ticklabels = [rich("10", superscript(signedstr(i))) for i in powersoften]
ticklabels = [[rich("−", x) for x in reverse(ticklabels)]; rich("0"); [rich("+", x) for x in ticklabels]]
scaled_ticks = (myscale.(ticks), ticklabels)

datashader_colormap = cgrad([:white; collect(cgrad(:managua))])

################################################################################
# Build plot
################################################################################

scaled_x = Float32.(myscale.(nzval_x))
scaled_y = Float32.(myscale.(nzval_y))
points = StructArray{Point2f}((scaled_x, scaled_y))

fig = Figure(; size = (800, 700))
ax = Axis(
    fig[1, 1];
    title = "$label_y vs $label_x nzval — density ($parentmodel, $model_config)",
    xlabel = "$label_x nzval (pseudolog10)",
    ylabel = "$label_y nzval (pseudolog10)",
    xticks = scaled_ticks,
    yticks = scaled_ticks,
    aspect = 1
)
ds = datashader!(ax, points; colormap = datashader_colormap, async = false)
ablines!(ax, 0, 1; color = (:black, 0.1), linewidth = 10)
vlines!(ax, 0; color = (:black, 0.1), linewidth = 10)
hlines!(ax, 0; color = (:black, 0.1), linewidth = 10)
Colorbar(fig[2, 1], ds; label = "Density of matrix coefficients", vertical = false, flipaxis = false)

outfile = joinpath(plots_dir, "nzval_datashader_$(label_y)_vs_$(label_x).png")
save(outfile, fig)
@info "Saved $outfile"

flush(stdout); flush(stderr)
@info "TM datashader plot complete"
