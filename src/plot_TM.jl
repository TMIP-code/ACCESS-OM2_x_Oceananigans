"""
Diagnostic plots for transport matrices.

Compares the nonzero values of the averaged transport matrices (avg12a, avg12b,
avg24) against the constant (time-mean) matrix.  Produces two plot types per
comparison:
1. Simple scatter of nzval(avg) vs nzval(const) with 1:1 line
2. Datashader density plot on signed-log (pseudolog10) axes

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_TM.jl [PARENT_MODEL]
```

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER
"""

@info "Loading packages for TM diagnostic plots"
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

@info "TM diagnostic plot configuration"
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

x_const_scaled = Float32.(myscale.(x_const))

################################################################################
# Set up reusable figures with Observables (pre-allocated at correct size)
################################################################################

# Pre-allocate point buffers with correct size; x-columns share existing arrays (no copy)
scatter_y = similar(x_const_f32)
scatter_points = StructArray{Point2f}((x_const_f32, scatter_y))
ds_y = similar(x_const_scaled)
ds_points = StructArray{Point2f}((x_const_scaled, ds_y))

# --- Simple scatter figure ---
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

# --- Datashader figure ---
ds_data = Observable(ds_points)
ds_fig = Figure(; size = (800, 700))
ds_ax = Axis(
    ds_fig[1, 1];
    title = Observable(""),
    xlabel = "const nzval (pseudolog10)",
    ylabel = "",
    xticks = scaled_ticks,
    yticks = scaled_ticks,
    aspect = 1
)
ds_plot = datashader!(ds_ax, ds_data; colormap = datashader_colormap, async = false)
ablines!(ds_ax, 0, 1; color = (:black, 0.1), linewidth = 10)
vlines!(ds_ax, 0; color = (:black, 0.1), linewidth = 10)
hlines!(ds_ax, 0; color = (:black, 0.1), linewidth = 10)
Colorbar(ds_fig[2, 1], ds_plot; label = "Density of matrix coefficients", vertical = false, flipaxis = false)

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

    # --- Update simple scatter (in-place, x-column unchanged) ---
    scatter_y .= y_avg
    notify(scatter_data)
    scatter_ax.title[] = "$label vs const nzval ($parentmodel, $model_config)"
    scatter_ax.ylabel[] = "$label nzval"
    autolimits!(scatter_ax)

    outfile = joinpath(plots_dir, "nzval_scatter_$(label)_vs_const.png")
    save(outfile, scatter_fig)
    @info "Saved $outfile"

    # --- Update datashader (in-place, x-column unchanged) ---
    ds_y .= myscale.(y_avg)
    notify(ds_data)
    ds_ax.title[] = "$label vs const nzval — density ($parentmodel, $model_config)"
    ds_ax.ylabel[] = "$label nzval (pseudolog10)"
    autolimits!(ds_ax)

    outfile_ds = joinpath(plots_dir, "nzval_datashader_$(label)_vs_const.png")
    save(outfile_ds, ds_fig)
    @info "Saved $outfile_ds"

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
@info "TM diagnostic plots complete"
