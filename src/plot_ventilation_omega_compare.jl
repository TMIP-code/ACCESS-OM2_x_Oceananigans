"""
Compare the surface ventilation diagnostic `calVdown` across OMEGA values
for a single (PARENT_MODEL, TIME_WINDOW, leg) at one model_config tag.

Loads four ventilation files (OMEGA ∈ {all, z500, z1500, z2500}) from
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK[_QAxB]/ventilation{omega_suffix}.jld2
normalises each with `1e16 / vtot` (same vtot — full ocean volume, set
once at compute_ventilation_diagnostic.jl:229), and lays out a 2 × 4 figure:

  Row 1 — per-OMEGA `calV` maps (orange ramp, pseudo-log levels picked
          from the joint max across the four panels)
  Row 2 — four "layer-contribution" diff maps (PRGn diverging, same
          normalization so a diff equals the ventilation contribution
          from the depth layer between the two OMEGA values):
              [2,1] all − z500     (0–500 m source)
              [2,2] z500 − z1500   (500–1500 m source)
              [2,3] z1500 − z2500  (1500–2500 m source)
              [2,4] z500 − z2500   (500–2500 m source; sanity-check sum)

Writes one PNG per leg to
  outputs/{PM}/{EXP}/plots/{MC}/calVdown_omega_compare_{forward|adjoint}_{TW}.png

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1968-1977 GRID_HZ=4 \\
    julia --project src/plot_ventilation_omega_compare.jl
```

Required env vars: PARENT_MODEL, TIME_WINDOW, plus the usual model-config
vars (VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER,
TIMESTEP_MULT, LUMP_AND_SPRAY, MONTHLY_KAPPAV, KAPPA_*) that resolve to
MODEL_CONFIG via env_defaults.sh. Optional: TRAF (default no).
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using CairoMakie
using GeoMakie
using GeometryBasics
using JLD2
using Printf
using Statistics

@info "Packages loaded"
flush(stdout); flush(stderr)

include("shared_functions.jl")
include(joinpath(@__DIR__, "shared_utils", "plotting_functions.jl"))

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment, experiment_dir, outputdir) = load_project_config()

model_config = require_env("MODEL_CONFIG")
TW = require_env("TIME_WINDOW")
TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
leg_tag = TRAF ? "adjoint" : "forward"
leg_label_long = TRAF ? "Adjoint 𝒱↓" : "Forward 𝒱↓"

ls = parse_lump_and_spray()

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

# Experiment-level outputs root (parent of the per-TW dirs).
exp_outdir = dirname(outputdir)

const OMEGAS = ["all", "z500", "z1500", "z2500"]
const DIFFS = [
    ("all", "z500", "0 – 500 m"),
    ("z500", "z1500", "500 – 1500 m"),
    ("z1500", "z2500", "1500 – 2500 m"),
    ("z500", "z2500", "500 – 2500 m"),
]

function ventilation_path(omega_tag)
    suffix = omega_tag == "all" ? "" : "_$(omega_tag)"
    basename = "ventilation$(suffix).jld2"
    tw_root = joinpath(exp_outdir, TW)
    periodic_roots = unique(
        [
            isempty(gpu_tag) ?
                joinpath(tw_root, "periodic", model_config) :
                joinpath(tw_root, "periodic", model_config, gpu_tag),
            joinpath(tw_root, "periodic", model_config),
        ]
    )
    candidate_dirs = unique(
        [
            joinpath(root, sub)
                for root in periodic_roots
                for sub in ("NK$(ls.dir_suffix)", "NK")
        ]
    )
    for d in candidate_dirs
        f = joinpath(d, basename)
        isfile(f) && return f
    end
    error(
        "$basename not found for OMEGA=$omega_tag, TW=$TW. Tried:\n" *
            join(["  " * joinpath(d, basename) for d in candidate_dirs], "\n"),
    )
end

vent_files = Dict(omega_tag => ventilation_path(omega_tag) for omega_tag in OMEGAS)

plot_dir = joinpath(exp_outdir, "plots", model_config)
mkpath(plot_dir)

@info "plot_ventilation_omega_compare.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- EXPERIMENT    = $experiment"
@info "- TIME_WINDOW   = $TW"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
for omega_tag in OMEGAS
    @info "- ventilation[$omega_tag] = $(vent_files[omega_tag])"
end
@info "- output dir    = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Load data
################################################################################

@info "Loading ventilation files"
flush(stdout); flush(stderr)
data = Dict(omega_tag => load(vent_files[omega_tag]) for omega_tag in OMEGAS)

# vtot from the OMEGA=all file; assert the others match (they should — vtot is
# the full ocean volume, independent of OMEGA).
vtot = data["all"]["vtot"]
for omega_tag in OMEGAS
    δ = abs(data[omega_tag]["vtot"] - vtot) / vtot
    δ < 1.0e-9 || @warn "v_tot disagreement" omega = omega_tag this_vtot = data[omega_tag]["vtot"] all_vtot = vtot reldiff = δ
end

Az_surf = data["all"]["Az_surf"]

# Sanity: all calVdown_raw fields share the same shape.
ref_size = size(data["all"]["calVdown_raw"])
for omega_tag in OMEGAS
    size(data[omega_tag]["calVdown_raw"]) == ref_size ||
        error("Shape mismatch for OMEGA=$omega_tag")
end

# Normalise: % v_tot / (10,000 km)². Prefactor 1e16 / vtot.
norm_factor = 1.0e16 / vtot
@info @sprintf("v_tot = %.3e m³;  1e16/v_tot = %.3e", vtot, norm_factor)

calV = Dict(omega_tag => data[omega_tag]["calVdown_raw"] .* norm_factor for omega_tag in OMEGAS)
diffs = Dict(
    (a, b) => calV[a] .- calV[b]
        for (a, b, _label) in DIFFS
)

for omega_tag in OMEGAS
    vals = filter(isfinite, calV[omega_tag])
    @info @sprintf(
        "calV[%-5s] [%% v_tot / (10,000 km)²]:  min = %+.3e   mean = %+.3e   max = %+.3e",
        omega_tag, minimum(vals), mean(vals), maximum(vals),
    )
end
for (a, b, label) in DIFFS
    vals = filter(isfinite, diffs[(a, b)])
    @info @sprintf(
        "diff[%-5s − %-5s] (%s):  min = %+.3e   mean = %+.3e   max = %+.3e",
        a, b, label, minimum(vals), mean(vals), maximum(vals),
    )
end

################################################################################
# Build gridmetrics from the Oceananigans tripolar grid
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny = ref_size
gridmetrics = gridmetrics_from_grid(grid, Nx, Ny)

lon_window_start = 20

################################################################################
# Colour scales — orange ramp for mean panels, PRGn for diff panels.
# Pick a single shared level set per row (so all 4 maps in a row share the
# same colorbar). The same `pick_levels` ladder used by plot_ventilation.jl.
################################################################################

function pick_levels(maxv; user_p = nothing)
    p = if user_p !== nothing
        user_p
    else
        k = floor(Int, log10(maxv / 30))
        10.0^k
    end
    return Float32[0, p, 3p, 10p, 30p, 100p]
end

user_p = (haskey(ENV, "VENT_LEVELS_P") && !isempty(ENV["VENT_LEVELS_P"])) ?
    parse(Float64, ENV["VENT_LEVELS_P"]) : nothing

maxv_mean = maximum([maximum(filter(isfinite, calV[o])) for o in OMEGAS])
levels_mean = pick_levels(maxv_mean; user_p)
@info "Mean panel levels = $levels_mean  (data max ≈ $maxv_mean)"

cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)

maxv_diff = maximum([maximum(abs, filter(isfinite, diffs[(a, b)])) for (a, b, _) in DIFFS])
diff_pos = pick_levels(maxv_diff; user_p)
levels_diff = Float32[-reverse(diff_pos[2:end]); diff_pos[2:end]]
@info "Diff panel levels = $levels_diff  (data |max| ≈ $maxv_diff)"

cm_diff_full = cgrad(withwhitecenter(Makie.ColorSchemes.PRGn), length(levels_diff) + 1; categorical = true)
lowclip_diff = cm_diff_full[1]
highclip_diff = cm_diff_full[end]
cm_diff = cgrad(collect(cm_diff_full[2:(end - 1)]); categorical = true)
scale_diff = mk_piecewise_linear(levels_diff)

################################################################################
# Build the 2 × 4 figure
#
# Layout (3 rows × 5 cols, plus title row 0):
#   row 0: title spanning cols 1-5
#   row 1: cb_mean | (a) all | (b) z500 | (c) z1500 | (d) z2500
#   row 2: cb_diff | (e) all−z500 | (f) z500−z1500 | (g) z1500−z2500 | (h) z500−z2500
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)

fig = Figure(;
    size = (2200, 900), fontsize = 14,
    fonts = (; regular = "Arial", bold = "Arial Bold"),
)

xticks_map = -0:90:1000
yticks_map = -90:30:90

mean_label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2"))
diff_label = rich("Δ % v", subscript("tot"), " / (10,000 km)", superscript("2"))

# Title row.
Label(
    fig[0, 1:5],
    rich(
        leg_label_long, " — ", parentmodel, " — ", TW, " — ", model_config;
        font = :bold,
    );
    fontsize = 16, halign = :center,
)

# Mean-row colorbar (col 1).
N_mean = length(levels_mean) - 1
cb_mean = Colorbar(
    fig[1, 1];
    colormap = cm_mean,
    colorrange = (0, N_mean),
    highclip = highclip_mean,
    ticks = (0:N_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = mean_label,
    vertical = true, flipaxis = false,
)
cb_mean.height = Relative(0.8)

# Diff-row colorbar (col 1).  levels_diff has 10 EDGES → 9 BINS, so
# colorrange = (0, 9) with 10 tick positions, one per edge.
N_diff = length(levels_diff) - 1
cb_diff = Colorbar(
    fig[2, 1];
    colormap = cm_diff,
    colorrange = (0, N_diff),
    lowclip = lowclip_diff,
    highclip = highclip_diff,
    ticks = (0:N_diff, [@sprintf("%g", x) for x in levels_diff]),
    label = diff_label,
    vertical = true, flipaxis = false,
)
cb_diff.height = Relative(0.8)

# Convenience: per-OMEGA short labels and panel letters.
omega_titles = Dict(
    "all" => "OMEGA = all (full ocean)",
    "z500" => "OMEGA = z500 (below 500 m)",
    "z1500" => "OMEGA = z1500 (below 1500 m)",
    "z2500" => "OMEGA = z2500 (below 2500 m)",
)
panel_letters_mean = ('a', 'b', 'c', 'd')
panel_letters_diff = ('e', 'f', 'g', 'h')

# ----- Row 1: per-OMEGA maps -----
for (k, omega_tag) in enumerate(OMEGAS)
    ax = Axis(
        fig[1, k + 1];
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
        xticks = (xticks_map, lonticklabel.(xticks_map)),
        yticks = (yticks_map, latticklabel.(yticks_map)),
    )
    plotmap!(
        ax, calV[omega_tag], gridmetrics;
        colorrange = extrema(levels_mean),
        colormap = cm_mean,
        highclip = highclip_mean,
        colorscale = scale_mean,
        lon_window_start,
    )
    add_coastlines!(ax)
    ylims!(ax, (-90, 90))
    text!(
        ax, 0, 1;
        text = rich("($(panel_letters_mean[k])) ", omega_titles[omega_tag]),
        align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
    )
    hidexdecorations!(ax; ticks = false, grid = false, ticklabels = true, label = true)
    if k > 1
        hideydecorations!(ax; ticks = false, grid = false, ticklabels = true, label = true)
    end
end

# ----- Row 2: diff maps -----
for (k, (a, b, label)) in enumerate(DIFFS)
    ax = Axis(
        fig[2, k + 1];
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
        xticks = (xticks_map, lonticklabel.(xticks_map)),
        yticks = (yticks_map, latticklabel.(yticks_map)),
    )
    plotmap!(
        ax, diffs[(a, b)], gridmetrics;
        colorrange = extrema(levels_diff),
        colormap = cm_diff,
        lowclip = lowclip_diff,
        highclip = highclip_diff,
        colorscale = scale_diff,
        lon_window_start,
    )
    add_coastlines!(ax)
    ylims!(ax, (-90, 90))
    text!(
        ax, 0, 1;
        text = rich("($(panel_letters_diff[k])) ", a, " − ", b, " — ", label, " source"),
        align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
    )
    if k > 1
        hideydecorations!(ax; ticks = false, grid = false, ticklabels = true, label = true)
    end
end

# Tight column spacing between map panels (cols 2–5); leave the colorbar
# column (col 1) at its default width.
colgap!(fig.layout, 1, 10)
for c in 2:4
    colgap!(fig.layout, c, 5)
end
rowgap!(fig.layout, 1, 8)

################################################################################
# Save
################################################################################

out_png = joinpath(plot_dir, "calVdown_omega_compare_$(leg_tag)_$(TW).png")
@info "Saving $out_png"
flush(stdout); flush(stderr)
save(out_png, fig)
@info "plot_ventilation_omega_compare.jl complete — $out_png"
flush(stdout); flush(stderr)
