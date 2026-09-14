"""
Difference plots isolating the MELTWATER effect in the Li et al. (2023)
ACCESS-OM2-01 perturbation experiments (Task D of Li-etal_simulations.md).

Compares two periodic-NK age solutions on the SAME grid (both experiments share
the identical 0.1° tripolar grid — the perturbations are surface-forcing only):

  A = wthp  (wind + thermal only,          no meltwater)   → baseline
  B = wthmp (wind + thermal + meltwater)                   → with meltwater

The comparison primitives (`plot_age_comparison_slice/_zonal`,
`plot_age_profiles_basins`) render their diff panel as B − A, so
  B − A = wthmp − wthp = the meltwater contribution.

Works for BOTH the forward ideal age and the adjoint "time to re-emergence"
(TRAF) — select which by pointing MODEL_CONFIG at the forward tag or its `_traf`
variant (set TRAF=yes so env_defaults.sh appends `_traf`). The 1-year periodic
FTS produced by `run1yrNK`/`combine1yr` is the data source, located with the
same tag/partition fallback logic as `compute_ventilation_diagnostic.jl`.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=192GB -q hugemem -l walltime=02:00:00 -l ncpus=18 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
# forward ideal age:
MODEL_CONFIG=cgridtransports_wparent_upwind3_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS \\
  PARTITION=1x4 julia --project src/plot_Li_etal_meltwater_diff.jl
# adjoint age (time to re-emergence): add TRAF=yes and the _traf MODEL_CONFIG
```

Env vars:
  PARENT_MODEL   parent model tag                 (default ACCESS-OM2-01)
  EXPERIMENT_A   baseline experiment (no MW)       (default …_qian_wthp)
  EXPERIMENT_B   perturbed experiment (with MW)    (default …_qian_wthmp)
  TIME_WINDOW    real-year window                  (default 2040-2050)
  MODEL_CONFIG   solver config tag (forward or _traf)   (required)
  LINEAR_SOLVER  Pardiso | ParU | UMFPACK          (default Pardiso)
  LUMP_AND_SPRAY coarsening tag (e.g. 4x4)          (via parse_lump_and_spray)
  PARTITION[_X/_Y]  domain decomposition           (default from ENV; OM2-01 1x4)
  OMEGA          age-source depth restriction        (default all)
  TRAF           yes|no — only affects plot labels  (default no)
  FTS_VARNAME    variable name inside the FTS       (default age)
"""

@info "Loading packages for Li et al. meltwater difference plots"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using CairoMakie
using GeoMakie
using OceanBasins: oceanpolygons
using Statistics
using Printf

include("shared_functions.jl")
include(joinpath(@__DIR__, "shared_utils", "plotting_functions.jl"))

################################################################################
# Configuration
################################################################################

const repo_root = normpath(joinpath(@__DIR__, ".."))

const PM = get(ENV, "PARENT_MODEL", "ACCESS-OM2-01")
# A = baseline (wind+thermal only), B = with meltwater. Diff B − A = meltwater.
const EXPERIMENT_A = get(ENV, "EXPERIMENT_A", "01deg_jra55v13_ryf9091_qian_wthp")
const EXPERIMENT_B = get(ENV, "EXPERIMENT_B", "01deg_jra55v13_ryf9091_qian_wthmp")
const TW = get(ENV, "TIME_WINDOW", "2040-2050")
const MODEL_CONFIG = require_env("MODEL_CONFIG")

const LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) ||
    error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

const ls = parse_lump_and_spray()
const lumpspray_tag = ls.tag

const px = parse(Int, get(ENV, "PARTITION_X", "1"))
const py = parse(Int, get(ENV, "PARTITION_Y", "1"))
const gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

const omega = parse_omega()
const fts_basename = "age_periodic_1year$(omega.suffix).jld2"
const FTS_VARNAME = get(ENV, "FTS_VARNAME", "age")

const TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
const VALUE_LABEL = TRAF ? "TRAF age / time to re-emergence (years)" : "Age (years)"
const DIFF_LABEL = TRAF ? "Δ re-emergence (years)" : "Δ Age (years)"

const LABEL_A = "wthp (WT)"          # baseline: wind + thermal only
const LABEL_B = "wthmp (WT+MW)"      # with meltwater

const SECS_PER_YEAR = 365.25 * 86400
const DEPTHS_M = [100, 200, 500, 1000, 2000, 3000]

const OUT_ROOT = joinpath(
    repo_root, "outputs", "comparisons", "Li_etal_meltwater", MODEL_CONFIG,
)
mkpath(OUT_ROOT)

@info "plot_Li_etal_meltwater_diff.jl configuration" PM EXPERIMENT_A EXPERIMENT_B TW MODEL_CONFIG gpu_tag lumpspray_tag TRAF OUT_ROOT
flush(stdout); flush(stderr)

################################################################################
# Path resolution + field loading (mirrors compute_ventilation_diagnostic.jl)
################################################################################

outputdir_for(exp) = joinpath(repo_root, "outputs", PM, exp, TW)
grid_path_for(exp) = joinpath(repo_root, "preprocessed_inputs", PM, exp, "grid.jld2")

"""Locate the 1-year periodic age FTS for `exp`, trying the current lump/spray
tag then the legacy LSprec/prec tags, across the gpu_tag and bare roots."""
function resolve_fts(exp)
    outputdir = outputdir_for(exp)
    roots = unique(
        [
            isempty(gpu_tag) ?
                joinpath(outputdir, "periodic", MODEL_CONFIG) :
                joinpath(outputdir, "periodic", MODEL_CONFIG, gpu_tag),
            joinpath(outputdir, "periodic", MODEL_CONFIG),
        ]
    )
    candidates = unique(
        [
            joinpath(root, "1year", "$(LINEAR_SOLVER)_$(tag)", fts_basename)
                for root in roots
                for tag in (lumpspray_tag, "LSprec", "prec")
        ]
    )
    hit = findfirst(isfile, candidates)
    hit === nothing && error(
        "No 1-year periodic age FTS for $exp. Tried:\n" *
            join(["  " * f for f in candidates], "\n") *
            "\nRun `run1yrNK`/`combine1yr` for this experiment first.",
    )
    return candidates[hit]
end

"""Volume-weighted time-mean of the FTS interior, in years; dry cells → 0.
`check_age_field` errors on non-finite / out-of-range wet cells (a divergent
NK iterate), rather than silently masking them."""
function time_mean_years(fts, wet3D, grid; label = "")
    Nx, Ny, Nz = size(wet3D)
    Nt = length(fts.times)
    acc = zeros(Float64, Nx, Ny, Nz)
    for t in 1:Nt
        snap = Array(interior(fts[t]))
        @inbounds @. acc += ifelse(wet3D, Float64(snap) / SECS_PER_YEAR, 0.0)
    end
    acc ./= Nt
    check_age_field(acc, wet3D, grid; kind = "annual mean", min_yr = -1000.0, max_yr = 10_000.0, label)
    return acc
end

function load_mean_age(exp)
    fts_file = resolve_fts(exp)
    @info "Loading FTS for $exp" fts_file
    flush(stdout); flush(stderr)
    grid = load_tripolar_grid(grid_path_for(exp), CPU())
    wet3D = compute_wet_mask(grid).wet3D
    vol_3D = Array(interior(compute_volume(grid)))
    # InMemory(2): keep only 2 snapshots resident (was InMemory(), i.e. all 25
    # ≈ 160 GB per experiment at OM2-01). Changed after commit a93b4df; revert
    # here if snapshot access misbehaves.
    fts = FieldTimeSeries(fts_file, FTS_VARNAME; backend = InMemory(2))
    @info "$exp FTS: $(length(fts.times)) snapshots, interior $(size(interior(fts[1])))"
    m = time_mean_years(fts, wet3D, grid; label = exp)
    @info "$exp annual-mean age (yr)" min = minimum(m) max = maximum(m)
    return (; age_3D = m, grid, wet3D, vol_3D)
end

################################################################################
# Load both experiments (identical grids) and plot the A|B|(B−A) suite
################################################################################

A = load_mean_age(EXPERIMENT_A)
B = load_mean_age(EXPERIMENT_B)

size(A.wet3D) == size(B.wet3D) || error(
    "Grid size mismatch between $EXPERIMENT_A ($(size(A.wet3D))) and " *
        "$EXPERIMENT_B ($(size(B.wet3D))) — the two runs must share one grid.",
)

# Both experiments share one grid; use A's grid/wet/volume for the shared panels.
grid, wet3D, vol_3D = A.grid, A.wet3D, A.vol_3D
basins = compute_ocean_basin_masks(grid, wet3D)
Nx′, Ny′ = size(wet3D)[1:2]
bconf = [
    ("global", trues(Nx′, Ny′)),
    ("atlantic", basins.ATL),
    ("pacific", basins.PAC),
    ("indian", basins.IND),
]

@info "Rendering depth-slice difference panels"
for d in DEPTHS_M
    plot_age_comparison_slice(
        A.age_3D, B.age_3D, grid, wet3D, OUT_ROOT;
        label_A = LABEL_A, label_B = LABEL_B, depth_m = d,
        value_label = VALUE_LABEL, diff_label = DIFF_LABEL,
    )
end

@info "Rendering basin zonal-mean difference panels"
for (basin_label, mask) in bconf
    plot_age_comparison_zonal(
        A.age_3D, B.age_3D, grid, wet3D, vol_3D, mask, basin_label, OUT_ROOT;
        label_A = LABEL_A, label_B = LABEL_B,
        value_label = VALUE_LABEL, diff_label = DIFF_LABEL,
    )
end

@info "Rendering basin profile overlays"
plot_age_profiles_basins(
    [
        (; label = LABEL_A, age_3D = A.age_3D, grid, wet3D, vol_3D),
        (; label = LABEL_B, age_3D = B.age_3D, grid, wet3D, vol_3D),
    ],
    OUT_ROOT;
    value_label = VALUE_LABEL,
)

@info "plot_Li_etal_meltwater_diff.jl complete — outputs under $OUT_ROOT"
flush(stdout); flush(stderr)
