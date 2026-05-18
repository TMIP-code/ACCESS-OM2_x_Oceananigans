"""
Compare periodic NK age solutions across time windows and resolutions.

Implements docs/IAF_NK_age_comparison_plan.md:
  Phase 1 — Annual mean within each resolution (TW comparison)
  Phase 2 — Cross-resolution comparison (OM2-1 vs OM2-025 regridded to OM2-1)
  Phase 3 — Seasonality (max−min over the periodic cycle) re-run on all
            Phase 1/2 plot types
  Phase 3b — Peak-index (phase shift) maps (optional; not yet implemented)

Inputs: the four periodic 1-year FTS files produced by `run1yrNK`, at
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/1year/Pardiso_LSprec/age_periodic_1year.jld2
with MC = "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12".

Outputs go under outputs/comparisons/NK_age/{MC}/{phase…}/.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=02:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/compare_NK_ages.jl
```

Env vars (all optional):
  RUN_PHASE1   yes|no  (default yes)
  RUN_PHASE2   yes|no  (default yes)
  RUN_PHASE3   yes|no  (default yes)
  RUN_PHASE3B  yes|no  (default no)
  REGRID_DIRECTION  fine2coarse | coarse2fine  (default fine2coarse — OM2-025→OM2-1)
"""

@info "Loading packages for NK age comparison"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!

using CairoMakie
using ConservativeRegridding
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using Statistics
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

const MC = "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12"
const SOLVER_TAG = "Pardiso_LSprec"

const PM1, EXP1 = "ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6"
const PM025, EXP025 = "ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6"
const TW_A, TW_B = "1968-1977", "1999-2008"

const SECS_PER_YEAR = 365.25 * 86400
const DEPTHS_M = [100, 200, 500, 1000, 2000, 3000]

const RUN_PHASE1 = lowercase(get(ENV, "RUN_PHASE1", "yes")) == "yes"
const RUN_PHASE2 = lowercase(get(ENV, "RUN_PHASE2", "yes")) == "yes"
const RUN_PHASE3 = lowercase(get(ENV, "RUN_PHASE3", "yes")) == "yes"
const RUN_PHASE3B = lowercase(get(ENV, "RUN_PHASE3B", "no")) == "yes"

# Plan default: OM2-025 (fine) → OM2-1 (coarse), an honest diff at the scale
# both grids resolve.  Flip to coarse2fine to interpolate OM2-1 onto OM2-025.
const REGRID_FINE2COARSE = lowercase(get(ENV, "REGRID_DIRECTION", "fine2coarse")) == "fine2coarse"

const repo_root = normpath(joinpath(@__DIR__, ".."))
const OUT_ROOT = joinpath(repo_root, "outputs", "comparisons", "NK_age", MC)

@info "compare_NK_ages.jl configuration" RUN_PHASE1 RUN_PHASE2 RUN_PHASE3 RUN_PHASE3B REGRID_FINE2COARSE OUT_ROOT
mkpath(OUT_ROOT)
flush(stdout); flush(stderr)

################################################################################
# Paths and basic helpers
################################################################################

fts_path(PM, EXP, TW) = joinpath(
    repo_root, "outputs", PM, EXP, TW, "periodic", MC, "1year", SOLVER_TAG,
    "age_periodic_1year.jld2",
)
grid_path(PM, EXP) = joinpath(repo_root, "preprocessed_inputs", PM, EXP, "grid.jld2")

for path in (
        fts_path(PM1, EXP1, TW_A), fts_path(PM1, EXP1, TW_B),
        fts_path(PM025, EXP025, TW_A), fts_path(PM025, EXP025, TW_B),
        grid_path(PM1, EXP1), grid_path(PM025, EXP025),
    )
    isfile(path) || error("Required input not found: $path")
end

label_for(PM, TW) = PM == PM1 ? "OM2-1 $TW" : "OM2-025 $TW"

"""Volume-weighted time-mean of FTS interior, in years; dry cells set to 0."""
function time_mean_years(fts, wet3D)
    Nx, Ny, Nz = size(wet3D)
    Nt = length(fts.times)
    acc = zeros(Float64, Nx, Ny, Nz)
    for t in 1:Nt
        snap = Array(interior(fts[t]))
        @inbounds @. acc += ifelse(wet3D, Float64(snap) / SECS_PER_YEAR, 0.0)
    end
    acc ./= Nt
    return acc
end

"""Apply a single-layer regridder per k-level to a 3D array."""
function regrid_3d!(dst_3D, regridder, src_3D)
    Nz_src = size(src_3D, 3)
    Nz_dst = size(dst_3D, 3)
    @assert Nz_src == Nz_dst "src and dst must share Nz (got $Nz_src vs $Nz_dst)"
    fill!(dst_3D, 0.0)
    for k in 1:Nz_src
        ConservativeRegridding.regrid!(
            vec(view(dst_3D, :, :, k)),
            regridder,
            vec(view(src_3D, :, :, k)),
        )
    end
    return dst_3D
end

"""Per-layer conservation check, returns max relative error."""
function check_conservation(dst_3D, src_3D, regridder; rtol = 1.0e-7)
    Nz = size(src_3D, 3)
    max_rel = 0.0
    for k in 1:Nz
        s_dst = sum(vec(view(dst_3D, :, :, k)) .* regridder.dst_areas)
        s_src = sum(vec(view(src_3D, :, :, k)) .* regridder.src_areas)
        denom = max(abs(s_src), eps())
        rel = abs(s_dst - s_src) / denom
        rel > max_rel && (max_rel = rel)
    end
    @info "Conservation check: max relative error across all layers = $max_rel (tol $rtol)"
    max_rel ≤ rtol || @warn "Conservation tolerance exceeded" max_rel rtol
    return max_rel
end

################################################################################
# Load grids and per-grid derived quantities (volumes, wet masks, basin masks)
################################################################################

@info "Loading OM2-1 grid"
grid1 = load_tripolar_grid(grid_path(PM1, EXP1), CPU())
wet3D_1 = compute_wet_mask(grid1).wet3D
vol_3D_1 = Array(interior(compute_volume(grid1)))
basins_1 = compute_ocean_basin_masks(grid1, wet3D_1)
z1 = znodes(grid1, Center(), Center(), Center())

@info "Loading OM2-025 grid"
grid025 = load_tripolar_grid(grid_path(PM025, EXP025), CPU())
wet3D_025 = compute_wet_mask(grid025).wet3D
vol_3D_025 = Array(interior(compute_volume(grid025)))
basins_025 = compute_ocean_basin_masks(grid025, wet3D_025)
z025 = znodes(grid025, Center(), Center(), Center())

# Open-question #1: verify the vertical grids agree before overlaying profiles
# / using k→k regridding.
@info "Grid Nz check" Nz1 = length(z1) Nz025 = length(z025)
@assert length(z1) == length(z025) "OM2-1 and OM2-025 have different Nz — k→k regridding invalid"
if !isapprox(z1, z025; rtol = 1.0e-3)
    @warn "z-centers differ between OM2-1 and OM2-025; depth-overlaid profiles may be misaligned" z1 z025
end

# Basin configurations: (key in vertical-mask form, label, mask)
function basin_configs(wet3D, basins)
    Nx′, Ny′ = size(wet3D)[1:2]
    return [
        ("global", trues(Nx′, Ny′)),
        ("atlantic", basins.ATL),
        ("pacific", basins.PAC),
        ("indian", basins.IND),
    ]
end

bconf_1 = basin_configs(wet3D_1, basins_1)
bconf_025 = basin_configs(wet3D_025, basins_025)

################################################################################
# Generic A vs B plot block (used for both annual mean and seasonal range)
################################################################################

"""Run the full slice + zonal + profile suite for an A/B comparison on a single grid."""
function plot_comparison_same_grid(
        A_3D, B_3D, grid, wet3D, vol_3D, bconf, out_dir;
        label_A, label_B, value_label, diff_label,
        slice_colorrange = :auto,
    )
    mkpath(out_dir)
    for d in DEPTHS_M
        plot_age_comparison_slice(
            A_3D, B_3D, grid, wet3D, out_dir;
            label_A, label_B, depth_m = d,
            value_label, diff_label,
            colorrange = slice_colorrange,
        )
    end
    for (basin_label, mask) in bconf
        plot_age_comparison_zonal(
            A_3D, B_3D, grid, wet3D, vol_3D, mask, basin_label, out_dir;
            label_A, label_B, value_label, diff_label,
        )
    end
    plot_age_profiles_basins(
        [
            (; label = label_A, age_3D = A_3D, grid, wet3D, vol_3D),
            (; label = label_B, age_3D = B_3D, grid, wet3D, vol_3D),
        ],
        out_dir;
        value_label,
    )
    return nothing
end

"""
Phase 2 / cross-resolution variant: A and B share a destination grid for
slices and zonal averages, but profiles overlay each pipeline on its own
native (grid, vol, wet3D).
"""
function plot_comparison_cross_resolution(
        A_3D, B_3D, dst_grid, dst_wet3D, dst_vol_3D, dst_bconf, out_dir;
        label_A, label_B, value_label, diff_label,
        profile_pipelines,  # Vector of NamedTuples for native-resolution overlays
    )
    mkpath(out_dir)
    for d in DEPTHS_M
        plot_age_comparison_slice(
            A_3D, B_3D, dst_grid, dst_wet3D, out_dir;
            label_A, label_B, depth_m = d, value_label, diff_label,
        )
    end
    for (basin_label, mask) in dst_bconf
        plot_age_comparison_zonal(
            A_3D, B_3D, dst_grid, dst_wet3D, dst_vol_3D, mask, basin_label, out_dir;
            label_A, label_B, value_label, diff_label,
        )
    end
    plot_age_profiles_basins(profile_pipelines, out_dir; value_label)
    return nothing
end

################################################################################
# Lazy fetch: compute (and cache in this session) per-pipeline fields
################################################################################

# Keys: ("1" or "025", TW), value = 3D Array{Float64,3} in years
const _means = Dict{Tuple{String, String}, Array{Float64, 3}}()
const _sranges = Dict{Tuple{String, String}, Array{Float64, 3}}()

function get_time_mean(res::String, TW::String)
    haskey(_means, (res, TW)) && return _means[(res, TW)]
    PM, EXP, wet3D, backend = if res == "1"
        (PM1, EXP1, wet3D_1, InMemory())
    else
        (PM025, EXP025, wet3D_025, OnDisk())
    end
    path = fts_path(PM, EXP, TW)
    @info "Loading FTS + computing annual mean: $res / $TW" path
    flush(stdout); flush(stderr)
    fts = FieldTimeSeries(path, "age"; backend)
    Nt = length(fts.times)
    @info "FTS has $Nt snapshots"
    m = time_mean_years(fts, wet3D)
    _means[(res, TW)] = m
    return m
end

function get_seasonal_range(res::String, TW::String)
    haskey(_sranges, (res, TW)) && return _sranges[(res, TW)]
    PM, EXP, wet3D, backend = if res == "1"
        (PM1, EXP1, wet3D_1, InMemory())
    else
        (PM025, EXP025, wet3D_025, OnDisk())
    end
    path = fts_path(PM, EXP, TW)
    @info "Loading FTS + computing seasonal range: $res / $TW" path
    flush(stdout); flush(stderr)
    fts = FieldTimeSeries(path, "age"; backend)
    Nt = length(fts.times)
    @info "FTS has $Nt snapshots"
    sr = seasonal_range(fts; wet3D)
    _sranges[(res, TW)] = sr
    return sr
end

################################################################################
# Regridder — built once, reused across all OM2-025 → OM2-1 calls
################################################################################

const _regridder = Ref{Any}(nothing)

function get_regridder()
    _regridder[] === nothing || return _regridder[]
    if REGRID_FINE2COARSE
        # dst = OM2-1 (coarse), src = OM2-025 (fine)
        dst_field = CenterField(grid1)
        src_field = CenterField(grid025)
    else
        dst_field = CenterField(grid025)
        src_field = CenterField(grid1)
    end
    @info "Building Regridder (this is the slow step)" REGRID_FINE2COARSE
    flush(stdout); flush(stderr)
    t0 = time()
    r = try
        ConservativeRegridding.Regridder(dst_field, src_field; progress = true)
    catch e
        @warn "Regridder build failed on ImmersedBoundaryGrid fields — retrying on underlying grids" exception = e
        ConservativeRegridding.Regridder(
            CenterField(dst_field.grid.underlying_grid),
            CenterField(src_field.grid.underlying_grid);
            progress = true,
        )
    end
    @info "Regridder built in $(round(time() - t0; digits = 1)) s"
    _regridder[] = r
    return r
end

"""Regrid a native-OM2-025 field onto the OM2-1 grid (or the reverse if the
direction flag is flipped)."""
function regrid_to_dst(src_3D)
    regridder = get_regridder()
    if REGRID_FINE2COARSE
        @assert size(src_3D)[1:2] == size(wet3D_025)[1:2] "expected OM2-025 src horizontal dims"
        dst_3D = zeros(Float64, size(wet3D_1)...)
    else
        @assert size(src_3D)[1:2] == size(wet3D_1)[1:2] "expected OM2-1 src horizontal dims"
        dst_3D = zeros(Float64, size(wet3D_025)...)
    end
    regrid_3d!(dst_3D, regridder, src_3D)
    check_conservation(dst_3D, src_3D, regridder)
    return dst_3D
end

################################################################################
# Phase 1 — Annual-mean A vs B within each resolution
################################################################################

if RUN_PHASE1
    @info "=== Phase 1: annual-mean TW comparison ==="
    flush(stdout); flush(stderr)

    # OM2-1
    A1 = get_time_mean("1", TW_A)
    B1 = get_time_mean("1", TW_B)
    plot_comparison_same_grid(
        A1, B1, grid1, wet3D_1, vol_3D_1, bconf_1,
        joinpath(OUT_ROOT, "phase1_tw_OM2-1");
        label_A = label_for(PM1, TW_A),
        label_B = label_for(PM1, TW_B),
        value_label = "Age (years)",
        diff_label = "Δ Age (years)",
    )

    # OM2-025
    A025 = get_time_mean("025", TW_A)
    B025 = get_time_mean("025", TW_B)
    plot_comparison_same_grid(
        A025, B025, grid025, wet3D_025, vol_3D_025, bconf_025,
        joinpath(OUT_ROOT, "phase1_tw_OM2-025");
        label_A = label_for(PM025, TW_A),
        label_B = label_for(PM025, TW_B),
        value_label = "Age (years)",
        diff_label = "Δ Age (years)",
    )
end

################################################################################
# Phase 2 — Cross-resolution annual-mean (OM2-1 vs OM2-025-regridded)
################################################################################

if RUN_PHASE2
    @info "=== Phase 2: cross-resolution comparison ==="
    flush(stdout); flush(stderr)

    for TW in (TW_A, TW_B)
        @info "Phase 2 / TW = $TW"
        flush(stdout); flush(stderr)

        m1 = get_time_mean("1", TW)
        m025 = get_time_mean("025", TW)

        if REGRID_FINE2COARSE
            A_3D = m1
            B_3D_native = m025
            B_3D = regrid_to_dst(m025)
            dst_grid, dst_wet3D, dst_vol_3D, dst_bconf = grid1, wet3D_1, vol_3D_1, bconf_1
            label_A = label_for(PM1, TW)
            label_B = "$(label_for(PM025, TW)) → OM2-1"
        else
            A_3D = regrid_to_dst(m1)
            B_3D_native = m025
            B_3D = m025
            dst_grid, dst_wet3D, dst_vol_3D, dst_bconf = grid025, wet3D_025, vol_3D_025, bconf_025
            label_A = "$(label_for(PM1, TW)) → OM2-025"
            label_B = label_for(PM025, TW)
        end

        # Profiles: overlay native fields on their own grids
        profile_pipelines = [
            (; label = label_for(PM1, TW), age_3D = m1, grid = grid1, wet3D = wet3D_1, vol_3D = vol_3D_1),
            (; label = label_for(PM025, TW), age_3D = m025, grid = grid025, wet3D = wet3D_025, vol_3D = vol_3D_025),
        ]

        plot_comparison_cross_resolution(
            A_3D, B_3D, dst_grid, dst_wet3D, dst_vol_3D, dst_bconf,
            joinpath(OUT_ROOT, "phase2_resolution_$TW");
            label_A, label_B,
            value_label = "Age (years)",
            diff_label = "Δ Age (years)",
            profile_pipelines,
        )
    end
end

################################################################################
# Phase 3 — Seasonality (re-run Phase 1 + 2 with seasonal_range fields)
################################################################################

if RUN_PHASE3
    @info "=== Phase 3: seasonality (max−min over cycle) ==="
    flush(stdout); flush(stderr)

    # Phase 3 + Phase 1: TW comparison of seasonal range, per resolution
    sA1 = get_seasonal_range("1", TW_A)
    sB1 = get_seasonal_range("1", TW_B)
    plot_comparison_same_grid(
        sA1, sB1, grid1, wet3D_1, vol_3D_1, bconf_1,
        joinpath(OUT_ROOT, "phase3_seasonality", "tw_OM2-1");
        label_A = "$(label_for(PM1, TW_A)) seasonal range",
        label_B = "$(label_for(PM1, TW_B)) seasonal range",
        value_label = "Seasonal range (years)",
        diff_label = "Δ range (years)",
    )

    sA025 = get_seasonal_range("025", TW_A)
    sB025 = get_seasonal_range("025", TW_B)
    plot_comparison_same_grid(
        sA025, sB025, grid025, wet3D_025, vol_3D_025, bconf_025,
        joinpath(OUT_ROOT, "phase3_seasonality", "tw_OM2-025");
        label_A = "$(label_for(PM025, TW_A)) seasonal range",
        label_B = "$(label_for(PM025, TW_B)) seasonal range",
        value_label = "Seasonal range (years)",
        diff_label = "Δ range (years)",
    )

    # Phase 3 + Phase 2: cross-resolution seasonal-range comparison per TW
    for TW in (TW_A, TW_B)
        @info "Phase 3 / cross-resolution TW = $TW"
        flush(stdout); flush(stderr)

        s1 = get_seasonal_range("1", TW)
        s025 = get_seasonal_range("025", TW)

        if REGRID_FINE2COARSE
            A_3D = s1
            B_3D = regrid_to_dst(s025)
            dst_grid, dst_wet3D, dst_vol_3D, dst_bconf = grid1, wet3D_1, vol_3D_1, bconf_1
            label_A = "$(label_for(PM1, TW)) seasonal range"
            label_B = "$(label_for(PM025, TW)) seasonal range → OM2-1"
        else
            A_3D = regrid_to_dst(s1)
            B_3D = s025
            dst_grid, dst_wet3D, dst_vol_3D, dst_bconf = grid025, wet3D_025, vol_3D_025, bconf_025
            label_A = "$(label_for(PM1, TW)) seasonal range → OM2-025"
            label_B = "$(label_for(PM025, TW)) seasonal range"
        end

        profile_pipelines = [
            (; label = "$(label_for(PM1, TW)) seas. range", age_3D = s1, grid = grid1, wet3D = wet3D_1, vol_3D = vol_3D_1),
            (; label = "$(label_for(PM025, TW)) seas. range", age_3D = s025, grid = grid025, wet3D = wet3D_025, vol_3D = vol_3D_025),
        ]

        plot_comparison_cross_resolution(
            A_3D, B_3D, dst_grid, dst_wet3D, dst_vol_3D, dst_bconf,
            joinpath(OUT_ROOT, "phase3_seasonality", "resolution_$TW");
            label_A, label_B,
            value_label = "Seasonal range (years)",
            diff_label = "Δ range (years)",
            profile_pipelines,
        )
    end
end

################################################################################
# Phase 3b — Phase shift (peak-month maps) — not yet implemented
################################################################################

if RUN_PHASE3B
    @warn "Phase 3b (peak-snapshot maps) is not yet implemented; skipping. " *
        "Add a `peak_index` streaming helper analogous to `seasonal_range` " *
        "before enabling RUN_PHASE3B=yes."
end

@info "compare_NK_ages.jl complete — outputs under $OUT_ROOT"
flush(stdout); flush(stderr)
