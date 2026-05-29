"""
Compute the surface ventilation diagnostic `calVdown` from the annual mean of
the 1-year re-run of a converged periodic-NK age solution.

For each (parent model, experiment, time window, model config) combination,
this script loads the 1-year periodic age `FieldTimeSeries` from
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/1year/{solver_tag}/age_periodic_1year.jld2
time-averages the surface layer over the first N−1 of the N half-monthly
snapshots (the last snapshot is the periodicity check — redundant with n=1),
and writes the raw (m³/m² = m) ventilation field to
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/ventilation.jld2

The `% v_tot / (10,000 km)²` normalisation from Pasquier *et al.* 2024 is
applied in the plot script `src/plot_ventilation.jl`, not here — the saved
JLD2 is unit-neutral. The total ocean volume `vtot` is saved so the plot
script doesn't have to recompute it.

Definition (Pasquier *et al.* 2024, doi:10.1029/2024JC021043; see also
docs/ventilation_figures.md):

  calVdown_raw(i, j) = V(i, j, Nz) * mean_n age(i, j, Nz, n) / (τ * A(i, j, Nz))
                     = Δz_top * mean_n age(i, j, Nz, n) / τ      [units: m]

with τ = 3·Δt (matches `setup_model.jl:347` — `relaxation_timescale = 3Δt`),
v_tot = Σ V(i, j, k) over all wet cells. The plot script applies the
`1e16 / vtot` prefactor (1e14 m² / (10,000 km)² × 100 % of v_tot).

The script writes a self-describing JLD2 with keys
  calVdown_raw, wet_surf, Az_surf, V_surf, age_surf,
  vtot, tau_seconds, n_avg, units, formula

This script handles both the forward (IAF) and adjoint (TRAF) legs
uniformly — the `_traf` suffix on `model_config` is appended automatically
by `env_defaults.sh` when `TRAF=yes`, and the directory layouts are
resolved accordingly.

Runs on CPU only; no GPU required.

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/compute_ventilation_diagnostic.jl")
```

Environment variables:
  PARENT_MODEL      – ACCESS-OM2-1 | ACCESS-OM2-025 | ACCESS-OM2-01
  EXPERIMENT        – source experiment (auto from PM if unset)
  TIME_WINDOW       – e.g. 1968-1977
  VELOCITY_SOURCE   – cgridtransports | totaltransport
  W_FORMULATION     – wdiagnosed | wprescribed
  ADVECTION_SCHEME  – centered2 | weno3 | weno5
  TIMESTEPPER       – AB2 | SRK2 | SRK3 | SRK4 | SRK5
  LINEAR_SOLVER     – Pardiso | ParU | UMFPACK (default: Pardiso)
  LUMP_AND_SPRAY    – no | AxB (default: no; e.g. 5x5)
  TRAF              – yes | no (default: no)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using JLD2
using Printf
using Statistics

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir, outputdir, Δt_seconds) =
    load_project_config()

model_config = require_env("MODEL_CONFIG")

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) ||
    error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

ls = parse_lump_and_spray()
LUMP_AND_SPRAY = ls.on
lumpspray_tag = ls.tag

# Path resolution. Two roots:
#   - periodic_root: parent of both NK and 1year subtrees
#   - fts_dir:       1year/{solver_tag}/  (data source for the diagnostic)
#   - nk_output_dir: NK[_QAxB]/           (where ventilation.jld2 is written;
#                                          stays in NK/ to match plot script's
#                                          existing layout & old comparisons)
# Both are searched against new (`{LINEAR_SOLVER}_{lumpspray_tag}`) and legacy
# (`{LINEAR_SOLVER}_LSprec`, `{LINEAR_SOLVER}_prec`) tag variants, so this
# script works on the in-flight historical naming.
px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

# Search both the gpu_tag root and the gpu_tag-less root: historically some
# partitioned (e.g. OM2-025 1x2) runs wrote to the bare `periodic/{MC}` path
# without a `{gpu_tag}` component, so fall back to it when present.
periodic_roots = unique(
    [
        isempty(gpu_tag) ?
            joinpath(outputdir, "periodic", model_config) :
            joinpath(outputdir, "periodic", model_config, gpu_tag),
        joinpath(outputdir, "periodic", model_config),
    ]
)

omega = parse_omega()
omega_suffix = omega.suffix
fts_basename = "age_periodic_1year$(omega_suffix).jld2"

# Locate the 1-year FTS (input). Try the new tag first, then legacy "LSprec"/"prec",
# across both candidate roots.
fts_candidates = unique(
    [
        joinpath(root, "1year", "$(LINEAR_SOLVER)_$(tag)", fts_basename)
            for root in periodic_roots
            for tag in (lumpspray_tag, "LSprec", "prec")
    ]
)
fts_hit = findfirst(isfile, fts_candidates)
fts_hit === nothing && error(
    "No 1-year periodic age FieldTimeSeries found. Tried:\n" *
        join(["  " * f for f in fts_candidates], "\n") *
        "\nRun the 1-year re-run step (`run1yrNK` in driver.sh) first.",
)
fts_file = fts_candidates[fts_hit]

# Resolve the periodic_root that actually held the FTS so the NK output lands
# in the same tree: <root>/1year/<solver_tag>/<basename> → root.
periodic_root = dirname(dirname(dirname(fts_file)))

# NK output dir — written next to NK/age_*.jld2 so plot scripts find it where
# they already look. Same dual-naming fallback as the input.
nk_candidates = [
    joinpath(periodic_root, "NK$(ls.dir_suffix)"),
    joinpath(periodic_root, "NK"),
]
nk_hit = findfirst(isdir, nk_candidates)
nk_output_dir = nk_hit === nothing ? nk_candidates[1] : nk_candidates[nk_hit]
mkpath(nk_output_dir)

ventilation_file = joinpath(nk_output_dir, "ventilation$(omega_suffix).jld2")

τ = 3 * Δt_seconds   # surface-sink relaxation timescale (s); matches setup_model.jl:347

@info "compute_ventilation_diagnostic.jl configuration"
@info "- PARENT_MODEL    = $parentmodel"
@info "- EXPERIMENT      = $experiment"
@info "- TIME_WINDOW     = $time_window"
@info "- model_config    = $model_config"
@info "- LINEAR_SOLVER   = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY  = $LUMP_AND_SPRAY (tag: $lumpspray_tag)"
@info "- OMEGA           = $(omega.tag) (suffix='$(omega_suffix)')"
@info "- FTS input       = $fts_file"
@info "- output          = $ventilation_file"
@info "- τ = 3·Δt        = $(τ) s (= $(τ / 3600) h)"
flush(stdout); flush(stderr)

################################################################################
# Load grid
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny, Nz = size(grid)
@info "Grid loaded: size = (Nx=$Nx, Ny=$Ny, Nz=$Nz)"
flush(stdout); flush(stderr)

################################################################################
# Load periodic 1-year age FieldTimeSeries and compute annual mean (surface)
################################################################################

@info "Loading age FieldTimeSeries from $fts_file"
flush(stdout); flush(stderr)

age_fts = FieldTimeSeries(fts_file, "age")
n_times = length(age_fts.times)
@info "Found $n_times output timesteps in 1-year FTS"
flush(stdout); flush(stderr)

# Wet mask (interior-sized; excludes the tripolar fold point in y)
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)
@info "Interior grid: (Nx′=$Nx′, Ny′=$Ny′, Nz′=$Nz′); wet cells = $Nidx"
flush(stdout); flush(stderr)

# Time-average over snapshots 1..(n_times-1) — skip the final snapshot, which
# is the periodicity check (redundant with n=1). Mirrors plot_periodic_1year_age.jl
# lines 159-165 but restricted to the surface layer (k = Nz′).
k_surf = Nz′
wet_surf = wet3D[:, :, k_surf]
n_avg = n_times - 1

@info "Time-averaging surface age over first $n_avg of $n_times snapshots"
flush(stdout); flush(stderr)

age_surf_accum = zeros(Float64, Nx′, Ny′)
for n in 1:n_avg
    age_n = interior(age_fts[n])           # (Nx′, Ny′, Nz′), seconds
    age_surf_n = @view age_n[:, :, k_surf]
    @. age_surf_accum += ifelse(wet_surf, age_surf_n, 0.0)
end
age_surf = similar(age_surf_accum)
@. age_surf = ifelse(wet_surf, age_surf_accum / n_avg, NaN)

################################################################################
# Surface cell metrics (V, A) and total wet-cell volume
################################################################################

@info "Computing cell volumes and total wet-cell volume"
flush(stdout); flush(stderr)

vol_3D = Array(interior(compute_volume(grid)))  # (Nx′, Ny′, Nz′), m³
@assert size(vol_3D) == size(wet3D) "compute_volume and wet3D shapes disagree"

V_surf = vol_3D[:, :, k_surf]              # (Nx′, Ny′), m³
vtot = sum(vol_3D[idx])                    # m³, sum over all wet cells

# Horizontal cell area at center: read from grid.jld2 (stored with halos around
# the full tripolar size; interior fields exclude the fold point in y).
Az_full = load(grid_file, "Azᶜᶜᵃ")
Hx = grid.underlying_grid.Hx
Hy = grid.underlying_grid.Hy
Az_surf = Az_full[Hx .+ (1:Nx′), Hy .+ (1:Ny′)]   # (Nx′, Ny′), m²
@assert size(Az_surf) == size(V_surf) "Az and V shapes disagree after trim"

Δz_top = V_surf[1, 1] / Az_surf[1, 1]      # report only
@info @sprintf("Surface Δz ≈ %.3f m (from V/A at (1,1))", Δz_top)
@info @sprintf("Total wet-cell volume v_tot = %.3e m³", vtot)

################################################################################
# Ventilation diagnostic — raw (m). The %v_tot / (10,000 km)² normalisation is
# applied in plot_ventilation.jl; here we just log it for sanity.
################################################################################

@info "Computing calVdown_raw = V_surf · ⟨age_surf⟩ / (τ · Az_surf)   [m]"
flush(stdout); flush(stderr)

calVdown_raw = (V_surf .* age_surf) ./ (τ .* Az_surf)   # m³ · s / s / m² = m
calVdown_raw[.!wet_surf] .= NaN

# Sanity check: finite where wet
n_wet = count(wet_surf)
n_finite_wet_raw = count(isfinite, calVdown_raw[wet_surf])
@info "Wet surface cells: $n_wet; finite calVdown_raw: $n_finite_wet_raw"
@assert n_finite_wet_raw == n_wet "Non-finite calVdown_raw at wet surface cells"

raw_vals = filter(isfinite, calVdown_raw)
raw_min, raw_max = extrema(raw_vals)
@info @sprintf(
    "calVdown_raw  [m]:                  min = %.3e   mean = %.3e   max = %.3e",
    raw_min, mean(raw_vals), raw_max
)

# Log the normalised form too (computed locally) so the level set is sane.
norm_factor = 1.0e16 / vtot
nrm_vals = raw_vals .* norm_factor
nrm_min, nrm_max = extrema(nrm_vals)
@info @sprintf(
    "calVdown_norm [%% v_tot / (10,000 km)²]:  min = %.3e   mean = %.3e   max = %.3e",
    nrm_min, mean(nrm_vals), nrm_max
)
@info @sprintf("Plot-script prefactor 1e16 / v_tot = %.3e (m⁻³)", norm_factor)
for q in (0.5, 0.9, 0.99, 0.999)
    @info @sprintf("calVdown_norm quantile q=%.3f → %.3e", q, quantile(nrm_vals, q))
end

################################################################################
# Seasonal surface-age means → seasonal calVdown_raw (DJF/MAM/JJA/SON)
#
# The 1-year run starts at t=0 (= Jan 1) and saves half-monthly snapshots, so
# snapshot n (over the averaged 1..n_avg) sits at age_fts.times[n] seconds from
# Jan 1. Map each to a calendar month (mid-month climatology, index 1 = January;
# see prep_velocities.jl:146) and bin into the four standard seasons.
################################################################################

@info "Computing seasonal surface-age means (DJF/MAM/JJA/SON)"
flush(stdout); flush(stderr)

year_s = 365.25 * 86400
month_s = year_s / 12
const SEASONS = (:DJF, :MAM, :JJA, :SON)
const SEASON_MONTHS = Dict(
    :DJF => (12, 1, 2), :MAM => (3, 4, 5), :JJA => (6, 7, 8), :SON => (9, 10, 11),
)
function season_of_time(t)
    midx = mod(floor(Int, t / month_s + 1.0e-9), 12) + 1   # 1..12
    for s in SEASONS
        midx in SEASON_MONTHS[s] && return s
    end
    return :DJF   # unreachable
end

age_surf_seasonal = Dict(s => zeros(Float64, Nx′, Ny′) for s in SEASONS)
season_counts = Dict(s => 0 for s in SEASONS)
for n in 1:n_avg
    s = season_of_time(age_fts.times[n])
    age_n = interior(age_fts[n])
    age_surf_n = @view age_n[:, :, k_surf]
    acc = age_surf_seasonal[s]
    @. acc += ifelse(wet_surf, age_surf_n, 0.0)
    season_counts[s] += 1
end

calVdown_raw_seasonal = Dict{Symbol, Matrix{Float64}}()
for s in SEASONS
    cnt = season_counts[s]
    cnt > 0 || @warn "No snapshots fell in season $s — seasonal map will be all-NaN"
    asurf = similar(age_surf_seasonal[s])
    @. asurf = ifelse(wet_surf, age_surf_seasonal[s] / max(cnt, 1), NaN)
    cv = (V_surf .* asurf) ./ (τ .* Az_surf)
    cv[.!wet_surf] .= NaN
    calVdown_raw_seasonal[s] = cv
    sv = filter(isfinite, cv) .* norm_factor
    @info @sprintf(
        "  %s: %d snapshots; calVdown_norm min=%.3e mean=%.3e max=%.3e",
        s, cnt, isempty(sv) ? NaN : minimum(sv),
        isempty(sv) ? NaN : mean(sv), isempty(sv) ? NaN : maximum(sv),
    )
end

################################################################################
# Save
################################################################################

@info "Saving ventilation field to $ventilation_file"
flush(stdout); flush(stderr)

jldsave(
    ventilation_file;
    calVdown_raw = calVdown_raw,
    calVdown_raw_DJF = calVdown_raw_seasonal[:DJF],
    calVdown_raw_MAM = calVdown_raw_seasonal[:MAM],
    calVdown_raw_JJA = calVdown_raw_seasonal[:JJA],
    calVdown_raw_SON = calVdown_raw_seasonal[:SON],
    season_counts = (;
        DJF = season_counts[:DJF], MAM = season_counts[:MAM],
        JJA = season_counts[:JJA], SON = season_counts[:SON],
    ),
    wet_surf = wet_surf,
    Az_surf = Az_surf,
    V_surf = V_surf,
    age_surf = age_surf,
    vtot = vtot,
    tau_seconds = τ,
    n_avg = n_avg,
    units = "m³/m² (= m); plot script normalises by 1e16 / vtot to obtain % v_tot / (10,000 km)²",
    formula = "calVdown_raw = V_surf .* mean_n(age_surf_n) ./ (tau .* Az_surf)",
)

@info "compute_ventilation_diagnostic.jl complete"
flush(stdout); flush(stderr)
