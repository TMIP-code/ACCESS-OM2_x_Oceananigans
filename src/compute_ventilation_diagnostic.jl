"""
Compute the surface ventilation diagnostic `calVdown` from a converged
periodic-NK age solution.

For each (parent model, experiment, time window, model config) combination,
this script loads the converged 3D age field from
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/age_{LINEAR_SOLVER}_{tag}.jld2
and writes the 2D surface ventilation field to
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/ventilation.jld2

Definition (Pasquier *et al.* 2024, doi:10.1029/2024JC021043; see also
docs/TRAF_simulations.md):

  calVdown(i, j) = V(i, j, Nz) * age(i, j, Nz) / (τ * A(i, j))
                 = Δz_top * age(i, j, Nz) / τ

where Nz is the top (surface) k-index, V the cell volume, A the horizontal
cell area, and τ = 3·Δt the same surface-sink relaxation timescale used
inside the forward integrator. Units: metres.

This script handles both the forward (IAF) and adjoint (TRAF) legs
uniformly — the `_traf` suffix on `model_config` is appended automatically
by `build_model_config` when `TRAF=yes`, and the NK output directory is
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

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir, outputdir, Δt_seconds) =
    load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) ||
    error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

ls = parse_lump_and_spray()
LUMP_AND_SPRAY = ls.on
lumpspray_tag = ls.tag

# Match solve_periodic_NK.jl's output directory layout: serial → periodic/{MC}/NK[_QAxB],
# distributed → periodic/{MC}/{px}x{py}/NK[_QAxB]. Pre-refactor runs saved under
# plain `NK/` with `age_<solver>_LSprec.jld2`; we accept either.
px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

periodic_root = isempty(gpu_tag) ?
    joinpath(outputdir, "periodic", model_config) :
    joinpath(outputdir, "periodic", model_config, gpu_tag)

candidates = [
    (joinpath(periodic_root, "NK$(ls.dir_suffix)"), "age_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2"),
    (joinpath(periodic_root, "NK"), "age_$(LINEAR_SOLVER)_LSprec.jld2"),
    (joinpath(periodic_root, "NK"), "age_$(LINEAR_SOLVER)_prec.jld2"),
]
hit = findfirst(((d, f),) -> isfile(joinpath(d, f)), candidates)
hit === nothing && error(
    "No converged NK age file found. Tried:\n" *
        join(["  " * joinpath(d, f) for (d, f) in candidates], "\n"),
)
nk_output_dir, nk_filename = candidates[hit]
nk_file = joinpath(nk_output_dir, nk_filename)

ventilation_file = joinpath(nk_output_dir, "ventilation.jld2")

τ = 3 * Δt_seconds   # surface-sink relaxation timescale (s); matches setup_model.jl:351

@info "compute_ventilation_diagnostic.jl configuration"
@info "- PARENT_MODEL    = $parentmodel"
@info "- EXPERIMENT      = $experiment"
@info "- TIME_WINDOW     = $time_window"
@info "- model_config    = $model_config"
@info "- LINEAR_SOLVER   = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY  = $LUMP_AND_SPRAY (tag: $lumpspray_tag)"
@info "- NK input        = $nk_file"
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
# Load converged NK age
################################################################################

@info "Loading converged periodic NK age from $nk_file"
flush(stdout); flush(stderr)
nk_data = load(nk_file)
age_3D = nk_data["age"]                    # (Nx′, Ny′, Nz′), units: seconds (per solve_periodic_NK.jl)
wet3D = nk_data["wet3D"]
Nx′, Ny′, Nz′ = size(age_3D)
@info "Loaded age: size = $(size(age_3D)), wet cells = $(count(wet3D))"
@assert (Nx′, Ny′, Nz′) == size(wet3D) "age and wet3D shapes disagree"
flush(stdout); flush(stderr)

################################################################################
# Compute surface cell metrics
################################################################################

@info "Computing surface cell volumes"
flush(stdout); flush(stderr)

vol_3D = interior(compute_volume(grid))    # (Nx′, Ny′, Nz′), m³
@assert size(vol_3D) == size(age_3D) "volume and age shapes disagree"

# Top (surface) layer is k = Nz′ in Oceananigans (k=1 is bottom).
k_surf = Nz′
V_surf = vol_3D[:, :, k_surf]              # (Nx′, Ny′), m³

# Horizontal cell area at center: read from grid.jld2 (saved with halos around
# the full tripolar size; interior fields exclude the fold point in y, so we
# slice to match age_3D's (Nx′, Ny′) shape).
Az_full = load(grid_file, "Azᶜᶜᵃ")         # (Nx + 2Hx, Ny_full + 2Hy)
Hx = grid.underlying_grid.Hx
Hy = grid.underlying_grid.Hy
Az_surf = Az_full[Hx .+ (1:Nx′), Hy .+ (1:Ny′)]  # (Nx′, Ny′), m²
@assert size(Az_surf) == size(V_surf) "Az and V shapes disagree after trim"

Δz_top = V_surf[1, 1] / Az_surf[1, 1]      # report only; calVdown uses V/A elementwise
@info "Surface Δz ≈ $(Δz_top) m (from V/A at (1,1))"

################################################################################
# Ventilation diagnostic
################################################################################

@info "Computing calVdown = V · age / τ / A (units: m)"
flush(stdout); flush(stderr)

age_surf = age_3D[:, :, k_surf]            # (Nx′, Ny′), s
calVdown = (V_surf .* age_surf) ./ (τ .* Az_surf)  # m³ · s / s / m² = m

# Mask dry surface cells with NaN
wet_surf = wet3D[:, :, k_surf]
calVdown[.!wet_surf] .= NaN

# Sanity check: finite where wet
n_wet = count(wet_surf)
n_finite_wet = count(isfinite, calVdown[wet_surf])
@info "Wet surface cells: $n_wet; finite calVdown among them: $n_finite_wet"
@assert n_finite_wet == n_wet "Non-finite calVdown at wet surface cells"

cv_min, cv_max = extrema(filter(isfinite, calVdown))
cv_mean = sum(x for x in calVdown if isfinite(x)) / n_finite_wet
@info @sprintf(
    "calVdown stats over wet surface: min = %.3e m, mean = %.3e m, max = %.3e m",
    cv_min, cv_mean, cv_max
)

################################################################################
# Save
################################################################################

@info "Saving ventilation field to $ventilation_file"
flush(stdout); flush(stderr)

jldsave(
    ventilation_file;
    calVdown = calVdown,
    wet_surf = wet_surf,
    Az_surf = Az_surf,
    V_surf = V_surf,
    age_surf = age_surf,
    tau_seconds = τ,
    units = "metres",
    formula = "V_surf .* age_surf ./ (tau .* Az_surf)",
)

@info "compute_ventilation_diagnostic.jl complete"
flush(stdout); flush(stderr)
