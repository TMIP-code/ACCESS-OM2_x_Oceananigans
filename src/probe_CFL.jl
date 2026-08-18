"""
Read-only diagnostic: compute the advective CFL number of the *prescribed input
velocities* — no model run required.

Because this is an offline passive-tracer simulation, the velocities that advect
the tracer are exactly the 12 monthly `FieldTimeSeries` snapshots on disk. The
CFL of those snapshots therefore *is* the CFL the model will see, so it can be
measured up front instead of discovering a blow-up mid-run. With `Cyclical`
linear-in-time interpolation the intermediate states are convex combinations of
neighbouring snapshots, and |αU₁ + (1-α)U₂| ≤ max(|U₁|, |U₂|) cell-wise, so the
worst CFL over the 12 snapshots bounds the worst CFL over the whole year.

Uses Oceananigans' own `Advection.cell_advection_timescale(grid, velocities)`
(the function behind `Diagnostics.AdvectiveCFL` and `TimeStepWizard`):

    τ(i,j,k) = 1 / (|u|/Δxᶠᶜᶜ + |v|/Δyᶜᶠᶜ + |w|/Δzᶜᶜᶠ),   CFL = Δt / min τ

and additionally reports the per-direction breakdown (which of x/y/z sets the
limit) and the location of the worst cell.

Reads one snapshot at a time (OnDisk backend) so it fits in modest RAM despite
the large per-field files. CPU only, no model run.

Env (beyond the usual PARENT_MODEL / EXPERIMENT / TIME_WINDOW / TIMESTEP_MULT /
VELOCITY_SOURCE / W_FORMULATION / KAPPA_* read via load_project_config):
  CFL_TARGET  – target advective CFL for the Δt recommendation (default 0.7)
  CFL_LOCATE  – yes | no — materialise the CFL field to locate the worst cell
                (default yes; costs one extra 3D Float64 field of memory)
"""

@info "Loading packages for CFL probe"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Diagnostics: cell_diffusion_timescale
using Oceananigans.Grids: znodes, λnodes, φnodes
using Oceananigans.Operators: Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ
using Printf

include("shared_functions.jl")

################################################################################
# Kernel functions
################################################################################

# Guarded single-direction inverse timescale, |U| / Δ. In fully-immersed cells
# PartialCellBottom can give Δ = 0 → Δ⁻¹ = Inf, and 0 * Inf = NaN, which would
# poison the reduction; the isfinite guard makes those cells "no constraint".
# Genuine NaN/Inf in the *data* still propagates — that is a real problem and
# must not be hidden (see probe_velocity_extremes.jl).
@inline function guarded_inverse_timescale(U, i, j, k, Δ⁻¹)
    @inbounds a = abs(U[i, j, k])
    return ifelse(isfinite(Δ⁻¹), a * Δ⁻¹, zero(a))
end

@inline invτxᶜᶜᶜ(i, j, k, grid, u) = guarded_inverse_timescale(u, i, j, k, Δx⁻¹ᶠᶜᶜ(i, j, k, grid))
@inline invτyᶜᶜᶜ(i, j, k, grid, v) = guarded_inverse_timescale(v, i, j, k, Δy⁻¹ᶜᶠᶜ(i, j, k, grid))
@inline invτzᶜᶜᶜ(i, j, k, grid, w) = guarded_inverse_timescale(w, i, j, k, Δz⁻¹ᶜᶜᶠ(i, j, k, grid))

@inline invτᶜᶜᶜ(i, j, k, grid, u, v, w) =
    invτxᶜᶜᶜ(i, j, k, grid, u) + invτyᶜᶜᶜ(i, j, k, grid, v) + invτzᶜᶜᶜ(i, j, k, grid, w)

kfo(func, grid, args...) = KernelFunctionOperation{Center, Center, Center}(func, grid, args...)

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment, time_window, experiment_dir, monthly_dir, Δt_seconds) =
    load_project_config()

Δt = Δt_seconds
CFL_TARGET = parse(Float64, get(ENV, "CFL_TARGET", "0.7"))
CFL_TARGET > 0 || error("CFL_TARGET must be positive (got $CFL_TARGET)")
CFL_LOCATE = get(ENV, "CFL_LOCATE", "yes") == "yes"

VELOCITY_SOURCE = require_env("VELOCITY_SOURCE")
W_FORMULATION = require_env("W_FORMULATION")

@info "Probing CFL" parentmodel experiment time_window Δt CFL_TARGET CFL_LOCATE
flush(stdout); flush(stderr)

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
zC = collect(znodes(grid, Center(), Center(), Center()))

latlon = try
    (;
        λ = collect(λnodes(grid, Center(), Center(), Center())),
        φ = collect(φnodes(grid, Center(), Center(), Center())),
    )
catch
    nothing
end
coordstr(i, j) =
    latlon === nothing ? "" :
    @sprintf(" lon=%.2f lat=%.2f", latlon.λ[i, j], latlon.φ[i, j])

################################################################################
# Input files — mirror setup_model.jl so we probe what the model will read
################################################################################

vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
u_file = joinpath(monthly_dir, "u_from_$(vs_prefix)_monthly.jld2")
v_file = joinpath(monthly_dir, "v_from_$(vs_prefix)_monthly.jld2")

# w: mirror setup_model.jl's choice so we probe the w the model will actually use.
#   W_FORMULATION=wprescribed → the file it reads (parent or diagnosed)
#   W_FORMULATION=wdiagnosed  → computed online from continuity; the saved
#                               diagnosed-w file is that same field, so use it.
w_diag_suffix = VELOCITY_SOURCE == "totaltransport" ?
    "w_diagnosed_totaltransport_monthly" : "w_diagnosed_monthly"
w_diag_file = joinpath(monthly_dir, "$(w_diag_suffix).jld2")
w_parent_file = joinpath(monthly_dir, "w_from_$(vs_prefix)_monthly.jld2")

w_file, w_kind = if W_FORMULATION == "wprescribed"
    src = require_env("PRESCRIBED_W_SOURCE")
    src == "diagnosed" ? (w_diag_file, "prescribed, diagnosed") :
        src == "parent" ? (w_parent_file, "prescribed, parent model") :
        error("PRESCRIBED_W_SOURCE must be diagnosed or parent (got: $src)")
else
    (w_diag_file, "diagnosed online (probing the saved diagnose_w output)")
end
isfile(w_file) || error(
    "Missing w input: $w_file — run the `diagnose_w` (or `vel`) pipeline step first."
)

for f in (u_file, v_file)
    isfile(f) || error("Missing input velocity file: $f")
end

@info """Probing velocities (W_FORMULATION=$W_FORMULATION, VELOCITY_SOURCE=$VELOCITY_SOURCE):
- u: $u_file
- v: $v_file
- w: $w_file  [$w_kind]
"""
flush(stdout); flush(stderr)

u_fts = FieldTimeSeries(u_file, "u"; architecture = CPU(), grid, backend = OnDisk())
v_fts = FieldTimeSeries(v_file, "v"; architecture = CPU(), grid, backend = OnDisk())
w_fts = FieldTimeSeries(w_file, "w"; architecture = CPU(), grid, backend = OnDisk())

Nt = length(u_fts.times)
length(v_fts.times) == Nt || error(
    "u/v snapshot-count mismatch: u=$Nt, v=$(length(v_fts.times))"
)
u_fts.times ≈ v_fts.times || error("u and v FieldTimeSeries have different times")

# The diagnosed-w writer saves at *half*-monthly intervals (25 snapshots per
# year), so w is on a finer time grid than the 12 monthly u/v snapshots. Pair
# each u/v month with the w snapshot at the same time rather than by index.
month_spacing = Nt > 1 ? u_fts.times[2] - u_fts.times[1] : Inf
w_index = map(u_fts.times) do t
    gap, n = findmin(abs.(w_fts.times .- t))
    gap ≤ 1.0e-3 * month_spacing || error(
        "No w snapshot matches u/v time $t s (closest is $(w_fts.times[n]) s, " *
            "gap $gap s > 0.1% of the $(month_spacing) s month spacing). " *
            "Rebuild w with the `diagnose_w` step for this time window."
    )
    return n
end
@info "Snapshots: $Nt monthly u/v, $(length(w_fts.times)) w " *
    "→ w indices $(w_index) paired by time"

################################################################################
# Per-month CFL
################################################################################

"""
    probe_month(t, tw, grid, u_fts, v_fts, w_fts, Δt, CFL_LOCATE, zC, coordstr) -> NamedTuple

CFL diagnostics for u/v snapshot `t` paired with w snapshot `tw`. Wrapped in a
function (rather than inlined in a top-level loop) so every binding is a local —
no soft-scope surprises.
"""
function probe_month(t, tw, grid, u_fts, v_fts, w_fts, Δt, CFL_LOCATE, zC, coordstr, acc)
    u = u_fts[t]
    v = v_fts[t]
    w = w_fts[tw]

    # Library answer: exactly what AdvectiveCFL / TimeStepWizard would report.
    τ = cell_advection_timescale(grid, (u, v, w))
    cfl = Δt / τ

    # Per-direction breakdown (guarded; zero-Δ cells contribute 0).
    Cx = Δt * maximum(kfo(invτxᶜᶜᶜ, grid, u))
    Cy = Δt * maximum(kfo(invτyᶜᶜᶜ, grid, v))
    Cz = Δt * maximum(kfo(invτzᶜᶜᶜ, grid, w))

    @info @sprintf(
        "  month %2d: CFL = %.4g   (Cx=%.4g  Cy=%.4g  Cz=%.4g)   min τ = %.4g s = %.3g h",
        t, cfl, Cx, Cy, Cz, τ, τ / 3600,
    )
    isfinite(cfl) || @warn "Non-finite CFL for month $t — run probe_velocity_extremes.jl " *
        "to check the inputs for NaN/Inf."

    loc = nothing
    if CFL_LOCATE
        C = compute!(Field(kfo(invτᶜᶜᶜ, grid, u, v, w)))
        # Fold straight into the running per-cell max in `acc` instead of keeping
        # all 12 snapshots — one Float64 field is ~5.8 GB at OM2-01, so retaining
        # the whole set would cost ~70 GB for no reason.
        Ci = interior(C)
        Ci .= ifelse.(isfinite.(Ci), Ci, zero(eltype(Ci)))
        isempty(acc) ? push!(acc, collect(Ci)) : (acc[1] .= max.(acc[1], Ci))
        mx, ci = findmax(Ci)
        i, j, k = Tuple(ci)
        loc = (; cfl = Δt * mx, month = t, i, j, k, u = u[i, j, k], v = v[i, j, k], w = w[i, j, k])
        @info @sprintf(
            "            worst cell (i=%d,j=%d,k=%d) depth=%.1fm%s  CFL=%.4g  u=%+.4g v=%+.4g w=%+.4g m/s",
            i, j, k, zC[min(k, length(zC))], coordstr(i, j), loc.cfl, loc.u, loc.v, loc.w,
        )
    end
    flush(stdout); flush(stderr)

    return (; cfl, Cx, Cy, Cz, τ, loc)
end

# `acc` collects the per-cell max over months of the inverse advection timescale
# (empty when CFL_LOCATE is off).
acc = Array{Float64, 3}[]
results = [
    probe_month(t, w_index[t], grid, u_fts, v_fts, w_fts, Δt, CFL_LOCATE, zC, coordstr, acc)
        for t in 1:Nt
]

################################################################################
# Summary and Δt recommendation
################################################################################

cfl_max = maximum(r.cfl for r in results)
month_max = argmax([r.cfl for r in results])
τ_min = Δt / cfl_max

@info "──────── summary ────────"
@info @sprintf("Δt                = %.6g s (%.4g h)  [includes TIMESTEP_MULT]", Δt, Δt / 3600)
@info @sprintf("max advective CFL = %.5g   (month %d)", cfl_max, month_max)
@info @sprintf("min advection timescale = %.6g s (%.4g h)", τ_min, τ_min / 3600)
@info @sprintf(
    "direction maxima over all months: Cx=%.4g  Cy=%.4g  Cz=%.4g",
    maximum(r.Cx for r in results), maximum(r.Cy for r in results), maximum(r.Cz for r in results),
)

locs = [r.loc for r in results if r.loc !== nothing]
if !isempty(locs)
    worst = argmax(l -> l.cfl, locs)
    @info @sprintf(
        "worst cell: month %d (i=%d,j=%d,k=%d) depth=%.1fm%s  u=%+.4g v=%+.4g w=%+.4g m/s",
        worst.month, worst.i, worst.j, worst.k, zC[min(worst.k, length(zC))],
        coordstr(worst.i, worst.j), worst.u, worst.v, worst.w,
    )
end

@info @sprintf(
    "Δt for CFL = %.3g : %.6g s (%.4g h) → current Δt is %.3g× that",
    CFL_TARGET, CFL_TARGET * τ_min, CFL_TARGET * τ_min / 3600, Δt / (CFL_TARGET * τ_min),
)
if cfl_max > 1
    @warn @sprintf(
        "Advective CFL = %.4g > 1 — Δt is above the classic explicit stability limit. Reduce TIMESTEP_MULT by a factor of ≳ %d.",
        cfl_max, ceil(Int, cfl_max / CFL_TARGET),
    )
end

################################################################################
# CFL across the whole TIMESTEP_MULT ladder
################################################################################

# The CFL is purely kinematic: the velocities do not depend on M, the advection
# scheme, or the timestepper, and Δt = M·Δt_base. So CFL(M) = M · CFL(M=1)
# exactly, and one probe run gives the entire ladder for free. What *is*
# scheme- and integrator-dependent is the threshold this number has to stay
# under — compare against the recorded stability table in
# docs/timestep_multiplier.md.
M = parse(Int, require_env("TIMESTEP_MULT"))
Δt_base = Δt / M
year_seconds = 365.25 * 86400
N_base = round(Int, year_seconds / Δt_base)
cfl_per_M = cfl_max / M
practical_M_max = floor(Int, 18 * 3600 / Δt_base)
ladder = filter(≤(practical_M_max), _divisors(N_base))

@info @sprintf(
    "CFL ladder (Δt_base = %.6g s, N_base = %d, CFL per unit M = %.5g):",
    Δt_base, N_base, cfl_per_M,
)

# Per-cell worst-over-months CFL at M = 1. CFL(M) = M · CFL(1) cell-wise, so one
# array gives the exceedance count at every rung. A single marginal cell is far
# less alarming than a whole region over the limit, so report both.
cfl1_percell = isempty(acc) ? nothing : Δt_base .* acc[1]
n_wet = cfl1_percell === nothing ? 0 : count(>(0), cfl1_percell)

for m in ladder
    over = cfl1_percell === nothing ? "" :
        @sprintf(
            "   cells with CFL>1: %8d (%.3g%% of %d moving cells)",
            count(>(1 / m), cfl1_percell), 100 * count(>(1 / m), cfl1_percell) / n_wet, n_wet
        )
    @info @sprintf(
        "    M = %3d   Δt = %6.2f h   CFL = %.4g%s%s",
        m, m * Δt_base / 3600, m * cfl_per_M, over, m == M ? "   ← current" : "",
    )
end

# Explicit horizontal diffusion is the other Δt constraint (the vertical closure
# is VerticallyImplicit → no restriction). Only active when GM_REDI is off.
(; κH) = parse_kappa_env()
τ_diff = cell_diffusion_timescale(
    HorizontalScalarDiffusivity(κ = κH), nothing, grid, Clock(; time = 0.0), NamedTuple(),
)
@info @sprintf(
    "horizontal diffusive CFL (κH = %.4g m²/s) = %.5g   [τ_diff = %.6g s = %.4g h; applies only when GM_REDI is off]",
    κH, Δt / τ_diff, τ_diff, τ_diff / 3600,
)

@info "probe_CFL.jl complete"
flush(stdout); flush(stderr)
