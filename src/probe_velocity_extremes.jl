"""
Read-only diagnostic: scan the preprocessed monthly input velocity / free-surface
FieldTimeSeries for NaN/Inf and extreme magnitudes, to hunt the source of the
NK forward-map (Φ!) NaN blow-up seen for the Li et al. Qian OM2-01 experiments.

GLOBAL scan (not fold-focused). For each field (u, v, w, η) and each of the 12
monthly snapshots it reports: NaN/Inf counts, min/max, the location (i,j,k, depth,
lon/lat if available) of the largest |value|, and a magnitude histogram. A single
extreme cell (|u| ≫ a few m/s away from the fold) would be a smoking gun for a
local CFL blow-up; clean inputs would instead point at a dynamic instability
(→ localize via a model run).

Reads one snapshot at a time (OnDisk backend) so it fits in ~15 GB RAM despite
the ~80 GB per-field files. CPU only, no model run.

Env: PARENT_MODEL, EXPERIMENT, TIME_WINDOW (via load_project_config).
"""

@info "Loading packages for velocity-extremes probe"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znodes, λnodes, φnodes
using Statistics
using Printf

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir, monthly_dir) = load_project_config()
@info "Probing" parentmodel experiment time_window monthly_dir
flush(stdout); flush(stderr)

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
zC = collect(znodes(grid, Center(), Center(), Center()))

# Best-effort horizontal coordinates (tripolar → 2D arrays); fall back to indices.
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

# (var name inside the JLD2, on-disk file); vs_prefix is mass_transport for cgrid.
fields = [
    ("u", joinpath(monthly_dir, "u_from_mass_transport_monthly.jld2")),
    ("v", joinpath(monthly_dir, "v_from_mass_transport_monthly.jld2")),
    ("w", joinpath(monthly_dir, "w_from_mass_transport_monthly.jld2")),
    ("η", joinpath(monthly_dir, "eta_monthly.jld2")),
]

# Histogram thresholds (m/s for velocities; η is metres but reuses the bins).
thresholds = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

for (name, file) in fields
    if !isfile(file)
        @warn "missing input, skipping" name file
        continue
    end
    @info "──────── field $name ($(basename(file))) ────────"
    flush(stdout); flush(stderr)
    fts = FieldTimeSeries(file, name; backend = OnDisk())
    Nt = length(fts.times)

    tot_nan = 0
    tot_inf = 0
    g_absmax = -Inf
    g_loc = (0, 0, 0)
    g_month = 0
    g_signed = 0.0
    over = zeros(Int, length(thresholds))   # counts of |value| > threshold (all months)

    for t in 1:Nt
        a = Float64.(Array(interior(fts[t])))
        nnan = count(isnan, a)
        ninf = count(x -> isinf(x) && !isnan(x), a)
        tot_nan += nnan
        tot_inf += ninf

        # abs with non-finite set to -Inf so findmax ignores them
        absa = map(x -> isfinite(x) ? abs(x) : -Inf, a)
        mx, ci = findmax(absa)
        finite_any = mx > -Inf
        if finite_any
            i, j, k = Tuple(ci)
            for (n, thr) in enumerate(thresholds)
                over[n] += count(>(thr), absa)   # -Inf never exceeds thr
            end
            if mx > g_absmax
                g_absmax = mx
                g_loc = (i, j, k)
                g_month = t
                g_signed = a[ci]
            end
            lo, hi = extrema(x for x in a if isfinite(x))
            @info @sprintf(
                "  month %2d: min=%+.4g max=%+.4g  |max|=%.4g @ (i=%d,j=%d,k=%d) depth=%.1fm%s  NaN=%d Inf=%d",
                t, lo, hi, mx, i, j, k, zC[min(k, length(zC))], coordstr(i, j), nnan, ninf,
            )
        else
            @info @sprintf("  month %2d: ALL non-finite  NaN=%d Inf=%d", t, nnan, ninf)
        end
        flush(stdout); flush(stderr)
    end

    @info "  ── $name summary ──"
    @info @sprintf(
        "  global |max| = %.5g (signed %+.5g) @ month %d (i=%d,j=%d,k=%d) depth=%.1fm%s",
        g_absmax, g_signed, g_month, g_loc[1], g_loc[2], g_loc[3],
        zC[min(max(g_loc[3], 1), length(zC))],
        coordstr(max(g_loc[1], 1), max(g_loc[2], 1)),
    )
    @info "  total NaN=$tot_nan  Inf=$tot_inf across all $Nt months"
    for (n, thr) in enumerate(thresholds)
        @info @sprintf("  cells with |%s| > %5.1f : %d", name, thr, over[n])
    end
    flush(stdout); flush(stderr)
end

@info "probe_velocity_extremes.jl complete"
flush(stdout); flush(stderr)
