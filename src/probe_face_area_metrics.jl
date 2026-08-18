"""
Read-only diagnostic: compute the C-grid face-area metrics used by
prep_velocities.jl (AxFCC, AyCFC, AzCCF) and hunt degenerate (near-zero) wet
face areas. A near-zero AyCFC/AzCCF explains the ~1e298 v/w blow-up, since
prep_velocities does `v_mt .= ty / (ρ₀ * AyCFC)` — a normal ty (~1e6) over a
~1e-292 area gives ~1e298.

Reports, per metric: the smallest POSITIVE value and its (i,j,k)+depth, the max,
and how many wet cells fall below physical floors. Also prints the metric at a
caller-supplied cell (PROBE_I/J/K, default the wthp hotspot 848,1246,16) and the
implied v = TY_REF / (ρ₀ * AyCFC) there.

Env: PARENT_MODEL, EXPERIMENT, TIME_WINDOW; optional PROBE_I/J/K, TY_REF.
CPU only. ~17 GB for the three area fields.
"""

@info "Loading packages for face-area-metric probe"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.AbstractOperations: grid_metric_operation, Ax, Ay, Az
using Oceananigans.Grids: znodes
using Printf

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir) = load_project_config()
grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
zC = collect(znodes(grid, Center(), Center(), Center()))
const ρ₀ = 1035.0

pi_ = parse(Int, get(ENV, "PROBE_I", "848"))
pj_ = parse(Int, get(ENV, "PROBE_J", "1246"))
pk_ = parse(Int, get(ENV, "PROBE_K", "16"))
ty_ref = parse(Float64, get(ENV, "TY_REF", "1.3e6"))

@info "Probing face-area metrics" experiment time_window probe_cell = (pi_, pj_, pk_)
flush(stdout); flush(stderr)

metrics = [
    ("AxFCC", (Face(), Center(), Center()), Ax),
    ("AyCFC", (Center(), Face(), Center()), Ay),
    ("AzCCF", (Center(), Center(), Face()), Az),
]

for (nm, loc, op) in metrics
    A = Field(grid_metric_operation(loc, op, grid))
    compute!(A)
    ai = Array(interior(A))

    posmask = ai .> 0
    npos = count(posmask)
    # smallest positive value + location
    aim = map(x -> x > 0 ? x : Inf, ai)
    mn, ci = findmin(aim)
    i, j, k = Tuple(ci)
    mx = maximum(ai)

    @info "──── $nm ────"
    @info @sprintf(
        "  smallest positive = %.4g @ (i=%d,j=%d,k=%d) depth=%.1fm ; max = %.4g ; npos=%d",
        mn, i, j, k, zC[min(max(k, 1), length(zC))], mx, npos,
    )
    for floor in (1.0, 1.0e-2, 1.0e-6, 1.0e-12, 1.0e-100)
        @info @sprintf("  wet cells with 0 < %s < %.0e : %d", nm, floor, count(x -> 0 < x < floor, ai))
    end
    # value at the probe cell + implied v if this is AyCFC
    if checkbounds(Bool, ai, pi_, pj_, pk_)
        av = ai[pi_, pj_, pk_]
        @info @sprintf("  %s at probe cell (%d,%d,%d) = %.6g", nm, pi_, pj_, pk_, av)
        if nm == "AyCFC" && av != 0
            @info @sprintf("    ⇒ v = TY_REF/(ρ₀·AyCFC) = %.4g / (%.1f·%.4g) = %.4g m/s", ty_ref, ρ₀, av, ty_ref / (ρ₀ * av))
        end
    else
        @info "  probe cell $(pi_),$(pj_),$(pk_) out of interior bounds $(size(ai))"
    end
    A = nothing
    ai = nothing
    aim = nothing
    GC.gc()
    flush(stdout); flush(stderr)
end

@info "probe_face_area_metrics.jl complete"
flush(stdout); flush(stderr)
