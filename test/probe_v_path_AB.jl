"""
Diagnostic: compare Path A vs Path B of FieldTimeSeries `fts[Time(t)]`.

Path A: n₁ == n₂ — returns the stored Field directly (no compute!).
Path B: n₁ ≠ n₂ — builds Field(ψ₂*ñ + ψ₁*(1-ñ)) and runs compute!.

For v_ts loaded from disk, check whether the interpolated value at an
EXACT snapshot time matches the stored snapshot — for both u (no fold
interaction) and v (fold at j=Ny). If Path B at exact snapshot time
returns a different value than the stored snapshot, that's the source
of the original trafftsrev failure.
"""

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
using Oceananigans.Fields: interior
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days

include("../src/shared_functions.jl")

cfg = load_project_config()
(; experiment_dir, monthly_dir) = cfg

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
@info "Grid size" size_grid = size(grid)

for (label, file, name) in [
        ("u", "u_from_total_transport_monthly.jld2", "u"),
        ("v", "v_from_total_transport_monthly.jld2", "v"),
    ]
    @info "=== $label ==="
    fts = load_fts(CPU(), joinpath(monthly_dir, file), name, grid; backend = InMemory(), time_indexing = Cyclical(1year))
    @info "Loaded FTS" name boundary_conditions = fts.boundary_conditions

    t1 = fts.times[1]
    tN = fts.times[end]
    N = length(fts.times)

    # Path A: t = times[1] (exact first snapshot, expect n₁ == n₂ == 1)
    pA = fts[Time(t1)]
    # Stored snapshot at index 1
    s1 = fts[1]
    diff_A = maximum(abs, interior(pA) .- interior(s1))
    @info "Path A: fts[Time(times[1])] vs fts[1]" t1 max_abs_diff = diff_A pA_eq_s1_bywhat = (pA === s1 ? :same_object : :different_object)

    # Path B: t = times[end] (exact last snapshot, expect n₁=N-1, n₂=N, ñ=1.0)
    pB = fts[Time(tN)]
    sN = fts[N]
    diff_B = maximum(abs, interior(pB) .- interior(sN))
    @info "Path B: fts[Time(times[end])] vs fts[end]" tN max_abs_diff = diff_B pB_eq_sN_bywhat = (pB === sN ? :same_object : :different_object)

    # Where is the Path-B diff concentrated? Find the (i, j, k) of the max diff.
    if diff_B > 0
        IB = interior(pB)
        IsN = interior(sN)
        maxij = (0, 0, 0)
        maxd = 0.0
        for k in axes(IB, 3), j in axes(IB, 2), i in axes(IB, 1)
            d = abs(IB[i, j, k] - IsN[i, j, k])
            if d > maxd
                maxd = d
                maxij = (i, j, k)
            end
        end
        @info "Path B max diff location" maxij maxd
        # Report how many j-rows have nonzero diff
        rows_with_diff = 0
        for j in axes(IB, 2)
            row_max = 0.0
            for k in axes(IB, 3), i in axes(IB, 1)
                row_max = max(row_max, abs(IB[i, j, k] - IsN[i, j, k]))
            end
            if row_max > 1.0e-14
                rows_with_diff += 1
            end
        end
        @info "Number of j-rows with diff > 1e-14" rows_with_diff total_j = size(IB, 2)
    end

    # Also test t = a mid-snapshot value (Path B with ñ in middle) — sanity check
    Δt = fts.times[2] - fts.times[1]
    t_mid = fts.times[1] + Δt / 2
    pmid = fts[Time(t_mid)]
    # The expected interior at t_mid: linear blend of interior(s1) and interior(s2)
    s2 = fts[2]
    expected_mid = (interior(s1) .+ interior(s2)) ./ 2
    diff_mid = maximum(abs, interior(pmid) .- expected_mid)
    @info "Path B at mid-snapshot (sanity)" t_mid max_abs_diff_vs_naive_blend = diff_mid
end

@info "Done"
