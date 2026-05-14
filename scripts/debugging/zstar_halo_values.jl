"""
Probe the actual values stored in the seam-halo cells of the saved zstar files.
If the halo cells are *exactly* the initial values (sigma=1.0, eta=0.0, dt_sigma=0.0),
the halo was NEVER filled — meaning at runtime the model is reading stale placeholder
values when computing seam-crossing fluxes (cell face heights via zstar). If the halo
values are non-trivial (e.g., 0.998, 0.42 etc), the halo was filled at some point but
not re-filled before save.

Usage:
  DURATION_TAG=diag_cpu  julia --project scripts/debugging/zstar_halo_values.jl
  DURATION_TAG=diag      julia --project scripts/debugging/zstar_halo_values.jl
"""

using JLD2
using OffsetArrays
using Printf
using Statistics

const MC = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const DT = get(ENV, "DURATION_TAG", "diag_cpu")
const Hx, Hy = 13, 13

function load_iter(path, field, which)
    return jldopen(path, "r") do f
        haskey(f, "timeseries/$field") || return nothing
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        isempty(iters) && return nothing
        it = which == :first ? (length(iters) >= 2 ? iters[2] : iters[1]) : iters[end]
        return (it, f["timeseries/$field/$it"])
    end
end

as_array(a) = a isa OffsetArray ? parent(a) : a

println("# Seam-halo value probe — DURATION_TAG=$DT")
println()
println("Inspecting the actual stored values in rank 0's north-halo row (the cells")
println("immediately above its interior, which face rank 1 across the seam) and")
println("rank 1's south-halo row (immediately below its interior).")
println()
println("For Center-y fields, rank parent y=164..176 are halo cells (the 13 rows")
println("just above rank 0's interior). The row immediately adjacent to the interior")
println("is parent y=164 for rank 0 / parent y=13 for rank 1 — these are the seam rows")
println("the model reads for flux computation.")
println()

for (field, initval) in (
        ("sigma_cc", 1.0),
        ("eta_n", 0.0),
        ("dt_sigma", 0.0),
        ("u", 0.0),
        ("v", 0.0),
        ("w", 0.0),
        ("eta", 0.0),
        ("age", 0.0),
    )
    sp = joinpath(MC, "$(field)_$(DT).jld2")
    isfile(sp) || (println("$field: missing $sp"); continue)
    it, gd_raw = load_iter(sp, field, :last)
    gd = as_array(gd_raw)

    for r in 0:1
        rp = joinpath(MC, "1x2", "$(field)_$(DT)_rank$(r).jld2")
        isfile(rp) || continue
        rd_raw = load_iter(rp, field, :last)
        rd = as_array(rd_raw[2])

        # Probe several halo rows: nearest to interior, middle, outermost.
        # For rank 0: parent y=164 (seam-adjacent), 170 (middle), 176 (outermost).
        # For rank 1: parent y=13 (seam-adjacent), 7 (middle), 1 (outermost).
        halo_y = r == 0 ? get(ENV, "HALO_Y", "164") |> z -> parse(Int, z) : get(ENV, "HALO_Y", "13") |> z -> parse(Int, z)

        # Get the 1D strip across i for this halo row, at k=1 (or k=middle for 3D)
        n_k = ndims(rd) == 3 ? size(rd, 3) : 1
        k_idx = n_k > 1 ? max(1, n_k ÷ 2) : 1
        strip_rank = ndims(rd) == 3 ? rd[:, halo_y, k_idx] : rd[:, halo_y]
        strip_global = ndims(gd) == 3 ? gd[:, r == 0 ? halo_y : halo_y + 150, k_idx] : gd[:, r == 0 ? halo_y : halo_y + 150]

        # Compute stats
        flat_rank = filter(isfinite, strip_rank)
        flat_global = filter(isfinite, strip_global)
        n_at_init = count(==(initval), flat_rank)
        n_total = length(flat_rank)
        # Pick a couple of representative (i, j, k) cells near the equator for inspection
        @printf "## %s rank %d  (init val = %s)\n" field r initval
        @printf "  halo row probed: rank parent y=%d, k=%d  (size strip = %d)\n" halo_y k_idx length(strip_rank)
        @printf "  rank halo strip: min=%.6e  max=%.6e  mean=%.6e\n" minimum(flat_rank) maximum(flat_rank) mean(flat_rank)
        @printf "  global slice:    min=%.6e  max=%.6e  mean=%.6e\n" minimum(flat_global) maximum(flat_global) mean(flat_global)
        @printf "  rank cells == init value (%s): %d / %d (%.1f%%)\n" initval n_at_init n_total 100 * n_at_init / n_total

        # Sample 5 cells to show
        sample_is = round.(Int, range(20, size(rd, 1) - 20; length = 5))
        for i in sample_is
            rv = strip_rank[i]
            gv = strip_global[i]
            @printf "  i=%3d: rank=%.6e  global=%.6e  diff=%.3e\n" i rv gv (rv - gv)
        end
        println()
    end
end
