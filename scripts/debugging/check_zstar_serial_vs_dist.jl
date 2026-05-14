"""
Sanity-check whether the CPU diag zstar files are actually bit-identical
between serial (1x1) and distributed (1x2), bypassing the compare script's
slicing.

For each of sigma_cc, dt_sigma, eta_n at iter 0 and the last iter, this:
  1. Reads the global (1x1) array from `MC/{zname}_diag_cpu.jld2`
  2. Reads the per-rank (1x2) arrays from `MC/1x2/{zname}_diag_cpu_rank{0,1}.jld2`
  3. Reports sizes, dtypes
  4. Tries to align: for each rank, slices the global array to match the rank's
     shape via a sweep of likely (x_off, y_off) offsets, and reports the offset
     that yields max|diff| = 0 (if any).
"""

using JLD2
using Printf
using Statistics

const MC_DIR = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"

function load_iter(path, field, iter_label)
    return jldopen(path, "r") do f
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        it = iter_label == :first ? iters[1] : iters[end]
        return (it, f["timeseries/$field/$it"])
    end
end

function try_align(global_arr, rank_arr, name)
    nxs, nys = size(global_arr, 1), size(global_arr, 2)
    nxr, nyr = size(rank_arr, 1), size(rank_arr, 2)
    if (nxs, nys) == (nxr, nyr)
        diff = rank_arr .- global_arr
        max_abs = maximum(abs, filter(isfinite, vec(diff)))
        return (offset = (0, 0), max_abs = max_abs)
    end
    best = (offset = nothing, max_abs = Inf)
    for x_off in 0:(nxs - nxr), y_off in 0:(nys - nyr)
        sl = if ndims(global_arr) == 3
            global_arr[(x_off + 1):(x_off + nxr), (y_off + 1):(y_off + nyr), :]
        else
            global_arr[(x_off + 1):(x_off + nxr), (y_off + 1):(y_off + nyr)]
        end
        size(sl) == size(rank_arr) || continue
        diff = rank_arr .- sl
        flat = filter(isfinite, vec(diff))
        max_abs = isempty(flat) ? NaN : maximum(abs, flat)
        if max_abs < best.max_abs
            best = (offset = (x_off, y_off), max_abs = max_abs)
        end
    end
    return best
end

for zname in ("sigma_cc", "dt_sigma", "eta_n")
    println("=== $zname ===")
    serial_path = joinpath(MC_DIR, "$(zname)_diag_cpu.jld2")
    isfile(serial_path) || (println("  missing $serial_path"); continue)

    for it_label in (:first, :last)
        it_s, gdata = load_iter(serial_path, zname, it_label)
        println("  iter $it_s (serial): size=$(size(gdata)), dtype=$(eltype(gdata))")
        for r in 0:1
            rank_path = joinpath(MC_DIR, "1x2", "$(zname)_diag_cpu_rank$(r).jld2")
            isfile(rank_path) || (println("    rank $r missing"); continue)
            it_r, rdata = load_iter(rank_path, zname, it_label)
            @assert it_r == it_s
            res = try_align(gdata, rdata, zname)
            @printf "    rank %d size=%s -> best offset=%s max|diff|=%.3e\n" r size(rdata) res.offset res.max_abs
        end
    end
    println()
end
