"""
Iteration timeline for the seam tracer bug: at each saved iter, find the
absolute max |rank1_interior - serial| over the entire rank-1 first
interior row (parent y=14 = global Center y=151), and the (i, k) where it
peaks. This tells us when the bug first fires and where the worst cell is.

Usage:
  DURATION_TAG=diag  julia --project scripts/debugging/seam_iter_timeline.jl
  DURATION_TAG=diag_cpu julia --project scripts/debugging/seam_iter_timeline.jl
"""

using JLD2
using OffsetArrays
using Printf

const MC = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const DT = get(ENV, "DURATION_TAG", "diag")
const Hx, Hy = 13, 13

function load_iter(path, field, it)
    return jldopen(path, "r") do f
        return f["timeseries/$field/$it"]
    end
end

function list_iters(path, field)
    return jldopen(path, "r") do f
        return sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
    end
end

as_array(a) = a isa OffsetArray ? parent(a) : a

for field in ("age", "u", "v", "w", "eta", "sigma_cc", "eta_n", "dt_sigma")
    sp = joinpath(MC, "$(field)_$(DT).jld2")
    rp1 = joinpath(MC, "1x2", "$(field)_$(DT)_rank1.jld2")
    (isfile(sp) && isfile(rp1)) || continue
    iters = list_iters(sp, field)

    println("\n## $field — iter timeline at rank 1 first interior row (parent y=14 = global Center y=151)")
    println()
    println("| iter | max\\|r1-serial\\| at y=14 | location (i, k) | rank value at peak | serial value at peak |")
    println("|---|---|---|---|---|")

    for it in iters
        gd = as_array(load_iter(sp, field, it))
        rd1 = as_array(load_iter(rp1, field, it))

        # rank 1's parent y=14 maps to global parent y=164 (rank 1 starts at global y=151 in parent)
        py_r1 = 14
        py_g = 14 + 150
        rd_row = ndims(rd1) >= 3 ? rd1[:, py_r1, :] : rd1[:, py_r1]
        gd_row = ndims(gd) >= 3 ? gd[:, py_g, :] : gd[:, py_g]
        diff = rd_row .- gd_row

        flat = vec(diff)
        finite_mask = isfinite.(flat)
        any(finite_mask) || (println("| $it | (all NaN) | — | — | — |"); continue)
        abs_diff = ifelse.(finite_mask, abs.(flat), -Inf)
        idx_max = argmax(abs_diff)
        c = CartesianIndices(size(diff))[idx_max]
        loc_str = ndims(diff) == 2 ? "($(c[1]), $(c[2]))" : "($(c[1]), --)"
        @printf "| %4d | %.3e | %s | %.3e | %.3e |\n"  it  abs_diff[idx_max]  loc_str  rd_row[c]  gd_row[c]
    end
end
