"""
Read serial JLD2 outputs and print max|.| and mean|.| (over wet cells, NaN-filtered)
for u, v, w, eta at iter 1 (first non-zero step) for both diag and 1year.

Run interactively:
  julia --project scripts/debugging/field_magnitudes.jl
"""

using JLD2
using Statistics
using Printf

const MC_DIR = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"

function field_stats(path, field_name)
    return jldopen(path, "r") do f
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field_name"])))))
        first_nonzero = length(iters) >= 2 ? iters[2] : iters[1]
        last_iter = iters[end]
        data1 = f["timeseries/$field_name/$first_nonzero"]
        data2 = f["timeseries/$field_name/$last_iter"]
        return (first_nonzero, last_iter, data1, data2)
    end
end

function summarize(data)
    flat = filter(isfinite, vec(parent(data)))
    nz = filter(!iszero, flat)
    return (
        max_abs = isempty(flat) ? NaN : maximum(abs, flat),
        mean_abs = isempty(nz) ? NaN : mean(abs, nz),
        rms = isempty(nz) ? NaN : sqrt(mean(abs2, nz)),
    )
end

for duration in ("diag", "1year")
    println("=== $duration ===")
    for fld in ("u", "v", "w", "eta", "age")
        path = joinpath(MC_DIR, "$(fld)_$(duration).jld2")
        if !isfile(path)
            println("  $fld: missing $path")
            continue
        end
        try
            it1, it2, d1, d2 = field_stats(path, fld)
            s1 = summarize(d1)
            s2 = summarize(d2)
            @printf "  %-3s iter %5d: max=%.3e mean=%.3e rms=%.3e\n" fld it1 s1.max_abs s1.mean_abs s1.rms
            @printf "  %-3s iter %5d: max=%.3e mean=%.3e rms=%.3e\n" fld it2 s2.max_abs s2.mean_abs s2.rms
        catch err
            println("  $fld: error: $err")
        end
    end
end
