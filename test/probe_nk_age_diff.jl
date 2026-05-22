using JLD2, Statistics, Printf
using JLD2: load

const DIR = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic"
const CFG = Dict(
    "wdiagnosed" => "cgridtransports_wdiagnosed_centered2_AB2_mkappaV_DTx4",
    "wparent" => "cgridtransports_wparent_centered2_AB2_mkappaV_DTx4",
    "wprediag" => "cgridtransports_wprediag_centered2_AB2_mkappaV_DTx4",
)

function load_age(label)
    f = joinpath(DIR, CFG[label], "NK", "age_Pardiso_LSprec.jld2")
    age = load(f, "age")
    wet3D = load(f, "wet3D")
    yrs = age ./ (365.25 * 24 * 3600)   # convert seconds → years
    return yrs, wet3D
end

a_diag, wet = load_age("wdiagnosed")
a_par, _ = load_age("wparent")
a_pre, _ = load_age("wprediag")
ages = Dict("wdiagnosed" => a_diag, "wparent" => a_par, "wprediag" => a_pre)

println("Grid: ", size(ages["wdiagnosed"]), "  wet cells: ", count(wet))

println("\n--- Per-configuration NK age statistics (yr) ---")
@printf("%-12s %10s %10s %10s %10s\n", "config", "min", "mean", "max", "wet-mean")
for k in ("wdiagnosed", "wparent", "wprediag")
    a = ages[k]
    awet = a[wet]
    @printf(
        "%-12s %10.2f %10.2f %10.2f %10.2f\n",
        k, minimum(awet), mean(awet), maximum(awet), mean(awet)
    )
end

println("\n--- Pairwise NK age differences (yr) ---")
pairs = [("wdiagnosed", "wparent"), ("wdiagnosed", "wprediag"), ("wparent", "wprediag")]
@printf("%-30s %12s %12s %12s %12s\n", "pair (A − B)", "mean(A-B)", "RMS(A-B)", "max|A-B|", "max|rel%|")
for (a, b) in pairs
    d = ages[a] .- ages[b]
    dwet = d[wet]
    base = ages[b][wet]
    rel = 100 .* dwet ./ max.(abs.(base), 1.0)  # avoid /0 for very young water
    @printf(
        "%-30s %+12.4f %12.4f %12.4f %12.2f\n",
        "$a − $b", mean(dwet), sqrt(mean(dwet .^ 2)),
        maximum(abs, dwet), maximum(abs, rel)
    )
end
