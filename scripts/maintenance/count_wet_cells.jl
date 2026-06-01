# One-off: report wet-cell count (= TM dimension N) and nnz for each parent-model
# resolution by loading the saved transport matrix M.jld2.
# The TM is N×N where N = number of wet grid cells, so size(M) gives the count.

using JLD2
using SparseArrays

const CASES = [
    ("ACCESS-OM2-1   (1deg)", "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/TM/cgridtransports_wdiagnosed_centered2_AB2/const/M.jld2"),
    ("ACCESS-OM2-025 (0.25deg)", "outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/TM/totaltransport_wdiagnosed_centered2_SRK3_mkappaV_LBS_DTx9/const/M.jld2"),
    ("ACCESS-OM2-01  (0.1deg)", "outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/TM/cgridtransports_wparent_centered2_AB2_mkappaV_LBS_DTx2/const/M.jld2"),
]

for (label, path) in CASES
    if !isfile(path)
        println(rpad(label, 26), " : MISSING ($path)")
        continue
    end
    M = load(path, "M")
    n = size(M, 1)
    z = nnz(M)
    println(
        rpad(label, 26), " : wet cells N = ", n,
        "   (size = ", size(M), ", nnz = ", z,
        ", avg nnz/row = ", round(z / n, digits = 2), ")"
    )
    M = nothing
    GC.gc()
end
