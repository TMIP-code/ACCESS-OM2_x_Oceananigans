"""
Locate where the v-rank1 interior diff lives on CPU diag.
"""

using JLD2
using OffsetArrays
using Printf

const MC = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const DT = get(ENV, "DURATION_TAG", "diag_cpu")
const Hx, Hy = 13, 13

function load_iter(path, field, which)
    return jldopen(path, "r") do f
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        it = which == :first ? (length(iters) >= 2 ? iters[2] : iters[1]) : iters[end]
        return (it, f["timeseries/$field/$it"])
    end
end

it, gd = load_iter(joinpath(MC, "v_$(DT).jld2"), "v", :first)
gd = gd isa OffsetArray ? parent(gd) : gd
it2, r1 = load_iter(joinpath(MC, "1x2", "v_$(DT)_rank1.jld2"), "v", :first)
r1 = r1 isa OffsetArray ? parent(r1) : r1
@assert it == it2

println("v iter=$it")
println("  serial global parent size: ", size(gd))
println("  rank 1 parent size:        ", size(r1))
println("  rank 1 parent y=1 maps to global y=151 (rank0 interior=150)")

# Rank 1 parent covers global y = 151:327. Slice global:
sl = gd[1:size(r1, 1), 151:(151 + size(r1, 2) - 1), :]
diff = r1 .- sl
println("  diff shape: ", size(diff), "  max|diff|=", maximum(abs, diff))

# Top 20 diff locations (in rank 1's parent index frame)
abs_diff = abs.(diff)
flat = vec(abs_diff)
perm = sortperm(flat; rev = true)[1:30]
cs = CartesianIndices(size(diff))
println()
println("# Top diff locations (rank 1 parent indices; global y = rank_y + 150)")
println("| rank_i | rank_j | rank_k | global_i | global_j | rank_val | serial_val | diff |")
println("|--------|--------|--------|----------|----------|----------|------------|------|")
for p in perm
    c = cs[p]
    flat[p] == 0 && continue
    @printf "| %3d    | %3d    | %2d     | %3d      | %3d      | %.3e | %.3e  | %.3e |\n" c[1] c[2] c[3] c[1] (c[2] + 150) r1[c] sl[c] diff[c]
end

println()
println("Interior of rank 1 (Face-y v): parent y=14:164 (=151 cells = Ny_INT+1 - 150 = 301-150).")
println("In global y this is 164:314.")
println("Land/fold-row range to check: global y=Ny+1=301 → rank 1 parent y=151.")
