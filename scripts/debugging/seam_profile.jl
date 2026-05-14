"""
Per-row max|diff| profile across global y, focused on the rank-rank seam.

For a 1×2 partition, the seam is at global y ≈ 150 (interior split between
rank 0 and rank 1). This script prints max|diff| (over i, k) at every global
y in a band around the seam, for every saved field, so we can see exactly
which y-row first acquires a non-zero diff.

Usage:
  DURATION_TAG=diag     julia --project scripts/debugging/seam_profile.jl
  DURATION_TAG=diag_cpu julia --project scripts/debugging/seam_profile.jl
"""

using JLD2
using OffsetArrays
using Printf

const MC = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const DT = get(ENV, "DURATION_TAG", "diag")
const Hx, Hy = 13, 13
const Ny_INT = 300
const Ny_RANK_CENTER = 150   # rank 0 Center-y interior

# Seam band to scan (global Center-y indices): [Ny_RANK_CENTER - 5 .. Ny_RANK_CENTER + 5]
const SEAM_BAND = (Ny_RANK_CENTER - 5):(Ny_RANK_CENTER + 5)   # 145..155

const FIELDS = ("age", "u", "v", "w", "eta", "sigma_cc", "dt_sigma", "eta_n")

function load_iter(path, field, which)
    return jldopen(path, "r") do f
        haskey(f, "timeseries/$field") || return nothing
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        isempty(iters) && return nothing
        it = which == :first ? (length(iters) >= 2 ? iters[2] : iters[1]) : iters[end]
        return (it, f["timeseries/$field/$it"])
    end
end

function as_array(a)
    return a isa OffsetArray ? parent(a) : a
end

for field in FIELDS
    sp = joinpath(MC, "$(field)_$(DT).jld2")
    isfile(sp) || continue

    which = field in ("age", "sigma_cc", "dt_sigma", "eta_n") ? :last : :first
    res = load_iter(sp, field, which)
    res === nothing && continue
    it, gd_raw = res
    gd = as_array(gd_raw)
    nxs, nys = size(gd, 1), size(gd, 2)

    # Determine rank-1 file to splice in seam-neighbouring distributed values
    rp0 = joinpath(MC, "1x2", "$(field)_$(DT)_rank0.jld2")
    rp1 = joinpath(MC, "1x2", "$(field)_$(DT)_rank1.jld2")
    (isfile(rp0) && isfile(rp1)) || continue
    r0 = as_array(load_iter(rp0, field, which)[2])
    r1 = as_array(load_iter(rp1, field, which)[2])

    println("## $field  iter=$it  global parent y=1..$nys (Hy=$Hy)")
    println()
    println("Global Center-y interior is parent y=14..313 (300 cells), seam between")
    println("global Center-y=150 (rank 0's last interior) and 151 (rank 1's first interior),")
    println("which in global parent y is rows 163 and 164 respectively.")
    println()
    println("| global parent y | rank-0 has? (y_p) | rank-1 has? (y_p) | max\\|r0-serial\\|@y | max\\|r1-serial\\|@y |")
    println("|---|---|---|---|---|")

    # Translate the Center-y interior seam band to parent y. Center-y interior 145..155 → parent y 158..168.
    # Print parent y from 158 to 168 (the seam neighbourhood).
    for gp in 156:170
        gp >= 1 && gp <= nys || continue

        # rank 0's parent y range: 1..size(r0,2). global parent y == rank0 parent y for y∈1..176.
        r0_has = gp <= size(r0, 2)
        # rank 1's parent y range: 151..150+size(r1,2). global parent y == rank1 parent y + 150.
        r1_y = gp - 150
        r1_has = r1_y >= 1 && r1_y <= size(r1, 2)

        # max|diff| at this global parent y, over (i, k)
        s_slab = ndims(gd) >= 3 ? gd[:, gp, :] : gd[:, gp]
        d0_str, d1_str = "—", "—"
        d0_loc, d1_loc = "", ""
        if r0_has
            r0_slab = ndims(r0) >= 3 ? r0[:, gp, :] : r0[:, gp]
            d0 = r0_slab .- s_slab
            absd = abs.(d0)
            fmax = maximum(x -> isfinite(x) ? x : -Inf, absd)
            d0_str = isfinite(fmax) ? @sprintf("%.3e", fmax) : "NaN"
            d0_loc = "(y_p=$gp)"
        end
        if r1_has
            r1_slab = ndims(r1) >= 3 ? r1[:, r1_y, :] : r1[:, r1_y]
            d1 = r1_slab .- s_slab
            absd = abs.(d1)
            fmax = maximum(x -> isfinite(x) ? x : -Inf, absd)
            d1_str = isfinite(fmax) ? @sprintf("%.3e", fmax) : "NaN"
            d1_loc = "(y_p=$r1_y)"
        end

        # Annotate seam crossing
        tag = if gp == 163
            "  ← rank0 last interior (global Center y=150)"
        elseif gp == 164
            "  ← rank1 first interior (global Center y=151) **SEAM**"
        else
            ""
        end

        @printf "| %3d | %s | %s | %s | %s |%s\n"  gp  (r0_has ? "Y $d0_loc" : "—")  (r1_has ? "Y $d1_loc" : "—")  d0_str  d1_str  tag
    end
    println()
end
