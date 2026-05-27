"""
Probe A — κV halo at the rank-1 seam (CPU vs GPU).

Reads the iter-0 full-state dumps written by `test/probe_tracer_tendency.jl`
and compares the vertical-diffusion field κV CPU↔GPU at the parent y rows
that matter on rank 1:
  - parent y = Hy        = 13  : last south-halo row (rank-rank seam edge)
  - parent y = Hy + 1    = 14  : first interior row  (the "seam row")
  - parent y = Hy + 2    = 15  : second interior row
  - parent y = Ny+Hy     = Ny+13 : last interior row (fold-side)
  - parent y = Ny+Hy + 1 = Ny+14 : first north-halo row (tripolar fold)
  - parent y = Ny+Hy + 2 = Ny+15 : second north-halo row

If CPU and GPU disagree on κV at parent y = 13 or 14 on rank 1, the bug is
upstream of `implicit_step!` — in the κV halo-fill / partition path.

Expected outcome (per docs/next_probes_implicit_step.md): clean. κV is
loaded once and partitioned offline, and the existing probe-iter-0 dump
already showed every input identical except `w` (1 ULP).

Run on the login node (pure JLD2 reading; no GPU needed):
    julia --project scripts/debugging/probe_kV_at_seam.jl
"""

include("../../src/shared_utils/config.jl")

using JLD2
using Printf
using Statistics

(; outputdir) = load_project_config()
model_config = require_env("MODEL_CONFIG")

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "2"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
probe_root = isempty(gpu_tag) ?
    joinpath(outputdir, "standardrun", model_config, "probe") :
    joinpath(outputdir, "standardrun", model_config, gpu_tag, "probe")

iter = parse(Int, get(ENV, "PROBE_ITER", "0"))
acm_suffix = lowercase(get(ENV, "ACTIVE_CELLS_MAP", "yes")) == "no" ? "_noACM" : ""

# Halo size for OM2-1 with the bp/offline_ACCESS-OM2_v3 fork; matches
# scripts/debugging/compare_tendency_probes.jl.
const Hy = 13

probe_path(device, rank) = joinpath(
    probe_root,
    "probe_tendency_$(device)_iter$(iter)_rank$(rank)$(acm_suffix).jld2",
)

"""
List the `κV*` keys present in the file (the probe writer stores either a
single `κV` for an AbstractField, several `κV_<name>` entries for a
NamedTuple of fields, or a `κV_scalar` for a scalar). Skip the scalar form
since there's nothing to compare positionally.
"""
function kappa_v_keys(path)
    return jldopen(path, "r") do f
        ks = collect(keys(f))
        return filter(k -> startswith(k, "κV") && k != "κV_scalar", ks)
    end
end

function load_kappa_v(path, key)
    return jldopen(path, "r") do f
        return Float64.(f[key])
    end
end

function row_stats(label, a_row, b_row)
    d = b_row .- a_row
    flat = filter(isfinite, vec(d))
    if isempty(flat)
        @printf "    %-32s n_finite=0 (all NaN/land)\n" label
        return
    end
    a_flat = filter(isfinite, vec(a_row))
    b_flat = filter(isfinite, vec(b_row))
    @printf(
        "    %-32s n_finite=%-6d  max|cpu|=%.3e max|gpu|=%.3e  max|diff|=%.6e  n_differ=%d\n",
        label,
        length(flat),
        isempty(a_flat) ? NaN : maximum(abs, a_flat),
        isempty(b_flat) ? NaN : maximum(abs, b_flat),
        maximum(abs, flat),
        count(!iszero, flat),
    )
    return
end

@info "probe_kV_at_seam:"
@info "  probe_root = $probe_root"
@info "  iter       = $iter   PARTITION = $(px)x$(py)   ACM_suffix='$acm_suffix'"

for rank in 0:(px * py - 1)
    cpu_path = probe_path("cpu", rank)
    gpu_path = probe_path("gpu", rank)
    if !isfile(cpu_path)
        @warn "missing CPU probe" cpu_path
        continue
    end
    if !isfile(gpu_path)
        @warn "missing GPU probe" gpu_path
        continue
    end

    cpu_keys = kappa_v_keys(cpu_path)
    gpu_keys = kappa_v_keys(gpu_path)
    common_keys = sort(collect(intersect(cpu_keys, gpu_keys)))

    println("\n──────── rank $rank ────────")
    println("  CPU: $cpu_path")
    println("  GPU: $gpu_path")
    if isempty(common_keys)
        println("  (no κV* keys in either file — closure has no diffusivity field?)")
        continue
    end
    println("  κV keys: $common_keys")

    for key in common_keys
        a = load_kappa_v(cpu_path, key)
        b = load_kappa_v(gpu_path, key)
        if size(a) != size(b)
            @warn "shape mismatch" key cpu_shape = size(a) gpu_shape = size(b)
            continue
        end
        Nx_p, Ny_p, Nz_p = size(a)
        println("  → $key  parent shape = $(size(a))")

        # Global diff (sanity baseline).
        d = b .- a
        flat = filter(isfinite, vec(d))
        if !isempty(flat)
            @printf(
                "    %-32s n_finite=%-6d  max|diff|=%.6e  n_differ=%d\n",
                "GLOBAL parent",
                length(flat),
                maximum(abs, flat),
                count(!iszero, flat),
            )
        end

        # Rows of interest. The south-halo (Hy) rows on rank 1 face the
        # rank-rank seam; the north-halo rows face the tripolar fold.
        # On rank 0 the south-halo touches the global-south wall (no seam).
        interior_last_y = Ny_p - Hy   # last interior row in parent y
        rows = [
            (Hy, "south_halo  (parent y=$(Hy))"),
            (Hy + 1, "first_int   (parent y=$(Hy + 1)) ← SEAM ROW on rank 1"),
            (Hy + 2, "second_int  (parent y=$(Hy + 2))"),
            (interior_last_y, "last_int    (parent y=$(interior_last_y)) ← fold-side"),
            (interior_last_y + 1, "north_halo  (parent y=$(interior_last_y + 1)) ← fold row 1"),
            (interior_last_y + 2, "north_halo  (parent y=$(interior_last_y + 2)) ← fold row 2"),
        ]
        for (j, label) in rows
            (1 ≤ j ≤ Ny_p) || continue
            row_stats(label, a[:, j, :], b[:, j, :])
        end
    end
end

println("\nprobe_kV_at_seam done.")
