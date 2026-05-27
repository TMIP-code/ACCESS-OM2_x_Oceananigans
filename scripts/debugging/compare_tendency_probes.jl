"""
Compare per-rank tendency-probe JLD2 dumps from CPU vs GPU runs.

Reads probe files written by `test/probe_tracer_tendency.jl`:
    {outputdir}/standardrun/{MC}/{px x py}/probe/probe_tendency_{cpu|gpu}_iter{N}{_rank{R}}{_noACM}.jld2

and prints a per-iter, per-rank, per-field diff summary:
  - max|diff| over full parent array
  - n cells differing
  - for `Gn_age`, the seam-row stats (rank 1 parent y = Hy+1 = 14) — the
    decisive signal for the GPU seam tracer bug.

Set env vars to control the search (defaults match the canonical 1×2 OM2-1 setup):
  PARTITION         (default: 1x2)
  PROBE_NSTEPS      (default: 1)
  PARENT_MODEL      (default: ACCESS-OM2-1)
  ACTIVE_CELLS_MAP  (default: yes  → no `_noACM` suffix)
  EXPERIMENT, TIME_WINDOW, etc. → inherited via shared_functions.jl

Usage (login node):
  PROBE_NSTEPS=1 julia --project scripts/debugging/compare_tendency_probes.jl
"""

# Only pull in `config.jl` (TOML-only) — avoids loading Oceananigans for this
# pure post-processing pass.
include("../../src/shared_utils/config.jl")

using JLD2
using Printf
using Statistics

(; outputdir) = load_project_config()
model_config = require_env("MODEL_CONFIG")

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
probe_root = isempty(gpu_tag) ?
    joinpath(outputdir, "standardrun", model_config, "probe") :
    joinpath(outputdir, "standardrun", model_config, gpu_tag, "probe")

acm_suffix = lowercase(get(ENV, "ACTIVE_CELLS_MAP", "yes")) == "no" ? "_noACM" : ""
nsteps = parse(Int, get(ENV, "PROBE_NSTEPS", "1"))
ranks = (px == 1 && py == 1) ? [-1] : collect(0:(px * py - 1))

# Halo size for OM2-1 with the bp/offline_ACCESS-OM2_v3 fork. Used only for
# reporting "seam-row" stats on Gn_age — if you change Hy upstream, update.
const Hy = 13

function probe_path(device, iter, rank)
    suffix = rank < 0 ? "" : "_rank$(rank)"
    return joinpath(
        probe_root,
        "probe_tendency_$(device)_iter$(iter)$(suffix)$(acm_suffix).jld2",
    )
end

function age_snapshot_path(device, iter, stage, rank)
    suffix = rank < 0 ? "" : "_rank$(rank)"
    return joinpath(
        probe_root,
        "probe_age_$(device)_iter$(iter)_$(stage)$(suffix)$(acm_suffix).jld2",
    )
end

function open_keys(path)
    return jldopen(path, "r") do f
        return Dict(k => f[k] for k in keys(f) if !startswith(k, "meta"))
    end
end

"""
    diff_summary(a, b)

`max|b-a|, n_differ` over finite cells (handles NaN-on-land in both arrays).
"""
function diff_summary(a, b)
    a = Float64.(a); b = Float64.(b)
    d = b .- a
    flat = filter(isfinite, vec(d))
    isempty(flat) && return (max_abs = NaN, n_differ = 0, n_finite = 0)
    return (
        max_abs = maximum(abs, flat),
        n_differ = count(!iszero, flat),
        n_finite = length(flat),
    )
end

function seam_row_diff(a, b)
    ndims(a) == 3 || return nothing
    sz = size(a)
    sz[2] >= Hy + 1 || return nothing
    row_a = a[:, Hy + 1, :]
    row_b = b[:, Hy + 1, :]
    return diff_summary(row_a, row_b)
end

@info "compare_tendency_probes: probe_root = $probe_root"
@info "compare_tendency_probes: PROBE_NSTEPS=$nsteps PARTITION=$(px)x$(py) ACM_suffix='$acm_suffix'"

for iter in 0:nsteps, r in ranks
    cpu_path = probe_path("cpu", iter, r)
    gpu_path = probe_path("gpu", iter, r)
    rank_label = r < 0 ? "serial" : "rank$(r)"
    if !isfile(cpu_path)
        @warn "missing CPU probe" cpu_path
        continue
    end
    if !isfile(gpu_path)
        @warn "missing GPU probe" gpu_path
        continue
    end

    println("\n──────── iter=$iter $rank_label ────────")
    println("  CPU: $cpu_path")
    println("  GPU: $gpu_path")

    cpu = open_keys(cpu_path)
    gpu = open_keys(gpu_path)

    common_keys = sort(collect(intersect(keys(cpu), keys(gpu))))
    any_diff = false

    for k in common_keys
        a, b = cpu[k], gpu[k]
        if !(a isa AbstractArray && b isa AbstractArray)
            continue
        end
        if size(a) != size(b)
            @warn "shape mismatch" key = k cpu_shape = size(a) gpu_shape = size(b)
            continue
        end
        s = diff_summary(a, b)
        if s.max_abs > 0
            any_diff = true
            @printf "  %-25s shape=%-22s max|diff|=%.6e  n_differ=%d / %d\n" k string(size(a)) s.max_abs s.n_differ s.n_finite
        end
    end
    if !any_diff
        println("  (every field bit-identical CPU vs GPU)")
    end

    # Decisive signal: Gn_age at parent y = Hy+1 on rank 1 (the row docs flag).
    if haskey(cpu, "Gn_age") && haskey(gpu, "Gn_age") && r == 1
        seam = seam_row_diff(cpu["Gn_age"], gpu["Gn_age"])
        if seam !== nothing
            @printf "  → Gn_age seam row (rank1 parent y=%d): max|diff|=%.6e  n_differ=%d / %d\n" (Hy + 1) seam.max_abs seam.n_differ seam.n_finite
        end
    end
    if haskey(cpu, "age") && haskey(gpu, "age") && r == 1
        seam = seam_row_diff(cpu["age"], gpu["age"])
        if seam !== nothing
            @printf "  → age seam row    (rank1 parent y=%d): max|diff|=%.6e  n_differ=%d / %d\n" (Hy + 1) seam.max_abs seam.n_differ seam.n_finite
        end
    end
end

# Intra-step age snapshots — one per step, two stages (post_explicit / post_implicit).
# The story: if `post_explicit` already differs CPU vs GPU → bug is in
# `_ab2_step_tracer_field!`. If only `post_implicit` differs → bug is in
# `implicit_step!` (the vertical-diffusion tridiagonal solve).
for step_iter in 0:(nsteps - 1), stage in ("post_explicit", "post_implicit"), r in ranks
    cpu_path = age_snapshot_path("cpu", step_iter, stage, r)
    gpu_path = age_snapshot_path("gpu", step_iter, stage, r)
    if !isfile(cpu_path) || !isfile(gpu_path)
        @warn "missing intra-step snapshot" stage step_iter rank = r cpu = cpu_path gpu = gpu_path
        continue
    end

    rank_label = r < 0 ? "serial" : "rank$(r)"
    println("\n──── step $step_iter→$(step_iter + 1), $stage, $rank_label ────")

    cpu = open_keys(cpu_path)
    gpu = open_keys(gpu_path)

    any_diff = false
    for k in sort(collect(intersect(keys(cpu), keys(gpu))))
        a, b = cpu[k], gpu[k]
        (a isa AbstractArray && b isa AbstractArray) || continue
        size(a) == size(b) || continue
        s = diff_summary(a, b)
        if s.max_abs > 0
            any_diff = true
            @printf "  %-25s shape=%-22s max|diff|=%.6e  n_differ=%d / %d\n" k string(size(a)) s.max_abs s.n_differ s.n_finite
        end
    end
    any_diff || println("  (age + Gn_age bit-identical CPU vs GPU at this stage)")

    if haskey(cpu, "age") && haskey(gpu, "age") && r == 1
        seam = seam_row_diff(cpu["age"], gpu["age"])
        if seam !== nothing
            @printf "  → age seam row    (rank1 parent y=%d): max|diff|=%.6e  n_differ=%d / %d\n" (Hy + 1) seam.max_abs seam.n_differ seam.n_finite
        end
    end
end

println("\ncompare_tendency_probes done.")
