"""
Probe C — compare `implicit_step!`'s tri-diagonal LHS coefficients
(a, b, c) CPU vs GPU on rank 1.

Reads `probe_implicit_coeffs_{cpu|gpu}_iter{N}_rank{R}.jld2` produced by
`test/probe_tracer_tendency.jl` and reports:
  - global parent max|diff| per coefficient
  - rank-1 seam-row stats (interior j=1 = parent y = Hy+1 = 14) per coefficient

Interpretation:
  - Coefficients differ on rank 1 seam row → bug is in the coefficient
    builder (likely fold-row-aware κᶜᶜᶠ or vertical metric).
  - Coefficients bit-identical → bug is downstream in the Thomas sweep
    (`solve_batched_tridiagonal_system_kernel!`), i.e. inside
    BatchedTridiagonalSolver itself.

Run on the login node (no GPU needed; pure JLD2 reading):
    PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x2 PROBE_NSTEPS=1 \\
        julia --project scripts/debugging/compare_implicit_coeffs.jl
"""

include("../../src/shared_utils/config.jl")

using JLD2
using Printf
using Statistics

(; outputdir) = load_project_config()
(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(;
    VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER,
)

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "2"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
probe_root = isempty(gpu_tag) ?
    joinpath(outputdir, "standardrun", model_config, "probe") :
    joinpath(outputdir, "standardrun", model_config, gpu_tag, "probe")

acm_suffix = lowercase(get(ENV, "ACTIVE_CELLS_MAP", "yes")) == "no" ? "_noACM" : ""
nsteps = parse(Int, get(ENV, "PROBE_NSTEPS", "1"))
ranks = (px == 1 && py == 1) ? [-1] : collect(0:(px * py - 1))

# The interior coefficient arrays carry no halos (size Nx × Ny × Nz on the
# rank-local grid). Interior j=1 IS the seam row on rank 1.
const SEAM_INTERIOR_J = 1

function coeffs_path(device, iter, rank)
    suffix = rank < 0 ? "" : "_rank$(rank)"
    return joinpath(
        probe_root,
        "probe_implicit_coeffs_$(device)_iter$(iter)$(suffix)$(acm_suffix).jld2",
    )
end

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

function row_diff(a, b, j)
    ndims(a) == 3 || return nothing
    size(a, 2) >= j || return nothing
    return diff_summary(a[:, j, :], b[:, j, :])
end

@info "compare_implicit_coeffs: probe_root = $probe_root"
@info "compare_implicit_coeffs: PROBE_NSTEPS=$nsteps PARTITION=$(px)x$(py) ACM_suffix='$acm_suffix'"

for step_iter in 0:(nsteps - 1), r in ranks
    cpu_path = coeffs_path("cpu", step_iter, r)
    gpu_path = coeffs_path("gpu", step_iter, r)
    if !isfile(cpu_path) || !isfile(gpu_path)
        @warn "missing coefficient dump" step_iter rank = r cpu = cpu_path gpu = gpu_path
        continue
    end

    rank_label = r < 0 ? "serial" : "rank$(r)"
    println("\n──── step $step_iter→$(step_iter + 1), implicit-coeffs, $rank_label ────")
    println("  CPU: $cpu_path")
    println("  GPU: $gpu_path")

    cpu = jldopen(cpu_path, "r") do f
        Dict("a" => Float64.(f["a"]), "b" => Float64.(f["b"]), "c" => Float64.(f["c"]))
    end
    gpu = jldopen(gpu_path, "r") do f
        Dict("a" => Float64.(f["a"]), "b" => Float64.(f["b"]), "c" => Float64.(f["c"]))
    end

    for k in ("a", "b", "c")
        ca = cpu[k]; ga = gpu[k]
        size(ca) == size(ga) || (@warn "shape mismatch" k size_cpu = size(ca) size_gpu = size(ga); continue)
        s = diff_summary(ca, ga)
        @printf "  %s  shape=%-20s GLOBAL: max|diff|=%.6e  n_differ=%d / %d\n" k string(size(ca)) s.max_abs s.n_differ s.n_finite
        if r == 1
            seam = row_diff(ca, ga, SEAM_INTERIOR_J)
            if seam !== nothing
                @printf "       → seam row interior j=%d (parent y=14): max|diff|=%.6e  n_differ=%d / %d\n" SEAM_INTERIOR_J seam.max_abs seam.n_differ seam.n_finite
            end
        end
    end
end

println("\ncompare_implicit_coeffs done.")
