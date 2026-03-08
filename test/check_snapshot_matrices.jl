"""
Regression test: compare snapshot and averaged transport matrices against archived reference.

Loads each archived snapshot matrix and the corresponding newly-created one,
asserts identical sparsity patterns (colptr, rowval), and reports the maximum
absolute difference of nzval.  Repeats for the averaged matrices (avg12a,
avg12b, avg24).

Usage:
```
julia --project test/check_snapshot_matrices.jl [PARENT_MODEL]
```

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER (same as create_snapshot_matrices.jl)
"""

using SparseArrays
using LinearAlgebra
using JLD2
using Printf
using TOML

using Oceananigans.Units: second, seconds, day, days
year = years = 365.25days

include(joinpath(@__DIR__, "..", "src", "shared_functions.jl"))

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
else
    outputdir = profile["outputdir"]
end

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

repo_root = normpath(joinpath(@__DIR__, ".."))
matrices_dir = joinpath(outputdir, "TM", model_config)
archive_dir = joinpath(repo_root, "archive", "outputs", parentmodel, "TM", model_config)

@info "Regression test configuration"
@info "- PARENT_MODEL = $parentmodel"
@info "- model_config = $model_config"
@info "- matrices_dir = $matrices_dir"
@info "- archive_dir  = $archive_dir"
flush(stdout); flush(stderr)

isdir(archive_dir) || error("Archive directory not found: $archive_dir")

################################################################################
# Helper
################################################################################

function compare_matrices(label, file_new, file_ref)
    if !isfile(file_new)
        @warn "  [$label] New file not found: $file_new — SKIPPED"
        return false
    end
    if !isfile(file_ref)
        @warn "  [$label] Reference file not found: $file_ref — SKIPPED"
        return false
    end

    M_new = load(file_new, "M")
    M_ref = load(file_ref, "M")

    # Check dimensions
    if size(M_new) != size(M_ref)
        @error "  [$label] Size mismatch: new=$(size(M_new)), ref=$(size(M_ref))"
        return false
    end

    # Check sparsity pattern
    pattern_ok = M_new.colptr == M_ref.colptr && M_new.rowval == M_ref.rowval
    if !pattern_ok
        @error "  [$label] Sparsity pattern mismatch (colptr or rowval differ)"
        return false
    end

    # Compare values
    max_abs_diff = maximum(abs, M_new.nzval .- M_ref.nzval)
    frob_rel = norm(M_new.nzval .- M_ref.nzval) / max(norm(M_ref.nzval), eps())

    if max_abs_diff == 0.0
        @info "  [$label] PASS — identical (max|diff|=0)"
    elseif max_abs_diff < 1.0e-12
        @info "  [$label] PASS — near-identical (max|diff|=$(@sprintf("%.2e", max_abs_diff)), rel_frob=$(@sprintf("%.2e", frob_rel)))"
    else
        @warn "  [$label] DIFF — max|diff|=$(@sprintf("%.2e", max_abs_diff)), rel_frob=$(@sprintf("%.2e", frob_rel))"
    end
    flush(stdout); flush(stderr)
    return true
end

################################################################################
# Compare snapshot matrices
################################################################################

@info "="^72
@info "Comparing snapshot matrices"
@info "="^72
flush(stdout); flush(stderr)

snap_dir_new = joinpath(matrices_dir, "snapshots")
snap_dir_ref = joinpath(archive_dir, "snapshots")

if isdir(snap_dir_ref)
    ref_files = sort([f for f in readdir(snap_dir_ref) if startswith(f, "M_snapshot_") && endswith(f, ".jld2")])
    @info "Found $(length(ref_files)) archived snapshot matrices"
    for f in ref_files
        compare_matrices(f, joinpath(snap_dir_new, f), joinpath(snap_dir_ref, f))
    end
else
    @warn "No archived snapshot directory: $snap_dir_ref"
end
flush(stdout); flush(stderr)

################################################################################
# Compare averaged matrices
################################################################################

@info "="^72
@info "Comparing averaged matrices"
@info "="^72
flush(stdout); flush(stderr)

for label in ["const", "avg12a", "avg12b", "avg24"]
    file_new = joinpath(matrices_dir, label, "M.jld2")
    file_ref = joinpath(archive_dir, label, "M.jld2")
    compare_matrices(label, file_new, file_ref)
end

@info "="^72
@info "Regression test complete"
flush(stdout); flush(stderr)
