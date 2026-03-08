"""
Verify self-consistency of the source_rate toggle from test_nolinphi.jl outputs.

Usage:
    julia --project test/test_remove_linphi.jl TEST_DIR

Checks:
1. Determinism: det_src1_a ≈ det_src1_b (vol_norm tolerance)
2. Source toggle: toggle_with_src ≠ toggle_no_src, and max(with_src) > max(no_src)
3. Repeatability: toggle_with_src ≈ toggle_with_src_again (vol_norm tolerance)
"""

using JLD2
using LinearAlgebra: dot, Diagonal
using Statistics: mean

if length(ARGS) < 1
    error("Usage: julia --project test/test_remove_linphi.jl TEST_DIR")
end

test_dir = ARGS[1]
all_pass = true
TOLERANCE_YEARS = 1.0e-6  # vol_norm tolerance in years

################################################################################
# Load metadata and build vol_norm
################################################################################

metadata_path = joinpath(test_dir, "metadata.jld2")
if !isfile(metadata_path)
    error("Missing metadata file: $metadata_path")
end

v1D = load(metadata_path, "v1D")
year = load(metadata_path, "year")

inv_sumv = 1 / sum(v1D)
vol_norm(x) = sqrt(dot(x, Diagonal(v1D), x) * inv_sumv) / year

@info "Loaded metadata" n_wet_cells = length(v1D) year tolerance_years = TOLERANCE_YEARS

################################################################################
# Helper
################################################################################

function load_age(dir, name)
    fpath = joinpath(dir, "$name.jld2")
    if !isfile(fpath)
        @error "Missing file: $fpath"
        return nothing
    end
    return load(fpath, "age")
end

################################################################################
# Test 1: Determinism
################################################################################

@info "=== Test 1: Determinism ==="
det_a = load_age(test_dir, "det_src1_a")
det_b = load_age(test_dir, "det_src1_b")

if det_a !== nothing && det_b !== nothing
    diff = det_a .- det_b
    vn = vol_norm(diff)
    @info "Determinism diff stats" vol_norm_years = vn max_diff_years = maximum(abs, diff) / year mean_diff_years = mean(abs, diff) / year
    if vn < TOLERANCE_YEARS
        @info "PASS  Φ!(zeros; source_rate=1.0) is deterministic (vol_norm = $(vn) yr < $(TOLERANCE_YEARS) yr)"
    else
        @error "FAIL  Φ!(zeros; source_rate=1.0) is NOT deterministic" vol_norm_years = vn tolerance = TOLERANCE_YEARS
        global all_pass = false
    end
else
    global all_pass = false
end

################################################################################
# Test 2: Source toggle produces different results
################################################################################

@info "=== Test 2: Source toggle ==="
with_src = load_age(test_dir, "toggle_with_src")
no_src = load_age(test_dir, "toggle_no_src")

if with_src !== nothing && no_src !== nothing
    if with_src != no_src
        @info "PASS  source_rate=1.0 and source_rate=0.0 give different results"
    else
        @error "FAIL  source_rate=1.0 and source_rate=0.0 give identical results — toggle not working!"
        global all_pass = false
    end

    if maximum(abs, with_src) > maximum(abs, no_src)
        @info "PASS  max(age) with source > without" max_with_years = maximum(abs, with_src) / year max_without_years = maximum(abs, no_src) / year
    else
        @error "FAIL  max(age) with source should be greater than without" max_with_years = maximum(abs, with_src) / year max_without_years = maximum(abs, no_src) / year
        global all_pass = false
    end

    diff = with_src .- no_src
    @info "Source contribution stats" vol_norm_years = vol_norm(diff) max_diff_years = maximum(abs, diff) / year mean_diff_years = mean(abs, diff) / year
else
    global all_pass = false
end

################################################################################
# Test 3: Repeatability after toggle
################################################################################

@info "=== Test 3: Repeatability after toggle ==="
with_src_again = load_age(test_dir, "toggle_with_src_again")

if with_src !== nothing && with_src_again !== nothing
    diff = with_src .- with_src_again
    vn = vol_norm(diff)
    @info "Repeatability diff stats" vol_norm_years = vn max_diff_years = maximum(abs, diff) / year mean_diff_years = mean(abs, diff) / year
    if vn < TOLERANCE_YEARS
        @info "PASS  Φ!(input; source_rate=1.0) is repeatable after toggle (vol_norm = $(vn) yr < $(TOLERANCE_YEARS) yr)"
    else
        @error "FAIL  Φ!(input; source_rate=1.0) changed after toggling to 0.0 and back" vol_norm_years = vn tolerance = TOLERANCE_YEARS
        global all_pass = false
    end
else
    global all_pass = false
end

################################################################################
# Summary
################################################################################

if all_pass
    @info "All tests PASSED — source_rate toggle is correct and deterministic (within GPU FP tolerance)"
else
    @error "Some tests FAILED — see above"
end
