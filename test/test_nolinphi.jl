"""
Self-consistency test for the single-tracer source_rate toggle.

Verifies that toggling `source_rate` between 1.0 and 0.0 within the same model
produces correct and deterministic results:

1. Determinism: Φ!(zeros; source_rate=1.0) called twice gives bitwise identical output
2. Source toggle: Φ!(nonzero; source_rate=1.0) ≠ Φ!(nonzero; source_rate=0.0)
3. Repeatability: Φ!(nonzero; source_rate=1.0) after a source_rate=0.0 call gives
   bitwise identical output to the first source_rate=1.0 call
"""

include(joinpath(@__DIR__, "..", "src", "setup_model.jl"))

using LinearAlgebra: norm
using Oceananigans.Simulations: reset!

################################################################################
# Simulation, wet mask, buffers
################################################################################

set!(model, age = Returns(0.0))

simulation = Simulation(model; Δt, stop_time)
add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: $Nidx"
Nx′, Ny′, Nz′ = size(wet3D)
flush(stdout); flush(stderr)

age3D_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))

# Cell volumes for vol_norm (saved for comparison script)
grid_cpu = on_architecture(CPU(), grid)
v1D = interior(compute_volume(grid_cpu))[idx]

################################################################################
# Φ! with source_rate toggle
################################################################################

call_count = Ref(0)

function Φ!(age_out, age_in; source_rate = 1.0)
    call_count[] += 1
    call_num = call_count[]
    t_start = time()
    @info "Φ! call #$call_num starting (source_rate=$source_rate)" norm_age_years = norm(age_in) / year max_age_years = maximum(abs, age_in) / year
    flush(stdout); flush(stderr)

    copyto!(source_rate_arr, [source_rate])

    reset!(simulation)
    simulation.stop_time = stop_time

    fill!(age3D_cpu, 0)
    age3D_cpu[idx] .= age_in
    copyto!(age3D_gpu, age3D_cpu)
    set!(model, age = age3D_gpu)

    run!(simulation)

    copyto!(age3D_cpu, interior(model.tracers.age))
    age_out .= view(age3D_cpu, idx)

    elapsed = time() - t_start
    @info "Φ! call #$call_num done ($(round(elapsed; digits = 1))s)"
    flush(stdout); flush(stderr)
    return age_out
end

################################################################################
# Self-consistency test
################################################################################

test_output_dir = get(ENV, "TEST_OUTPUT_DIR", joinpath(outputdir, "test_nolinphi"))
mkpath(test_output_dir)

# Save metadata for comparison script (vol_norm needs v1D and year)
jldsave(joinpath(test_output_dir, "metadata.jld2"); v1D = v1D, year = year)

in_zeros = zeros(Nidx)

@info "=== Test 1: Determinism (Φ!(zeros; source_rate=1.0) twice) ==="
flush(stdout); flush(stderr)

out_src1_a = zeros(Nidx)
Φ!(out_src1_a, in_zeros; source_rate = 1.0)
jldsave(joinpath(test_output_dir, "det_src1_a.jld2"); age = copy(out_src1_a))

out_src1_b = zeros(Nidx)
Φ!(out_src1_b, in_zeros; source_rate = 1.0)
jldsave(joinpath(test_output_dir, "det_src1_b.jld2"); age = copy(out_src1_b))

@info "=== Test 2: Source toggle (nonzero input, source_rate=1.0 vs 0.0) ==="
flush(stdout); flush(stderr)

# Use out_src1_a as nonzero input
out_with_src = zeros(Nidx)
Φ!(out_with_src, out_src1_a; source_rate = 1.0)
jldsave(joinpath(test_output_dir, "toggle_with_src.jld2"); age = copy(out_with_src))

out_no_src = zeros(Nidx)
Φ!(out_no_src, out_src1_a; source_rate = 0.0)
jldsave(joinpath(test_output_dir, "toggle_no_src.jld2"); age = copy(out_no_src))

@info "=== Test 3: Repeatability after toggle (source_rate=1.0 again) ==="
flush(stdout); flush(stderr)

out_with_src_again = zeros(Nidx)
Φ!(out_with_src_again, out_src1_a; source_rate = 1.0)
jldsave(joinpath(test_output_dir, "toggle_with_src_again.jld2"); age = copy(out_with_src_again))

@info "Self-consistency test complete — outputs in $test_output_dir"
flush(stdout); flush(stderr)
