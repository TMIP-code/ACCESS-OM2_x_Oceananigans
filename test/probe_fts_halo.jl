"""
Probe the FTS halo immediately after model construction — before any
timestep, before any halo fill.

If GPU rank 1's south halo of `v_ts` is all-zeros (or differs systematically
from CPU's), the FTS loader on GPU is the culprit for the seam tracer bug.
If GPU matches CPU at this point, the FTS loader is fine and the bug
must be downstream.

Usage:
  # CPU 1x2:
  mpiexec -n 2 julia --project test/probe_fts_halo.jl
  # GPU 1x2 (via PBS): same script under the run_diagnostic_steps.sh wrapper
  # or any wrapper that sets PARTITION=1x2 and PBS_NGPUS=2.
"""

include("../src/setup_model.jl")

using Printf

if arch isa Distributed
    using MPI
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
else
    rank = 0
end

device_str = (arch isa Distributed ? (child_architecture(arch) isa CPU ? "CPU" : "GPU") : (arch isa CPU ? "CPU" : "GPU"))
@info "PROBE_FTS: rank=$rank arch_str=$arch_str device=$device_str"

function describe(name, fts)
    # parent(data) for FieldTimeSeries: 4-D OffsetArray (i, j, k, t)
    # Bring to CPU for inspection
    p_gpu = parent(fts.data)
    p = Array(p_gpu)   # CPU copy
    @info "PROBE_FTS: rank=$rank $name parent size=$(size(p_gpu)) eltype=$(eltype(p_gpu))"
    Hx = 13
    Hy = 13

    # Probe rank's south halo (parent y = 1..Hy), snapshot 1, at the surface k
    if size(p, 1) > 0 && size(p, 2) >= Hy && size(p, 3) > 0 && size(p, 4) >= 1
        n_k = size(p, 3)
        k_surf = n_k - 7   # near-surface, away from outer z halo
        south_halo_strip = p[:, 1:Hy, k_surf, 1]
        flat = filter(isfinite, vec(south_halo_strip))
        @info @sprintf(
            "PROBE_FTS: rank=%d %s south halo (y=1:%d, k=%d, t=1) min=%.6e max=%.6e mean(abs)=%.6e n_zero=%d/%d",
            rank, name, Hy, k_surf,
            isempty(flat) ? NaN : minimum(flat),
            isempty(flat) ? NaN : maximum(flat),
            isempty(flat) ? NaN : sum(abs, flat) / length(flat),
            count(==(0), flat), length(flat)
        )

        # Print the row immediately adjacent to the interior (parent y = Hy)
        adj_row = p[:, Hy, k_surf, 1]
        @info @sprintf(
            "PROBE_FTS: rank=%d %s seam-adjacent halo row y=%d k=%d t=1: min=%.6e max=%.6e mean(abs)=%.6e n_zero=%d/%d",
            rank, name, Hy, k_surf,
            minimum(adj_row), maximum(adj_row), sum(abs, adj_row) / length(adj_row),
            count(==(0), adj_row), length(adj_row)
        )

        # And the first interior row (y = Hy+1)
        first_int_row = p[:, Hy + 1, k_surf, 1]
        @info @sprintf(
            "PROBE_FTS: rank=%d %s first interior row y=%d k=%d t=1: min=%.6e max=%.6e mean(abs)=%.6e n_zero=%d/%d",
            rank, name, Hy + 1, k_surf,
            minimum(first_int_row), maximum(first_int_row), sum(abs, first_int_row) / length(first_int_row),
            count(==(0), first_int_row), length(first_int_row)
        )
    end
    flush(stdout); flush(stderr)
    return nothing
end

@info "PROBE_FTS: rank=$rank inspecting u_ts"
describe("u_ts", u_ts)
@info "PROBE_FTS: rank=$rank inspecting v_ts"
describe("v_ts", v_ts)
if @isdefined(η_ts)
    @info "PROBE_FTS: rank=$rank inspecting η_ts"
    describe("eta_ts", η_ts)
end

@info "PROBE_FTS: rank=$rank done. Exiting without time_step."
flush(stdout); flush(stderr)

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
