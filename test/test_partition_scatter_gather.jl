"""
Test that the production 1D wet-cell scatter/gather (src/mpi_partition_io.jl)
agrees bitwise with an Oceananigans full-3D-field reference round-trip.

Production:
    scatter!(age_local, age_global, arch)
    gather! (age_global, age_local, arch)
    — permutation-based MPI.Scatterv!/Gatherv! on 1D vectors (NK hot path).

Reference (this file, step 2b of the partitioned-NK plan):
    scatter_oceananigans!(...) / gather_oceananigans!(...)
    — full Nx·Ny·Nz Float64 field round-trip via
      set!(::DistributedField, ::Field) and reconstruct_global_field.

Each rank fills the global vector with unique integer labels 1..Nidx_global
(stored as Float64) so forward agreement and uniqueness can be checked
bitwise. Then the round-trip must reproduce 1..Nidx_global on rank 0.

Env vars:
    PARENT_MODEL, EXPERIMENT, TIME_WINDOW – standard
    PARTITION_X, PARTITION_Y              – must satisfy px*py == #ranks
    LOAD_BALANCE                          – no | surface | cell | mix | minmax
"""

using MPI
using Oceananigans
using Oceananigans.Architectures: CPU, AbstractArchitecture
using Oceananigans.DistributedComputations: Distributed, Partition, Sizes
using Oceananigans.Fields: CenterField, interior
import Oceananigans.Fields: set!
using JLD2
using Printf

include("../src/shared_functions.jl")
# parse_load_balance_env + compute_lb_y_sizes live here; the include is
# idempotent if shared_functions already pulled it in.
include("../src/shared_utils/load_balance.jl")

MPI.Initialized() || MPI.Init()
COMM = MPI.COMM_WORLD
rank = MPI.Comm_rank(COMM)
nranks = MPI.Comm_size(COMM)

# Configuration
px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "2"))
@assert px == 1 "PARTITION_X must be 1 (got $px)"
@assert px * py == nranks "px*py = $(px * py) but mpiexec launched $nranks ranks"

(LB_ACTIVE, LB_METHOD, _) = parse_load_balance_env("LOAD_BALANCE")

(; experiment_dir) = load_project_config()
grid_file = joinpath(experiment_dir, "grid.jld2")

if LB_ACTIVE
    Hy_grid = load(grid_file, "Hy")
    local_Ny_lb = compute_lb_y_sizes(grid_file, py; method = LB_METHOD, min_size = Hy_grid + 2)
    rank == 0 && @info "LB partition: local_Ny=$local_Ny_lb (method=$LB_METHOD)"
    arch = Distributed(CPU(); partition = Partition(y = Sizes(local_Ny_lb...)))
else
    rank == 0 && @info "Even partition: Partition($px, $py)"
    arch = Distributed(CPU(); partition = Partition(px, py))
end

rank == 0 && flush(stdout)
MPI.Barrier(COMM)

# Load both grids
grid_local = load_tripolar_grid(grid_file, arch)
grid_global = load_tripolar_grid(grid_file, CPU())

local_mask = compute_wet_mask(grid_local)
wet3D_local = local_mask.wet3D
idx_local = local_mask.idx
Nidx_local = local_mask.Nidx

# Global mask + partition_y_sizes (rank 0 only)
if rank == 0
    gm = compute_wet_mask(grid_global)
    wet3D_global = gm.wet3D
    idx_global = gm.idx
    Nidx_global = gm.Nidx
    Ny_global = size(wet3D_global, 2)
    partition_y_sizes = collect(Oceananigans.DistributedComputations.local_sizes(Ny_global, arch.partition.y))
    Ny_sum = sum(partition_y_sizes)
    Ny_sum != Ny_global && (partition_y_sizes[end] += Ny_global - Ny_sum)
    @info "[rank 0] Global setup" Nidx_global Ny_global partition_y_sizes
else
    wet3D_global = nothing
    idx_global = nothing
    Nidx_global = 0
    partition_y_sizes = Int[]
end

# Pull in production scatter!/gather! and build_global_permutation.
# After this include, the production methods reference our module-scope
# globals (perm, send_buf, recv_buf, counts, displs, Nidx_global, rank, COMM).
include("../src/mpi_partition_io.jl")

if rank == 0
    perm, counts, displs = build_global_permutation(wet3D_global, partition_y_sizes)
    send_buf = Vector{Float64}(undef, Nidx_global)
    recv_buf = Vector{Float64}(undef, Nidx_global)
    @info "[rank 0] Permutation built" counts displs
else
    perm = Int[]
    counts = Int[]
    displs = Int[]
    send_buf = Float64[]
    recv_buf = Float64[]
end

MPI.Barrier(COMM)

# ─── Oceananigans reference (step 2b) ──────────────────────────────────────

function scatter_oceananigans!(
        age_local, age_global_, arch::Distributed,
        grid_global_, grid_local_, idx_global_, idx_local_
    )
    f_global = CenterField(grid_global_)
    if rank == 0
        fi = interior(f_global)
        fill!(fi, 0)
        fi[idx_global_] .= age_global_
    end
    f_local = CenterField(grid_local_)
    set!(f_local, f_global)
    age_local .= view(interior(f_local), idx_local_)
    return age_local
end

function gather_oceananigans!(
        age_global_, age_local, arch::Distributed,
        grid_local_, idx_global_, idx_local_
    )
    f_local = CenterField(grid_local_)
    fill!(interior(f_local), 0)
    interior(f_local)[idx_local_] .= age_local
    f_global = Oceananigans.DistributedComputations.reconstruct_global_field(f_local)
    if rank == 0
        age_global_ .= view(interior(f_global), idx_global_)
    end
    return age_global_
end

# ─── Run the tests ─────────────────────────────────────────────────────────

# Allocate test buffers
age_local_A = Vector{Float64}(undef, Nidx_local)
age_local_B = Vector{Float64}(undef, Nidx_local)
age_global_input = rank == 0 ? Float64.(1:Nidx_global) : Float64[]

# ── Test 1: Forward scatter (production vs reference) ──
rank == 0 && (@info "TEST 1: forward scatter (production)"; flush(stdout))
fill!(age_local_A, NaN)
scatter!(age_local_A, age_global_input, arch)
MPI.Barrier(COMM)

rank == 0 && (@info "TEST 1: forward scatter (Oceananigans reference)"; flush(stdout))
fill!(age_local_B, NaN)
scatter_oceananigans!(age_local_B, age_global_input, arch, grid_global, grid_local, idx_global, idx_local)
MPI.Barrier(COMM)

agree_local = age_local_A == age_local_B
# Uniqueness/range check per rank (catches scatter overlap if both sides have the same bug)
local_int = Int.(age_local_A)
in_range_count = count(x -> 1 <= x <= sum(counts), local_int)
distinct_count = length(Set(local_int))
in_range_ok = in_range_count == Nidx_local
distinct_ok = distinct_count == Nidx_local

# Pretty per-rank report (serialized via barriers)
for r in 0:(nranks - 1)
    if rank == r
        Nidx_global_check = rank == 0 ? Nidx_global : 0
        println("[rank $r] TEST 1 forward: agree=$agree_local in_range=$in_range_ok distinct=$distinct_ok (Nidx_local=$Nidx_local)")
        flush(stdout)
    end
    MPI.Barrier(COMM)
end

agree_count = MPI.Allreduce(agree_local ? 1 : 0, MPI.SUM, COMM)
in_range_count_all = MPI.Allreduce(in_range_ok ? 1 : 0, MPI.SUM, COMM)
distinct_count_all = MPI.Allreduce(distinct_ok ? 1 : 0, MPI.SUM, COMM)
forward_all_pass = (agree_count == nranks) && (in_range_count_all == nranks) && (distinct_count_all == nranks)

# ── Test 2: Round-trip gather (global → local → global) ──
rank == 0 && (@info "TEST 2: gather (production)"; flush(stdout))
age_global_A = rank == 0 ? Vector{Float64}(undef, Nidx_global) : Float64[]
fill!(age_global_A, NaN)
gather!(age_global_A, age_local_A, arch)
MPI.Barrier(COMM)

rank == 0 && (@info "TEST 2: gather (Oceananigans reference)"; flush(stdout))
age_global_B = rank == 0 ? Vector{Float64}(undef, Nidx_global) : Float64[]
fill!(age_global_B, NaN)
gather_oceananigans!(age_global_B, age_local_B, arch, grid_local, idx_global, idx_local)
MPI.Barrier(COMM)

roundtrip_pass = true
if rank == 0
    expected = Float64.(1:Nidx_global)
    A_ok = age_global_A == expected
    B_ok = age_global_B == expected
    AB_ok = age_global_A == age_global_B
    roundtrip_pass = A_ok && B_ok && AB_ok
    @info "TEST 2 round-trip" production_matches_input = A_ok reference_matches_input = B_ok prod_eq_ref = AB_ok
    A_ok || @info "  production diffs: $(count(age_global_A .!= expected)) / $Nidx_global cells"
    B_ok || @info "  reference  diffs: $(count(age_global_B .!= expected)) / $Nidx_global cells"
end
roundtrip_pass = MPI.bcast(roundtrip_pass, 0, COMM)
MPI.Barrier(COMM)

# ── Summary ──
overall = forward_all_pass && roundtrip_pass
if rank == 0
    @info "===== scatter/gather test summary =====" forward_all_pass roundtrip_pass overall
    overall || error("scatter/gather test FAILED")
    @info "✓ All assertions passed"
end

MPI.Barrier(COMM)
MPI.Finalize()
