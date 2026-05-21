"""
Test that the production 1D wet-cell scatter/gather (src/mpi_partition_io.jl)
agrees bitwise with an independently-implemented reference that round-trips
through a full Nx·Ny·Nz Float64 buffer using only direct MPI primitives.

Production (1D, hot path):
    scatter!(age_local, age_global, arch)
    gather! (age_global, age_local, arch)
    — permutation-based MPI.Scatterv!/Gatherv! on 1D wet-cell vectors.

Reference (3D, this file):
    scatter_manual!(...) / gather_manual!(...)
    — broadcast/all-reduce a full Nx·Ny·Nz buffer on every rank, then each
      rank slices its y-stripe and extracts wet cells via its local idx.
      Uses only MPI.Bcast! and MPI.Allreduce! on contiguous Arrays — no
      Oceananigans distributed primitives are involved.

Each rank fills the global vector with unique integer labels 1..Nidx_global
(stored as Float64) so forward agreement and uniqueness can be checked
bitwise. The round-trip must reproduce 1..Nidx_global on rank 0.

Env vars:
    PARENT_MODEL, EXPERIMENT, TIME_WINDOW – standard
    PARTITION_X, PARTITION_Y              – must satisfy px*py == #ranks
    LOAD_BALANCE                          – no | surface | cell | mix | minmax
"""

using MPI
using Oceananigans
using Oceananigans.Architectures: CPU, AbstractArchitecture
using Oceananigans.DistributedComputations: Distributed, Partition, Sizes
using JLD2
using Printf

include("../src/shared_functions.jl")
include("../src/shared_utils/load_balance.jl")

MPI.Initialized() || MPI.Init()
COMM = MPI.COMM_WORLD
rank = MPI.Comm_rank(COMM)
nranks = MPI.Comm_size(COMM)

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

grid_local = load_tripolar_grid(grid_file, arch)
grid_global = load_tripolar_grid(grid_file, CPU())

local_mask = compute_wet_mask(grid_local)
wet3D_local = local_mask.wet3D
idx_local = local_mask.idx
Nidx_local = local_mask.Nidx

# Global wet mask on every rank (small, identical, ~13 MB BitArray for OM2-1).
# Each rank can derive partition_y_sizes from arch.partition.y locally — no MPI.
gm_global = compute_wet_mask(grid_global)
wet3D_global = gm_global.wet3D
idx_global = gm_global.idx
Nidx_global = gm_global.Nidx
Nx_global, Ny_global, Nz_global = size(wet3D_global)

partition_y_sizes = collect(Oceananigans.DistributedComputations.local_sizes(Ny_global, arch.partition.y))
Ny_sum = sum(partition_y_sizes)
Ny_sum ≠ Ny_global && (partition_y_sizes[end] += Ny_global - Ny_sum)

# Per-rank y-offset into the global grid (0-based first row)
y_rank = arch.local_index[2]  # 1-based
y_offset_local = sum(view(partition_y_sizes, 1:(y_rank - 1)))

rank == 0 && @info "Global setup" Nidx_global Nx_global Ny_global Nz_global partition_y_sizes
@info "[rank $rank] local slab" Nidx_local Ny_local = partition_y_sizes[y_rank] y_offset_local

# Production scatter!/gather! and build_global_permutation. After this include,
# the production methods reference our module-scope globals (perm, send_buf,
# recv_buf, counts, displs, Nidx_global, rank, COMM).
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

# ─── Manual MPI-only reference (step 2b) ──────────────────────────────────

"""
Scatter via a global Nx·Ny·Nz Float64 buffer broadcast from rank 0; each
rank then slices its own y-stripe and extracts wet cells.
"""
function scatter_manual!(
        age_local, age_global_, arch::Distributed,
        Nx, Ny, Nz, idx_global_, idx_local_, y_offset, Ny_local
    )
    buf = zeros(Float64, Nx, Ny, Nz)
    if rank == 0
        buf[idx_global_] .= age_global_
    end
    MPI.Bcast!(buf, 0, COMM)
    slab = view(buf, :, (y_offset + 1):(y_offset + Ny_local), :)
    age_local .= view(slab, idx_local_)
    return age_local
end

"""
Gather via an Allreduce-summed global buffer: each rank places its 1D
local age into a zero-padded global buffer at its y-offset; Allreduce-sum
collects them on every rank; rank 0 extracts wet cells via idx_global.
"""
function gather_manual!(
        age_global_, age_local, arch::Distributed,
        Nx, Ny, Nz, idx_global_, idx_local_, y_offset, Ny_local
    )
    buf = zeros(Float64, Nx, Ny, Nz)
    slab = view(buf, :, (y_offset + 1):(y_offset + Ny_local), :)
    slab[idx_local_] .= age_local
    MPI.Allreduce!(buf, +, COMM)
    if rank == 0
        age_global_ .= view(buf, idx_global_)
    end
    return age_global_
end

# ─── Run the tests ─────────────────────────────────────────────────────────

age_local_A = Vector{Float64}(undef, Nidx_local)
age_local_B = Vector{Float64}(undef, Nidx_local)
age_global_input = rank == 0 ? Float64.(1:Nidx_global) : Float64[]

Ny_local_r = partition_y_sizes[y_rank]

# ── Test 1: Forward scatter ──
rank == 0 && (@info "TEST 1: forward scatter (production)"; flush(stdout))
fill!(age_local_A, NaN)
scatter!(age_local_A, age_global_input, arch)
MPI.Barrier(COMM)

rank == 0 && (@info "TEST 1: forward scatter (reference)"; flush(stdout))
fill!(age_local_B, NaN)
scatter_manual!(
    age_local_B, age_global_input, arch,
    Nx_global, Ny_global, Nz_global, idx_global, idx_local, y_offset_local, Ny_local_r
)
MPI.Barrier(COMM)

agree_local = age_local_A == age_local_B
local_int = Int.(age_local_A)
in_range_ok = all(1 .<= local_int .<= Nidx_global)
distinct_ok = length(Set(local_int)) == Nidx_local

for r in 0:(nranks - 1)
    if rank == r
        println("[rank $r] TEST 1 forward: agree=$agree_local in_range=$in_range_ok distinct=$distinct_ok (Nidx_local=$Nidx_local)")
        flush(stdout)
    end
    MPI.Barrier(COMM)
end

agree_count = MPI.Allreduce(agree_local ? 1 : 0, MPI.SUM, COMM)
in_range_count_all = MPI.Allreduce(in_range_ok ? 1 : 0, MPI.SUM, COMM)
distinct_count_all = MPI.Allreduce(distinct_ok ? 1 : 0, MPI.SUM, COMM)
forward_all_pass = (agree_count == nranks) && (in_range_count_all == nranks) && (distinct_count_all == nranks)

# Cross-rank uniqueness: the union of labels across ranks must equal 1..Nidx_global.
# Reduce per-rank Set sizes by Allreduce-sum; total must equal Nidx_global.
local_set_size = length(Set(local_int))
total_distinct = MPI.Allreduce(local_set_size, MPI.SUM, COMM)
union_ok = total_distinct == Nidx_global  # since each Set is distinct and union must be 1:Nidx_global

# ── Test 2: Round-trip gather ──
rank == 0 && (@info "TEST 2: gather (production)"; flush(stdout))
age_global_A = rank == 0 ? Vector{Float64}(undef, Nidx_global) : Float64[]
fill!(age_global_A, NaN)
gather!(age_global_A, age_local_A, arch)
MPI.Barrier(COMM)

rank == 0 && (@info "TEST 2: gather (reference)"; flush(stdout))
age_global_B = rank == 0 ? Vector{Float64}(undef, Nidx_global) : Float64[]
fill!(age_global_B, NaN)
gather_manual!(
    age_global_B, age_local_B, arch,
    Nx_global, Ny_global, Nz_global, idx_global, idx_local, y_offset_local, Ny_local_r
)
MPI.Barrier(COMM)

roundtrip_A = false
roundtrip_B = false
roundtrip_AB = false
if rank == 0
    expected = Float64.(1:Nidx_global)
    roundtrip_A = age_global_A == expected
    roundtrip_B = age_global_B == expected
    roundtrip_AB = age_global_A == age_global_B
    @info "TEST 2 round-trip" production_matches_input = roundtrip_A reference_matches_input = roundtrip_B prod_eq_ref = roundtrip_AB
    roundtrip_A || @info "  production diffs: $(count(age_global_A .≠ expected)) / $Nidx_global cells"
    roundtrip_B || @info "  reference  diffs: $(count(age_global_B .≠ expected)) / $Nidx_global cells"
end
roundtrip_pass_int = MPI.bcast(Int(roundtrip_A && roundtrip_B && roundtrip_AB), 0, COMM)
roundtrip_pass = roundtrip_pass_int == 1
MPI.Barrier(COMM)

# ── Summary ──
overall = forward_all_pass && union_ok && roundtrip_pass
if rank == 0
    @info "===== scatter/gather test summary =====" forward_all_pass union_ok roundtrip_pass overall
    overall || error("scatter/gather test FAILED")
    @info "✓ All assertions passed"
end

MPI.Barrier(COMM)
MPI.Finalize()
