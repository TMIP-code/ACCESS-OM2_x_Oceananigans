"""
MWE: fill_halo_regions! BoundsError for Face-located fields on distributed tripolar grids.

This is a self-contained test that only depends on Oceananigans, MPI, and JLD2.
It can be run against any Oceananigans branch to verify the bug / a fix.

Run with 4 CPUs (no GPUs):
    mpiexec -n 4 julia --project test/test_distributed_halo_fill.jl

Run with 4 GPUs:
    mpiexec --bind-to socket --map-by socket -n 4 julia --project test/test_distributed_halo_fill.jl

Expected: fill_halo_regions! should work for all (LX, LY, LZ) locations.
Observed: BoundsError in _fill_north_send_buffer! for Face-located fields on 2D partitions.

Root cause: y_communication_buffer() in communication_buffers.jl uses `Nx = size(grid, 1)`
(interior size without location awareness) to size the TwoDBuffer. For Face-located fields,
the parent array has Nx+1 columns, so the buffer is too small → BoundsError.
The fold zipper code (distributed_zipper.jl) correctly uses `size(parent(data), 1)` for
the north fold rank, but non-fold ranks fall back to the buggy y_communication_buffer.
"""

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.Architectures: CPU
using JLD2
using Oceananigans.OutputWriters
using MPI

MPI.Init()

# --- Architecture setup ---
ngpus = parse(Int, get(ENV, "PBS_NGPUS", "0"))
if ngpus > 0
    using CUDA
    child_arch = GPU()
else
    child_arch = CPU()
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
@info "MPI rank $rank/$nranks, arch=$child_arch"

# Determine partition from number of ranks
px, py = if nranks == 4
    (2, 2)
elseif nranks == 2
    (1, 2)
else
    error("Expected 2 or 4 MPI ranks, got $nranks")
end

arch = Distributed(child_arch, partition = Partition(px, py))

# --- Grid setup (small, self-contained, no external files) ---
grid = TripolarGrid(
    arch, Float64;
    size = (60, 61, 10),
    z = (-1000, 0),
    halo = (7, 7, 7),
    first_pole_longitude = 75,
    north_poles_latitude = 55,
)

ibg = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -1000))

@info "Rank $rank: grid size = $(size(ibg)), topology = $(Oceananigans.Grids.topology(ibg))"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Test 1: fill_halo_regions! at all staggered locations
# =========================================================================

npass = 0
nfail = 0

locations = [
    (Center, Center, Center),
    (Face, Center, Center),
    (Center, Face, Center),
    (Center, Center, Face),
    (Face, Face, Center),
    (Face, Center, Face),
    (Center, Face, Face),
    (Face, Face, Face),
]

for loc in locations
    LX, LY, LZ = loc
    f = Field{LX, LY, LZ}(ibg)
    set!(f, (x, y, z) -> x + y + z)

    @info "Rank $rank: testing fill_halo_regions! at ($LX, $LY, $LZ)..."
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)

    try
        fill_halo_regions!(f)
        @info "Rank $rank: ($LX, $LY, $LZ) — PASS"
        npass += 1
    catch e
        @error "Rank $rank: ($LX, $LY, $LZ) — FAIL" exception = (e, catch_backtrace())
        nfail += 1
    end

    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 2: fill_halo_regions! on a FieldTimeSeries at Face location
# =========================================================================

@info "Rank $rank: testing fill_halo_regions! on FieldTimeSeries at (Center, Face, Center)..."
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

times = [0.0, 1.0, 2.0]
fts = FieldTimeSeries{Center, Face, Center}(ibg, times)
for n in eachindex(times)
    set!(fts[n], (x, y, z) -> x + y + z + n)
end

try
    fill_halo_regions!(fts)
    @info "Rank $rank: FTS (Center, Face, Center) — PASS"
    npass += 1
catch e
    @error "Rank $rank: FTS (Center, Face, Center) — FAIL" exception = (e, catch_backtrace())
    nfail += 1
end

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Test 3: JLD2OutputWriter with Face-located field (triggers fill_halo_regions!)
# =========================================================================

@info "Rank $rank: testing JLD2Writer with Face-located field..."
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

v_field = Field{Center, Face, Center}(ibg)
set!(v_field, (x, y, z) -> y + z)

model = HydrostaticFreeSurfaceModel(
    ibg;
    velocities = PrescribedVelocityFields(),
    tracers = (; age = CenterField(ibg)),
    free_surface = nothing,
)

simulation = Simulation(model; Δt = 1.0, stop_time = 1.0)

try
    simulation.output_writers[:v] = JLD2Writer(
        model, Dict("v" => v_field);
        schedule = TimeInterval(1.0),
        filename = "test_halofill_output_rank$(rank)",
        overwrite_existing = true,
        including = [],
    )
    run!(simulation)
    @info "Rank $rank: JLD2Writer with Face field — PASS"
    npass += 1
catch e
    @error "Rank $rank: JLD2Writer with Face field — FAIL" exception = (e, catch_backtrace())
    nfail += 1
end

# Cleanup
rm("test_halofill_output_rank$(rank).jld2"; force = true)

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Summary
# =========================================================================

@info "Rank $rank: $npass passed, $nfail failed ($(npass + nfail) total)"
flush(stdout); flush(stderr)
