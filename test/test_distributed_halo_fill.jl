"""
MWE: JLD2Writer hangs on distributed tripolar grids with prescribed velocity model.

This is a self-contained test that only depends on Oceananigans, MPI, and JLD2.
It can be run against any Oceananigans branch to verify the bug / a fix.

Run with 4 CPUs (no GPUs):
    mpiexec -n 4 julia --project test/test_distributed_halo_fill.jl

Run with 4 GPUs:
    mpiexec --bind-to socket --map-by socket -n 4 julia --project test/test_distributed_halo_fill.jl

Tests:
  1. fill_halo_regions! at all 8 staggered (LX, LY, LZ) locations
  2. fill_halo_regions! on a FieldTimeSeries at Face location
  3. JLD2Writer with a single Face-located Field (simple case)
  4. JLD2Writer with PrescribedVelocityFields + DiagnosticVerticalVelocity
     + PrescribedFreeSurface — outputs mixed field types (TSI + regular Field)
     This matches the setup that hangs in the actual simulation.

The key difference: model.velocities.u/v are TimeSeriesInterpolation (TSI),
model.velocities.w is a regular Field (from DiagnosticVerticalVelocity), and
fill_halo_regions!(::TimeSeriesInterpolation) is a no-op. The mix of no-op
and real MPI halo fills may cause rank desynchronization → deadlock.
"""

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: seconds
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
        global npass += 1
    catch e
        @error "Rank $rank: ($LX, $LY, $LZ) — FAIL" exception = (e, catch_backtrace())
        global nfail += 1
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
    global npass += 1
catch e
    @error "Rank $rank: FTS (Center, Face, Center) — FAIL" exception = (e, catch_backtrace())
    global nfail += 1
end

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Test 3: JLD2Writer with a single Face-located Field (simple case)
# =========================================================================

@info "Rank $rank: testing JLD2Writer with Face-located field..."
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

v_field = Field{Center, Face, Center}(ibg)
set!(v_field, (x, y, z) -> y + z)

model_simple = HydrostaticFreeSurfaceModel(
    ibg;
    velocities = PrescribedVelocityFields(),
    tracers = (; age = CenterField(ibg)),
    free_surface = nothing,
)

sim_simple = Simulation(model_simple; Δt = 1.0, stop_time = 1.0)

try
    sim_simple.output_writers[:v] = JLD2Writer(
        model_simple, Dict("v" => v_field);
        schedule = TimeInterval(1.0),
        filename = "test_halofill_simple_rank$(rank)",
        overwrite_existing = true,
        including = [],
    )
    run!(sim_simple)
    @info "Rank $rank: JLD2Writer simple — PASS"
    global npass += 1
catch e
    @error "Rank $rank: JLD2Writer simple — FAIL" exception = (e, catch_backtrace())
    global nfail += 1
end

rm("test_halofill_simple_rank$(rank).jld2"; force = true)

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Test 4: JLD2Writer with PrescribedVelocityFields + DiagnosticVerticalVelocity
#          + PrescribedFreeSurface (matches actual simulation setup)
# =========================================================================

@info "Rank $rank: testing JLD2Writer with prescribed velocities + diagnostic w..."
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# Create FieldTimeSeries for u, v, eta (mimics prescribed velocity model)
Δt_fts = 1.0
fts_times = [0.0, 1.0, 2.0, 3.0]

u_fts = FieldTimeSeries{Face, Center, Center}(ibg, fts_times)
v_fts = FieldTimeSeries{Center, Face, Center}(ibg, fts_times)
η_fts = FieldTimeSeries{Center, Center, Nothing}(ibg, fts_times)

for n in eachindex(fts_times)
    set!(u_fts[n], (x, y, z) -> 0.1 * cosd(y))
    set!(v_fts[n], (x, y, z) -> 0.0)
    set!(η_fts[n], (x, y) -> 0.0)
end

# Build model with prescribed u, v (→ TimeSeriesInterpolation), diagnostic w
model_prescribed = HydrostaticFreeSurfaceModel(
    ibg;
    velocities = PrescribedVelocityFields(
        u = u_fts, v = v_fts,
        formulation = DiagnosticVerticalVelocity()
    ),
    tracers = (; age = CenterField(ibg)),
    free_surface = PrescribedFreeSurface(displacement = η_fts),
)

# model.velocities.u/v are TimeSeriesInterpolation, model.velocities.w is a regular Field
@info "Rank $rank: u type = $(typeof(model_prescribed.velocities.u))"
@info "Rank $rank: v type = $(typeof(model_prescribed.velocities.v))"
@info "Rank $rank: w type = $(typeof(model_prescribed.velocities.w))"
@info "Rank $rank: η type = $(typeof(model_prescribed.free_surface.displacement))"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

sim_prescribed = Simulation(model_prescribed; Δt = Δt_fts, stop_time = 2 * Δt_fts)

# Output the same mix of field types as the actual simulation:
# age (CenterField), u (TSI), v (TSI), w (regular Field), eta (TSI)
output_fields = Dict(
    "age" => model_prescribed.tracers.age,
    "u" => model_prescribed.velocities.u,
    "v" => model_prescribed.velocities.v,
    "w" => model_prescribed.velocities.w,
    "eta" => model_prescribed.free_surface.displacement,
)

try
    sim_prescribed.output_writers[:fields] = JLD2Writer(
        model_prescribed, output_fields;
        schedule = TimeInterval(Δt_fts),
        filename = "test_halofill_prescribed_rank$(rank)",
        overwrite_existing = true,
        with_halos = true,
        including = [],
    )
    @info "Rank $rank: JLD2Writer created, running simulation..."
    flush(stdout); flush(stderr)
    run!(sim_prescribed)
    @info "Rank $rank: JLD2Writer prescribed — PASS"
    global npass += 1
catch e
    @error "Rank $rank: JLD2Writer prescribed — FAIL" exception = (e, catch_backtrace())
    global nfail += 1
end

rm("test_halofill_prescribed_rank$(rank).jld2"; force = true)

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Summary
# =========================================================================

@info "Rank $rank: $npass passed, $nfail failed ($(npass + nfail) total)"
flush(stdout); flush(stderr)
