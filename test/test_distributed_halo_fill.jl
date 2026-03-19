"""
MWE: Isolate which feature causes JLD2Writer to hang on distributed tripolar grids.

Each test adds ONE difference from the baseline (Test 3) that passes:
  3.  Baseline: GridFittedBottom, static z, PrescribedVelocityFields(), free_surface=nothing
  3a. PartialCellBottom (instead of GridFittedBottom)
  3b. MutableVerticalDiscretization (instead of static z tuple)
  3c. PrescribedVelocityFields with FieldTimeSeries u, v (→ TimeSeriesInterpolation)
  3d. PrescribedVelocityFields with FTS u, v + DiagnosticVerticalVelocity
  3e. PrescribedFreeSurface with FieldTimeSeries η

Run with 4 CPUs:
    mpiexec -n 4 julia --project test/test_distributed_halo_fill.jl

Run with 4 GPUs:
    mpiexec --bind-to socket --map-by socket -n 4 julia --project test/test_distributed_halo_fill.jl
"""

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: MutableVerticalDiscretization
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

px, py = if nranks == 4
    (2, 2)
elseif nranks == 2
    (1, 2)
else
    error("Expected 2 or 4 MPI ranks, got $nranks")
end

arch = Distributed(child_arch, partition = Partition(px, py))

# --- Shared grid parameters ---
Nx, Ny, Nz = 60, 61, 10
z_faces = collect(range(-1000, 0; length = Nz + 1))
halo = (7, 7, 7)
grid_kw = (; first_pole_longitude = 75, north_poles_latitude = 55)

# --- Helper: build a standard TripolarGrid ---
function make_grid(arch; z = (-1000, 0))
    return TripolarGrid(arch, Float64; size = (Nx, Ny, Nz), z = z, halo = halo, grid_kw...)
end

# --- Helper: build FieldTimeSeries for prescribed velocities ---
function make_velocity_fts(ibg)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    u_fts = FieldTimeSeries{Face, Center, Center}(ibg, fts_times)
    v_fts = FieldTimeSeries{Center, Face, Center}(ibg, fts_times)
    for n in eachindex(fts_times)
        set!(u_fts[n], (x, y, z) -> 0.1 * cosd(y))
        set!(v_fts[n], (x, y, z) -> 0.0)
    end
    return u_fts, v_fts
end

# --- Helper: build FieldTimeSeries for prescribed free surface ---
function make_eta_fts(ibg)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    η_fts = FieldTimeSeries{Center, Center, Nothing}(ibg, fts_times)
    for n in eachindex(fts_times)
        set!(η_fts[n], (x, y) -> 0.0)
    end
    return η_fts
end

# --- Helper: run a JLD2Writer test with a given model + output fields ---
function run_jld2writer_test(test_name, model, output_fields, Δt)
    @info "Rank $rank: [$test_name] starting..."
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)

    sim = Simulation(model; Δt = Δt, stop_time = 2 * Δt)
    filename = "test_halofill_$(test_name)_rank$(rank)"

    try
        sim.output_writers[:out] = JLD2Writer(
            model, output_fields;
            schedule = TimeInterval(Δt),
            filename = filename,
            overwrite_existing = true,
            with_halos = true,
            including = [],
        )
        @info "Rank $rank: [$test_name] JLD2Writer created, running..."
        flush(stdout); flush(stderr)
        run!(sim)
        @info "Rank $rank: [$test_name] — PASS"
        return true
    catch e
        @error "Rank $rank: [$test_name] — FAIL" exception = (e, catch_backtrace())
        return false
    finally
        rm("$(filename).jld2"; force = true)
        flush(stdout); flush(stderr)
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

npass = 0
nfail = 0

# =========================================================================
# Test 1: fill_halo_regions! at all staggered locations
# =========================================================================

grid_baseline = make_grid(arch)
ibg_baseline = ImmersedBoundaryGrid(grid_baseline, GridFittedBottom((x, y) -> -1000))

@info "Rank $rank: grid size = $(size(ibg_baseline)), topology = $(Oceananigans.Grids.topology(ibg_baseline))"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

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
    f = Field{LX, LY, LZ}(ibg_baseline)
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

fts_test = FieldTimeSeries{Center, Face, Center}(ibg_baseline, [0.0, 1.0, 2.0])
for n in 1:3
    set!(fts_test[n], (x, y, z) -> x + y + z + n)
end

try
    fill_halo_regions!(fts_test)
    @info "Rank $rank: FTS (Center, Face, Center) — PASS"
    global npass += 1
catch e
    @error "Rank $rank: FTS (Center, Face, Center) — FAIL" exception = (e, catch_backtrace())
    global nfail += 1
end

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# =========================================================================
# Test 3: Baseline JLD2Writer — GridFittedBottom, static z, simple model
# =========================================================================

model_3 = HydrostaticFreeSurfaceModel(
    ibg_baseline;
    velocities = PrescribedVelocityFields(),
    tracers = (; age = CenterField(ibg_baseline)),
    free_surface = nothing,
)

v_field = Field{Center, Face, Center}(ibg_baseline)
set!(v_field, (x, y, z) -> y + z)

if run_jld2writer_test("3_baseline", model_3, Dict("v" => v_field), 1.0)
    global npass += 1
else
    global nfail += 1
end

# =========================================================================
# Test 3a: PartialCellBottom (instead of GridFittedBottom)
# =========================================================================

try
    grid_3a = make_grid(arch)
    ibg_3a = ImmersedBoundaryGrid(grid_3a, PartialCellBottom((x, y) -> -800))

    model_3a = HydrostaticFreeSurfaceModel(
        ibg_3a;
        velocities = PrescribedVelocityFields(),
        tracers = (; age = CenterField(ibg_3a)),
        free_surface = nothing,
    )

    v_3a = Field{Center, Face, Center}(ibg_3a)
    set!(v_3a, (x, y, z) -> y + z)

    if run_jld2writer_test("3a_PartialCellBottom", model_3a, Dict("v" => v_3a), 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [3a_PartialCellBottom] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 3b: MutableVerticalDiscretization (instead of static z)
# =========================================================================

try
    grid_3b = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_3b = ImmersedBoundaryGrid(grid_3b, GridFittedBottom((x, y) -> -1000))

    model_3b = HydrostaticFreeSurfaceModel(
        ibg_3b;
        velocities = PrescribedVelocityFields(),
        tracers = (; age = CenterField(ibg_3b)),
        free_surface = nothing,
    )

    v_3b = Field{Center, Face, Center}(ibg_3b)
    set!(v_3b, (x, y, z) -> y + z)

    if run_jld2writer_test("3b_MutableVertDisc", model_3b, Dict("v" => v_3b), 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [3b_MutableVertDisc] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 3c: PrescribedVelocityFields with FTS u, v (→ TimeSeriesInterpolation)
# =========================================================================

try
    u_fts_3c, v_fts_3c = make_velocity_fts(ibg_baseline)

    model_3c = HydrostaticFreeSurfaceModel(
        ibg_baseline;
        velocities = PrescribedVelocityFields(u = u_fts_3c, v = v_fts_3c),
        tracers = (; age = CenterField(ibg_baseline)),
        free_surface = nothing,
    )

    @info "Rank $rank: [3c] u type = $(typeof(model_3c.velocities.u))"
    @info "Rank $rank: [3c] v type = $(typeof(model_3c.velocities.v))"
    flush(stdout); flush(stderr)

    if run_jld2writer_test(
            "3c_FTS_velocities", model_3c,
            Dict("age" => model_3c.tracers.age), 1.0
        )
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [3c_FTS_velocities] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 3d: FTS velocities + DiagnosticVerticalVelocity
# =========================================================================

try
    u_fts_3d, v_fts_3d = make_velocity_fts(ibg_baseline)

    model_3d = HydrostaticFreeSurfaceModel(
        ibg_baseline;
        velocities = PrescribedVelocityFields(
            u = u_fts_3d, v = v_fts_3d,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_baseline)),
        free_surface = nothing,
    )

    @info "Rank $rank: [3d] w type = $(typeof(model_3d.velocities.w))"
    flush(stdout); flush(stderr)

    if run_jld2writer_test(
            "3d_DiagnosticW", model_3d,
            Dict("age" => model_3d.tracers.age, "w" => model_3d.velocities.w), 1.0
        )
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [3d_DiagnosticW] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 3e: PrescribedFreeSurface with FTS η
# =========================================================================

try
    η_fts_3e = make_eta_fts(ibg_baseline)

    model_3e = HydrostaticFreeSurfaceModel(
        ibg_baseline;
        velocities = PrescribedVelocityFields(),
        tracers = (; age = CenterField(ibg_baseline)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_3e),
    )

    @info "Rank $rank: [3e] η type = $(typeof(model_3e.free_surface.displacement))"
    flush(stdout); flush(stderr)

    if run_jld2writer_test(
            "3e_PrescribedFreeSurf", model_3e,
            Dict("age" => model_3e.tracers.age), 1.0
        )
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [3e_PrescribedFreeSurf] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Summary
# =========================================================================

@info "Rank $rank: $npass passed, $nfail failed ($(npass + nfail) total)"
flush(stdout); flush(stderr)
