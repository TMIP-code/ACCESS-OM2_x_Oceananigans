"""
MWE: Isolate which feature causes JLD2Writer to hang on distributed tripolar grids.

Tests 1-3: Sanity checks (fill_halo_regions!, simple JLD2Writer) — all pass.

Test 4 (full): All features matching the actual simulation:
  - PartialCellBottom
  - MutableVerticalDiscretization
  - PrescribedVelocityFields with FTS u, v (→ TimeSeriesInterpolation)
  - DiagnosticVerticalVelocity (w = regular Field)
  - PrescribedFreeSurface with FTS η (→ TSI)
  - JLD2Writer outputting mixed types: age (Field), u (TSI), v (TSI), w (Field), eta (TSI)

Tests 4a-4f: Remove ONE feature from the full setup:
  4a. No PartialCellBottom → GridFittedBottom
  4b. No MutableVerticalDiscretization → static z
  4c. No FTS velocities → PrescribedVelocityFields() (no prescribed u/v/w)
  4d. No DiagnosticVerticalVelocity → prescribe w as FTS too
  4e. No PrescribedFreeSurface → free_surface=nothing
  4f. No mixed output → only output age (CenterField)

If Test 4 hangs but 4X passes → feature X is (part of) the culprit.

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

# --- Helpers ---

function make_grid(arch; z = (-1000, 0))
    return TripolarGrid(arch, Float64; size = (Nx, Ny, Nz), z = z, halo = halo, grid_kw...)
end

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

function make_w_fts(ibg)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    w_fts = FieldTimeSeries{Center, Center, Face}(ibg, fts_times)
    for n in eachindex(fts_times)
        set!(w_fts[n], (x, y, z) -> 0.0)
    end
    return w_fts
end

function make_eta_fts(ibg)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    η_fts = FieldTimeSeries{Center, Center, Nothing}(ibg, fts_times)
    for n in eachindex(fts_times)
        set!(η_fts[n], (x, y) -> 0.0)
    end
    return η_fts
end

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
# Test 3: Baseline JLD2Writer — simple model, single Face field
# =========================================================================

try
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
catch e
    @error "Rank $rank: [3_baseline] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4: FULL model (all features matching actual simulation)
#   PartialCellBottom + MVD + FTS(u,v) + DiagnosticW + PFS(η) + mixed output
# =========================================================================

try
    grid_4 = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4 = ImmersedBoundaryGrid(grid_4, PartialCellBottom((x, y) -> -800))

    u_fts_4, v_fts_4 = make_velocity_fts(ibg_4)
    η_fts_4 = make_eta_fts(ibg_4)

    model_4 = HydrostaticFreeSurfaceModel(
        ibg_4;
        velocities = PrescribedVelocityFields(
            u = u_fts_4, v = v_fts_4,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4),
    )

    @info "Rank $rank: [4_full] u=$(typeof(model_4.velocities.u)), w=$(typeof(model_4.velocities.w))"
    flush(stdout); flush(stderr)

    output_4 = Dict(
        "age" => model_4.tracers.age,
        "u" => model_4.velocities.u,
        "v" => model_4.velocities.v,
        "w" => model_4.velocities.w,
        "eta" => model_4.free_surface.displacement,
    )

    if run_jld2writer_test("4_full", model_4, output_4, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4_full] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4a: Remove PartialCellBottom → GridFittedBottom
# =========================================================================

try
    grid_4a = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4a = ImmersedBoundaryGrid(grid_4a, GridFittedBottom((x, y) -> -800))

    u_fts_4a, v_fts_4a = make_velocity_fts(ibg_4a)
    η_fts_4a = make_eta_fts(ibg_4a)

    model_4a = HydrostaticFreeSurfaceModel(
        ibg_4a;
        velocities = PrescribedVelocityFields(
            u = u_fts_4a, v = v_fts_4a,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4a)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4a),
    )

    output_4a = Dict(
        "age" => model_4a.tracers.age,
        "u" => model_4a.velocities.u,
        "v" => model_4a.velocities.v,
        "w" => model_4a.velocities.w,
        "eta" => model_4a.free_surface.displacement,
    )

    if run_jld2writer_test("4a_no_PCB", model_4a, output_4a, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4a_no_PCB] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4b: Remove MutableVerticalDiscretization → static z
# =========================================================================

try
    grid_4b = make_grid(arch)  # static z = (-1000, 0)
    ibg_4b = ImmersedBoundaryGrid(grid_4b, PartialCellBottom((x, y) -> -800))

    u_fts_4b, v_fts_4b = make_velocity_fts(ibg_4b)
    η_fts_4b = make_eta_fts(ibg_4b)

    model_4b = HydrostaticFreeSurfaceModel(
        ibg_4b;
        velocities = PrescribedVelocityFields(
            u = u_fts_4b, v = v_fts_4b,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4b)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4b),
    )

    output_4b = Dict(
        "age" => model_4b.tracers.age,
        "u" => model_4b.velocities.u,
        "v" => model_4b.velocities.v,
        "w" => model_4b.velocities.w,
        "eta" => model_4b.free_surface.displacement,
    )

    if run_jld2writer_test("4b_no_MVD", model_4b, output_4b, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4b_no_MVD] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4c: Remove FTS velocities → PrescribedVelocityFields() (no args)
#           (also removes DiagnosticVerticalVelocity since no u/v to diagnose from)
# =========================================================================

try
    grid_4c = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4c = ImmersedBoundaryGrid(grid_4c, PartialCellBottom((x, y) -> -800))

    η_fts_4c = make_eta_fts(ibg_4c)

    model_4c = HydrostaticFreeSurfaceModel(
        ibg_4c;
        velocities = PrescribedVelocityFields(),
        tracers = (; age = CenterField(ibg_4c)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4c),
    )

    output_4c = Dict(
        "age" => model_4c.tracers.age,
        "eta" => model_4c.free_surface.displacement,
    )

    if run_jld2writer_test("4c_no_FTS_vel", model_4c, output_4c, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4c_no_FTS_vel] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4d: Remove DiagnosticVerticalVelocity → prescribe w as FTS too
# =========================================================================

try
    grid_4d = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4d = ImmersedBoundaryGrid(grid_4d, PartialCellBottom((x, y) -> -800))

    u_fts_4d, v_fts_4d = make_velocity_fts(ibg_4d)
    w_fts_4d = make_w_fts(ibg_4d)
    η_fts_4d = make_eta_fts(ibg_4d)

    model_4d = HydrostaticFreeSurfaceModel(
        ibg_4d;
        velocities = PrescribedVelocityFields(u = u_fts_4d, v = v_fts_4d, w = w_fts_4d),
        tracers = (; age = CenterField(ibg_4d)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4d),
    )

    output_4d = Dict(
        "age" => model_4d.tracers.age,
        "u" => model_4d.velocities.u,
        "v" => model_4d.velocities.v,
        "w" => model_4d.velocities.w,
        "eta" => model_4d.free_surface.displacement,
    )

    if run_jld2writer_test("4d_no_DiagW", model_4d, output_4d, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4d_no_DiagW] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4e: Remove PrescribedFreeSurface → free_surface=nothing
# =========================================================================

try
    grid_4e = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4e = ImmersedBoundaryGrid(grid_4e, PartialCellBottom((x, y) -> -800))

    u_fts_4e, v_fts_4e = make_velocity_fts(ibg_4e)

    model_4e = HydrostaticFreeSurfaceModel(
        ibg_4e;
        velocities = PrescribedVelocityFields(
            u = u_fts_4e, v = v_fts_4e,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4e)),
        free_surface = nothing,
    )

    output_4e = Dict(
        "age" => model_4e.tracers.age,
        "u" => model_4e.velocities.u,
        "v" => model_4e.velocities.v,
        "w" => model_4e.velocities.w,
    )

    if run_jld2writer_test("4e_no_PFS", model_4e, output_4e, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4e_no_PFS] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Test 4f: Remove mixed output → only output age (CenterField)
# =========================================================================

try
    grid_4f = make_grid(arch; z = MutableVerticalDiscretization(z_faces))
    ibg_4f = ImmersedBoundaryGrid(grid_4f, PartialCellBottom((x, y) -> -800))

    u_fts_4f, v_fts_4f = make_velocity_fts(ibg_4f)
    η_fts_4f = make_eta_fts(ibg_4f)

    model_4f = HydrostaticFreeSurfaceModel(
        ibg_4f;
        velocities = PrescribedVelocityFields(
            u = u_fts_4f, v = v_fts_4f,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4f)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4f),
    )

    output_4f = Dict("age" => model_4f.tracers.age)

    if run_jld2writer_test("4f_age_only", model_4f, output_4f, 1.0)
        global npass += 1
    else
        global nfail += 1
    end
catch e
    @error "Rank $rank: [4f_age_only] — FAIL (setup)" exception = (e, catch_backtrace())
    global nfail += 1
    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

# =========================================================================
# Summary
# =========================================================================

@info "Rank $rank: $npass passed, $nfail failed ($(npass + nfail) total)"
flush(stdout); flush(stderr)
