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
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: MutableVerticalDiscretization, RightFaceFolded
using Oceananigans.OrthogonalSphericalShellGrids: fold_set!
using Oceananigans.OutputReaders: InMemory
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity, VerticalScalarDiffusivity,
    VerticallyImplicitTimeDiscretization
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

# --- Shared grid parameters (match real model: 360×301×50) ---
Nx, Ny, Nz = 360, 301, 50
z_faces = collect(range(-5500, 0; length = Nz + 1))
halo = (7, 7, 7)
grid_kw = (; first_pole_longitude = 75, north_poles_latitude = 55, fold_topology = RightFaceFolded)
bottom_height_func(x, y) = -4000  # shared bottom function for serial/distributed IB consistency

# --- Helpers ---

function make_grid(arch; z = (-1000, 0))
    return TripolarGrid(arch, Float64; size = (Nx, Ny, Nz), z = z, halo = halo, grid_kw...)
end

"""
    make_velocity_fts(ibg, z, ib_constructor)

Create u, v FieldTimeSeries by saving to JLD2 from a serial grid then reloading,
mimicking the real pipeline (create_velocities.jl → load_fts in data_loading.jl).
On distributed grids, rank 0 creates the serial FTS file, all ranks load it.
`z` is the z-specification for TripolarGrid (tuple or MutableVerticalDiscretization).
`ib_constructor` is a function `grid -> immersed_boundary` (e.g. PartialCellBottom).
"""
function make_velocity_fts(ibg, z, ib_constructor)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    grid_arch = Oceananigans.Architectures.architecture(ibg)

    if grid_arch isa Distributed
        u_file = "test_halofill_mwe_u.jld2"
        v_file = "test_halofill_mwe_v.jld2"

        # Rank 0 creates serial FTS and saves to JLD2
        if rank == 0
            serial_ibg = _make_serial_ibg(z, ib_constructor)
            _save_fts_to_jld2(
                serial_ibg, Face, Center, Center, "u",
                (x, y, z) -> 0.1 * cosd(y), fts_times, u_file
            )
            _save_fts_to_jld2(
                serial_ibg, Center, Face, Center, "v",
                (x, y, z) -> 0.0, fts_times, v_file
            )
        end
        MPI.Barrier(MPI.COMM_WORLD)

        # All ranks: load via serial CPU grid then distribute (matches load_fts(::Distributed))
        cpu_ibg = _make_serial_ibg(z, ib_constructor)
        time_indexing = Oceananigans.OutputReaders.Cyclical(last(fts_times))
        u_fts = _load_fts_distributed(u_file, "u", ibg, cpu_ibg, time_indexing)
        v_fts = _load_fts_distributed(v_file, "v", ibg, cpu_ibg, time_indexing)

        MPI.Barrier(MPI.COMM_WORLD)
        rm(u_file; force = true)
        rm(v_file; force = true)
        return u_fts, v_fts
    else
        u_fts = FieldTimeSeries{Face, Center, Center}(ibg, fts_times)
        v_fts = FieldTimeSeries{Center, Face, Center}(ibg, fts_times)
        for n in eachindex(fts_times)
            set!(u_fts[n], (x, y, z) -> 0.1 * cosd(y))
            set!(v_fts[n], (x, y, z) -> 0.0)
        end
        return u_fts, v_fts
    end
end

function make_w_fts(ibg, z, ib_constructor)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    grid_arch = Oceananigans.Architectures.architecture(ibg)

    if grid_arch isa Distributed
        w_file = "test_halofill_mwe_w.jld2"
        if rank == 0
            serial_ibg = _make_serial_ibg(z, ib_constructor)
            _save_fts_to_jld2(
                serial_ibg, Center, Center, Face, "w",
                (x, y, z) -> 0.0, fts_times, w_file
            )
        end
        MPI.Barrier(MPI.COMM_WORLD)

        cpu_ibg = _make_serial_ibg(z, ib_constructor)
        time_indexing = Oceananigans.OutputReaders.Cyclical(last(fts_times))
        w_fts = _load_fts_distributed(w_file, "w", ibg, cpu_ibg, time_indexing)

        MPI.Barrier(MPI.COMM_WORLD)
        rm(w_file; force = true)
        return w_fts
    else
        w_fts = FieldTimeSeries{Center, Center, Face}(ibg, fts_times)
        for n in eachindex(fts_times)
            set!(w_fts[n], (x, y, z) -> 0.0)
        end
        return w_fts
    end
end

function make_eta_fts(ibg, z, ib_constructor)
    fts_times = [0.0, 1.0, 2.0, 3.0]
    grid_arch = Oceananigans.Architectures.architecture(ibg)

    if grid_arch isa Distributed
        η_file = "test_halofill_mwe_eta.jld2"
        if rank == 0
            serial_ibg = _make_serial_ibg(z, ib_constructor)
            _save_fts_to_jld2(
                serial_ibg, Center, Center, Nothing, "η",
                (x, y) -> 0.0, fts_times, η_file
            )
        end
        MPI.Barrier(MPI.COMM_WORLD)

        cpu_ibg = _make_serial_ibg(z, ib_constructor)
        time_indexing = Oceananigans.OutputReaders.Cyclical(last(fts_times))
        η_fts = _load_fts_distributed(η_file, "η", ibg, cpu_ibg, time_indexing)

        MPI.Barrier(MPI.COMM_WORLD)
        rm(η_file; force = true)
        return η_fts
    else
        η_fts = FieldTimeSeries{Center, Center, Nothing}(ibg, fts_times)
        for n in eachindex(fts_times)
            set!(η_fts[n], (x, y) -> 0.0)
        end
        return η_fts
    end
end

# --- Internal helpers for save/load FTS pipeline ---

"""Build a serial CPU ImmersedBoundaryGrid (used by rank 0 to save FTS and by all ranks to load)."""
function _make_serial_ibg(z, ib_constructor)
    serial_grid = TripolarGrid(CPU(), Float64; size = (Nx, Ny, Nz), z = z, halo = halo, grid_kw...)
    return ImmersedBoundaryGrid(serial_grid, ib_constructor)
end

"""
Save a FieldTimeSeries to JLD2 using the OnDisk backend, matching the pipeline
in create_velocities.jl: create FTS with OnDisk(), then set! each snapshot.
"""
function _save_fts_to_jld2(ibg, LX, LY, LZ, name, func, times, filepath)
    time_indexing = Oceananigans.OutputReaders.Cyclical(last(times))
    ondisk_fts = FieldTimeSeries{LX, LY, LZ}(
        ibg, times;
        backend = Oceananigans.OutputReaders.OnDisk(),
        path = filepath, name = name,
        time_indexing = time_indexing,
    )
    field = Field{LX, LY, LZ}(ibg)
    for n in eachindex(times)
        set!(field, func)
        fill_halo_regions!(field)
        set!(ondisk_fts, field, n)
    end
    return nothing
end

"""Load FTS from JLD2 on distributed grid, mimicking load_fts(::Distributed, ...) in data_loading.jl."""
function _load_fts_distributed(filepath, name, dist_ibg, cpu_ibg, time_indexing)
    cpu_fts = FieldTimeSeries(
        filepath, name;
        architecture = CPU(), grid = cpu_ibg,
        backend = InMemory(), time_indexing
    )

    dist_fts = FieldTimeSeries(
        instantiated_location(cpu_fts), dist_ibg, cpu_fts.times;
        backend = InMemory(), time_indexing
    )

    conformal_mapping = dist_ibg.underlying_grid.conformal_mapping
    y_loc = instantiated_location(cpu_fts)[2]
    for n in eachindex(cpu_fts.times)
        fold_set!(dist_fts[n], Array(interior(cpu_fts[n])), conformal_mapping, typeof(y_loc))
    end
    fill_halo_regions!(dist_fts)
    return dist_fts
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
        @info "Rank $rank: [$test_name] JLD2Writer created"
        if rank == 0
            @info "Rank 0: [$test_name] Grid:"
            show(stdout, MIME"text/plain"(), model.grid)
            println(stdout)
            @info "Rank 0: [$test_name] Model:"
            show(stdout, MIME"text/plain"(), model)
            println(stdout)
            @info "Rank 0: [$test_name] Simulation:"
            show(stdout, MIME"text/plain"(), sim)
            println(stdout)
        end
        flush(stdout); flush(stderr)
        MPI.Barrier(MPI.COMM_WORLD)
        @info "Rank $rank: [$test_name] running..."
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

#= Tests 1-3 commented out for faster iteration — uncomment to re-enable

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

=# # end of Tests 1-3 block comment

# =========================================================================
# Test 4: FULL model (all features matching actual simulation)
#   PartialCellBottom + MVD + FTS(u,v) + DiagnosticW + PFS(η) + mixed output
# =========================================================================

try
    z_4 = MutableVerticalDiscretization(z_faces)
    ib_4 = PartialCellBottom(bottom_height_func)
    grid_4 = make_grid(arch; z = z_4)
    ibg_4 = ImmersedBoundaryGrid(grid_4, ib_4)

    u_fts_4, v_fts_4 = make_velocity_fts(ibg_4, z_4, ib_4)
    η_fts_4 = make_eta_fts(ibg_4, z_4, ib_4)

    # Match real model closure: horizontal + implicit vertical diffusivity with Field κ
    κV_field = CenterField(ibg_4)
    set!(κV_field, (x, y, z) -> 1.0e-4)  # uniform for simplicity
    implicit_vertical_diffusion = VerticalScalarDiffusivity(
        VerticallyImplicitTimeDiscretization();
        κ = κV_field,
    )
    horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
    closure = (horizontal_diffusion, implicit_vertical_diffusion)

    model_4 = HydrostaticFreeSurfaceModel(
        ibg_4;
        velocities = PrescribedVelocityFields(
            u = u_fts_4, v = v_fts_4,
            formulation = DiagnosticVerticalVelocity(),
        ),
        tracers = (; age = CenterField(ibg_4)),
        free_surface = PrescribedFreeSurface(displacement = η_fts_4),
        closure = closure,
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

#= Tests 4a-4f commented out for faster iteration — uncomment to re-enable

# =========================================================================
# Test 4a: Remove PartialCellBottom → GridFittedBottom
# =========================================================================
... (4a through 4f) ...
=#

# =========================================================================
# Summary
# =========================================================================

@info "Rank $rank: $npass passed, $nfail failed ($(npass + nfail) total)"
flush(stdout); flush(stderr)
