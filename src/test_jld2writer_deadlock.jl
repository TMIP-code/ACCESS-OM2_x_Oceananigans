"""
MWE: JLD2Writer deadlock when serializing a distributed tripolar grid.

`JLD2Writer` saves grid metadata by calling `serializeproperty!` on each field
of the model's grid. For a `DistributedGrid`, this internally constructs a fresh
`Distributed(CPU(); ...)` object, which triggers MPI collective operations
(e.g. `MPI_Bcast`). On GPU runs, ranks can desynchronize (one rank finishes GPU
kernels and enters the output write while others are still computing), so they
arrive at the MPI collective at different times and deadlock. Even on CPU, having
multiple MPI collectives execute inside `serializeproperty!` without synchronizing
all ranks is unsafe.

The fix is `including=[]`, which skips grid/model metadata serialization entirely.

Run with 2 CPU ranks (no GPUs required to reproduce the deadlock path):
    mpiexec -n 2 julia --project src/test_jld2writer_deadlock.jl

In stock Oceananigans.jl the writer is called `JLD2OutputWriter`; in this fork
it is `JLD2Writer`. The bug exists in both.

Expected: simulation hangs at the first output write (MPI deadlock inside
          `JLD2HDF5.serializeproperty!` → `Distributed(CPU(); ...)` constructor).
"""

using MPI
using Oceananigans
using Oceananigans.OutputWriters
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

MPI.Init()

@info "GIT_COMMIT = $(get(ENV, "GIT_COMMIT", "unknown"))"

arch = Distributed(CPU(), partition = Partition(1, 2))
rank = MPI.Comm_rank(MPI.COMM_WORLD)

grid = TripolarGrid(
    arch, Float64;
    size = (20, 21, 2),
    z = (-100, 0),
    halo = (7, 7, 7),
    first_pole_longitude = 75,
    north_poles_latitude = 55,
)

@info "Rank $rank: TripolarGrid built"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

model = HydrostaticFreeSurfaceModel(
    grid;
    velocities = PrescribedVelocityFields(),
    tracers = (; c = CenterField(grid)),
    free_surface = nothing,
)
simulation = Simulation(model; Δt = 1.0, stop_time = 3.0)

@info "Rank $rank: model + simulation built"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# BUG: hangs at first write because JLD2Writer calls serializeproperty! on
# the DistributedGrid, which internally constructs Distributed(CPU(); ...)
# and triggers an MPI collective that not all ranks reach simultaneously.
#
# FIX: add `including = []` to skip grid/model metadata serialization.
simulation.output_writers[:c] = JLD2Writer(
    model, model.tracers;
    schedule = IterationInterval(1),
    filename = "test_jld2_deadlock_rank$(rank)",
    overwrite_existing = true,
    # including = [],  # ← uncomment to fix the deadlock
)

@info "Rank $rank: JLD2Writer added (construction did not hang)"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

@info "Rank $rank: calling run!() — hang expected inside first JLD2 write (iteration 0)"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)

@info "Rank $rank: run!() returned — this should NOT print without including=[]"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

rm("test_jld2_deadlock_rank$(rank).jld2"; force = true)
@info "Rank $rank: done"
flush(stdout); flush(stderr)
