using Oceananigans
using CUDA
using MPI

arch = Distributed(GPU())
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)

@info "Rank $rank of $Nranks on device $(CUDA.device())"

grid = RectilinearGrid(
    arch; size = (64, 64, 64),
    x = (0, 1), y = (0, 1), z = (0, 1),
    topology = (Periodic, Periodic, Bounded)
)

free_surface = SplitExplicitFreeSurface(grid; substeps = 10)
model = HydrostaticFreeSurfaceModel(grid; tracers = :c, free_surface)
set!(model, c = (x, y, z) -> x + y + z)

simulation = Simulation(model, Δt = 1, stop_iteration = 10)
run!(simulation)

@info "Rank $rank: simulation completed successfully!"
