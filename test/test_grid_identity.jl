"""
Test that the distributed grid has identical metrics to the serial grid.

Each MPI rank loads the global grid (serial) and builds its local distributed
partition, then compares the 20 coordinate/metric arrays. All diffs must be
exactly zero — the distributed grid must use the same pre-computed arrays, not
recompute them.

Run with 4 CPUs:
    mpiexec -n 4 julia --project test/test_grid_identity.jl
"""

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: on_architecture
using Oceananigans.DistributedComputations: Distributed, local_size, concatenate_local_sizes, ranks
using JLD2
using MPI

MPI.Init()

@info "GIT_COMMIT = $(get(ENV, "GIT_COMMIT", "unknown"))"

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)

px, py = if nranks == 4
    (2, 2)
elseif nranks == 2
    (1, 2)
else
    error("Expected 2 or 4 MPI ranks, got $nranks")
end

arch = Distributed(CPU(), partition = Partition(px, py))
@info "MPI rank $rank/$nranks, partition=$(px)x$(py)"

# Load shared functions for build_underlying_grid and load_tripolar_grid
include("../src/shared_functions.jl")

# Load project config to find grid file
(; parentmodel) = load_project_config()
preprocessed_inputs_dir = joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel)
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Rank $rank: Loading grid from $grid_file"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# Build both grids
gd = load(grid_file)
serial_grid = build_underlying_grid(gd, CPU(), Float64)
dist_grid = build_underlying_grid(gd, arch, Float64)

# Compute this rank's interior index ranges (matching the partition logic)
Nx, Ny, Nz = size(serial_grid)
lsize = local_size(arch, (Nx, Ny, Nz))
nxlocal = concatenate_local_sizes(lsize, arch, 1)
nylocal = concatenate_local_sizes(lsize, arch, 2)

workers = ranks(arch.partition)
xrank = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
yrank = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)

istart = 1 + sum(nxlocal[1:xrank])
iend = xrank == workers[1] - 1 ? Nx : sum(nxlocal[1:(xrank + 1)])
jstart = 1 + sum(nylocal[1:yrank])
jend = yrank == workers[2] - 1 ? Ny : sum(nylocal[1:(yrank + 1)])

# Interior range in global coordinates
iglobal = istart:iend
jglobal = jstart:jend

# Interior range in local coordinates (1-based)
nx_local = nxlocal[xrank + 1]
ny_local = nylocal[yrank + 1]
ilocal = 1:nx_local
jlocal = 1:ny_local

@info "Rank $rank: global i=$iglobal, j=$jglobal, local nx=$nx_local, ny=$ny_local"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

# Compare all 20 coordinate/metric arrays
metric_names = [
    :λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ,
    :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
    :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
    :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
    :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ,
]

all_pass = true
for name in metric_names
    serial_arr = parent(getproperty(serial_grid, name))
    dist_arr = parent(getproperty(dist_grid, name))

    # Extract matching interior regions
    serial_slice = serial_arr[iglobal, jglobal]
    dist_slice = dist_arr[ilocal, jlocal]

    maxdiff = maximum(abs.(serial_slice .- dist_slice))

    if maxdiff > 0
        @error "Rank $rank: $name MISMATCH — max|diff| = $maxdiff"
        global all_pass = false
    else
        @info "Rank $rank: $name — OK (max|diff| = 0)"
    end
end

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

if all_pass
    @info "Rank $rank: ALL METRICS MATCH — grid identity test PASSED"
else
    @error "Rank $rank: grid identity test FAILED"
    exit(1)
end

flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)
