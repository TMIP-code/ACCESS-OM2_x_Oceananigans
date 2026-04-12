"""
Bit-exact comparison of distributed grid metrics against serial grid metrics.

Each MPI rank:
1. Loads the serial tripolar grid on CPU.
2. Loads the distributed tripolar grid on CPU (using its portion of the partition).
3. For each coordinate/metric array (λ, φ, Δx, Δy, Az at CC/CF/FC/FF),
   extracts its local parent array.
4. Computes the expected serial-parent slice at the corresponding global positions
   using the same (x_offset, y_offset, local_parent_size) logic as partition_data.jl.
5. Prints max|diff| per metric per rank.

If everything is bit-identical, all max|diff| should be 0.0. Any non-zero value
indicates that rank N's metric at some halo position differs from the serial
metric at the same global position — which would explain non-reproducibility
in diagnostics that read grid metrics near rank boundaries (e.g., compute_w).

Usage:
    mpiexec -n {RANKS} julia --project test/test_grid_metrics_distributed.jl

Env vars:
    PARENT_MODEL, EXPERIMENT, TIME_WINDOW — standard
    PARTITION — e.g. "1x2", "1x3", "2x2" (REQUIRED; x*y must equal MPI rank count)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using MPI
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.DistributedComputations: Distributed, Partition, local_size, concatenate_local_sizes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using JLD2
using Printf

include("../src/shared_functions.jl")

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

PARTITION = get(ENV, "PARTITION", "1x2")
px, py = parse.(Int, split(PARTITION, "x"))
@assert px * py == nranks "PARTITION=$PARTITION has $(px * py) ranks but MPI launched $nranks"

arch = Distributed(CPU(); partition = Partition(px, py))

(; parentmodel, experiment_dir) = load_project_config()
grid_file = joinpath(experiment_dir, "grid.jld2")

################################################################################
# Load both grids on this rank
################################################################################

rank == 0 && @info "Loading serial grid from $grid_file"
MPI.Barrier(comm)
serial_grid = load_tripolar_grid(grid_file, CPU())
serial_ug = serial_grid isa ImmersedBoundaryGrid ? serial_grid.underlying_grid : serial_grid

@info "Rank $rank: loading distributed grid ($PARTITION)"
flush(stdout); flush(stderr)
dist_grid = load_tripolar_grid(grid_file, arch)
dist_ug = dist_grid isa ImmersedBoundaryGrid ? dist_grid.underlying_grid : dist_grid

Nx, Ny, Nz = serial_ug.Nx, serial_ug.Ny, serial_ug.Nz

# Reconstruct offsets using the same logic as partition_data.jl
lsize = local_size(arch, (Nx, Ny, Nz))
nxlocal = concatenate_local_sizes(lsize, arch, 1)
nylocal = concatenate_local_sizes(lsize, arch, 2)
xrank = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
yrank = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)
x_offset = sum(nxlocal[1:xrank])
y_offset = sum(nylocal[1:yrank])

@info "Rank $rank: local_Nx=$(dist_ug.Nx), local_Ny=$(dist_ug.Ny), offsets=($x_offset, $y_offset)"
flush(stdout); flush(stderr)
MPI.Barrier(comm)

################################################################################
# Compare per-rank metric parents against serial parent slices
################################################################################

function compare_parent!(name::String, serial_parent, dist_parent)
    # If shapes match exactly, the distributed field is global-sized (e.g. z-star
    # MutableVerticalDiscretization fields are shared across ranks). In that case
    # just compare the full arrays directly.
    serial_slice = if size(serial_parent) == size(dist_parent)
        serial_parent
    else
        npx_local, npy_local = size(dist_parent, 1), size(dist_parent, 2)
        i_range = (1 + x_offset):(x_offset + npx_local)
        j_range = (1 + y_offset):(y_offset + npy_local)

        if last(i_range) > size(serial_parent, 1) || last(j_range) > size(serial_parent, 2)
            @error "Rank $rank: $name slice out of bounds (serial=$(size(serial_parent)), slice=($i_range, $j_range))"
            return 1
        end

        if ndims(serial_parent) == 3
            serial_parent[i_range, j_range, :]
        else
            serial_parent[i_range, j_range]
        end
    end

    if size(serial_slice) != size(dist_parent)
        @error "Rank $rank: $name shape mismatch: dist=$(size(dist_parent)) serial_slice=$(size(serial_slice))"
        return 1
    end

    diff = dist_parent .- serial_slice
    absdiff = abs.(filter(!isnan, diff))
    max_abs = isempty(absdiff) ? 0.0 : maximum(absdiff)
    n_non_zero = count(!iszero, diff)

    worst_ij = "(none)"
    if max_abs > 0
        lin = findall(x -> abs(x) > max_abs / 2, diff)
        worst_ij = isempty(lin) ? "(?)" : string(first(lin))
    end

    status = max_abs == 0 ? "OK" : "MISMATCH"
    @info @sprintf(
        "Rank %d: %-10s %-9s max|diff|=%.3e n_nonzero=%d worst=%s",
        rank, name, status, max_abs, n_non_zero, worst_ij,
    )
    flush(stdout); flush(stderr)
    return max_abs == 0 ? 0 : 1
end

horizontal_metrics = (
    :λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ,
    :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
    :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
    :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
    :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ,
)

total_mismatches = Ref(0)
for name in horizontal_metrics
    total_mismatches[] += compare_parent!(
        string(name),
        Array(parent(getproperty(serial_ug, name))),
        Array(parent(getproperty(dist_ug, name))),
    )
end

# bottom_height (from PartialCellBottom immersed boundary)
if serial_grid isa ImmersedBoundaryGrid && dist_grid isa ImmersedBoundaryGrid
    serial_ib = serial_grid.immersed_boundary
    dist_ib = dist_grid.immersed_boundary
    if hasproperty(serial_ib, :bottom_height) && hasproperty(dist_ib, :bottom_height)
        total_mismatches[] += compare_parent!(
            "bottom",
            Array(parent(serial_ib.bottom_height.data)),
            Array(parent(dist_ib.bottom_height.data)),
        )
    end
end

# z-star scaling fields (σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, ηⁿ, ∂t_σ)
# These live in ug.z (MutableVerticalDiscretization) as OffsetArrays
if hasproperty(serial_ug.z, :σᶜᶜⁿ) && hasproperty(dist_ug.z, :σᶜᶜⁿ)
    for zname in (:σᶜᶜⁿ, :σᶠᶜⁿ, :σᶜᶠⁿ, :σᶠᶠⁿ, :ηⁿ, :∂t_σ)
        hasproperty(serial_ug.z, zname) || continue
        hasproperty(dist_ug.z, zname) || continue
        total_mismatches[] += compare_parent!(
            string(zname),
            Array(parent(getproperty(serial_ug.z, zname))),
            Array(parent(getproperty(dist_ug.z, zname))),
        )
    end
end

MPI.Barrier(comm)

global_mismatches = MPI.Allreduce(total_mismatches[], +, comm)
if rank == 0
    if global_mismatches == 0
        @info "PASS: all grid metrics match bit-for-bit across all ranks"
    else
        @error "FAIL: $global_mismatches metric mismatches across all ranks"
    end
    flush(stdout); flush(stderr)
end

MPI.Finalize()
