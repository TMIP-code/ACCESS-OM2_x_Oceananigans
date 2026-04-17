"""
Pre-partition grid and FTS data for distributed (MPI) simulations.

Runs under MPI on CPU. Each rank writes its own JLD2 files containing
local grid arrays and FTS snapshots with correct halos, sliced directly
from the serial global data. This eliminates the need for runtime
`set!` + `fill_halo_regions!` on distributed FTS fields.

Usage (via driver.sh):
    PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x2 JOB_CHAIN=partition bash scripts/driver.sh

Or interactively:
    mpiexec -n 4 julia --project src/partition_data.jl
"""

@info "Loading packages for partition_data"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU, architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: Distributed, local_size, concatenate_local_sizes
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: on_architecture, total_size
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory, Cyclical
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days

using JLD2
using MPI

include("shared_functions.jl")

################################################################################
# MPI initialization
################################################################################

MPI.Init()
px = parse(Int, ENV["PARTITION_X"])
py = parse(Int, ENV["PARTITION_Y"])
arch = Distributed(CPU(), partition = Partition(px, py))
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
@info "MPI rank $rank/$nranks, partition=$(px)x$(py) (CPU)"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config(; parentmodel_arg_index = 2)
(; VELOCITY_SOURCE) = parse_config_env()

experiment = get(ENV, "EXPERIMENT", "1deg_jra55_iaf_omip2_cycle6")
time_window = get(ENV, "TIME_WINDOW", "1968-1977")

grid_file = joinpath(experiment_dir, "grid.jld2")
monthly_dir = joinpath(experiment_dir, time_window, "monthly")

# Output directories
grid_partition_dir = joinpath(experiment_dir, "partitions", "$(px)x$(py)")
fts_partition_dir = joinpath(experiment_dir, time_window, "partitions", "$(px)x$(py)")
mkpath(grid_partition_dir)
mkpath(fts_partition_dir)

@info "Rank $rank: partition_data configuration"
@info "  grid_file = $grid_file"
@info "  monthly_dir = $monthly_dir"
@info "  grid_partition_dir = $grid_partition_dir"
@info "  fts_partition_dir = $fts_partition_dir"
flush(stdout); flush(stderr)

################################################################################
# Load global serial grid
################################################################################

@info "Rank $rank: Loading global serial grid"
flush(stdout); flush(stderr)

gd = load(grid_file)
serial_grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny, Nz = size(serial_grid isa ImmersedBoundaryGrid ? serial_grid.underlying_grid : serial_grid)
ug_serial = serial_grid isa ImmersedBoundaryGrid ? serial_grid.underlying_grid : serial_grid
Hx, Hy, Hz = ug_serial.Hx, ug_serial.Hy, ug_serial.Hz

@info "Rank $rank: Global grid size = ($Nx, $Ny, $Nz), halo = ($Hx, $Hy, $Hz)"
flush(stdout); flush(stderr)

################################################################################
# Build distributed grid (to get local sizes and offsets)
################################################################################

@info "Rank $rank: Building distributed grid"
flush(stdout); flush(stderr)

dist_grid = load_tripolar_grid(grid_file, arch)
ug_dist = dist_grid isa ImmersedBoundaryGrid ? dist_grid.underlying_grid : dist_grid
local_Nx, local_Ny = ug_dist.Nx, ug_dist.Ny

# Compute global index ranges for FTS partitioning (mirrors grid.jl logic)
lsize = local_size(arch, (Nx, Ny, Nz))
nxlocal = concatenate_local_sizes(lsize, arch, 1)
nylocal = concatenate_local_sizes(lsize, arch, 2)
xrank = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
yrank = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)
x_offset = sum(nxlocal[1:xrank])
y_offset = sum(nylocal[1:yrank])
global_i_range = (1 + x_offset):(local_Nx + x_offset + 2Hx)
global_j_range = (1 + y_offset):(local_Ny + y_offset + 2Hy)

@info "Rank $rank: Local grid size = ($local_Nx, $local_Ny, $Nz), offsets = ($x_offset, $y_offset)"
@info "Rank $rank: nxlocal = $nxlocal, nylocal = $nylocal"
@info "Rank $rank: global_i_range = $global_i_range  (length $(length(global_i_range)))"
@info "Rank $rank: global_j_range = $global_j_range  (length $(length(global_j_range)))"
flush(stdout); flush(stderr)

################################################################################
# Save per-rank grid
################################################################################

@info "Rank $rank: Saving per-rank grid to $grid_partition_dir"
flush(stdout); flush(stderr)

# Save metrics directly from the distributed grid (already correctly partitioned)
metric_names = [
    :λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ,
    :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
    :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
    :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
    :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ,
]

grid_rank_file = joinpath(grid_partition_dir, "grid_rank$(rank).jld2")
jldopen(grid_rank_file, "w") do f
    # Dimensions
    f["Nx"] = local_Nx
    f["Ny"] = local_Ny
    f["Nz"] = Nz
    f["Hx"] = Hx
    f["Hy"] = Hy
    f["Hz"] = Hz
    f["Lz"] = gd["Lz"]

    # Coordinate and metric arrays (from the distributed grid)
    for name in metric_names
        f[string(name)] = Array(parent(getproperty(ug_dist, name)))
    end

    # z-faces (shared across all ranks)
    f["z_faces"] = gd["z_faces"]

    # Bottom height (from the distributed grid, already partitioned)
    if dist_grid isa ImmersedBoundaryGrid
        f["bottom"] = Array(parent(dist_grid.immersed_boundary.bottom_height.data))
    end

    # Conformal mapping parameters
    f["radius"] = ug_serial.radius
    f["north_poles_latitude"] = ug_serial.conformal_mapping.north_poles_latitude
    f["first_pole_longitude"] = ug_serial.conformal_mapping.first_pole_longitude
    f["southernmost_latitude"] = ug_serial.conformal_mapping.southernmost_latitude

    # Partition metadata
    f["rank"] = rank
    f["partition"] = "$(px)x$(py)"
end

@info "Rank $rank: Grid saved to $grid_rank_file"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

################################################################################
# Verify grid against serial
################################################################################

@info "Rank $rank: Verifying per-rank grid against serial"
flush(stdout); flush(stderr)

n_mismatches = 0
for name in metric_names
    dist_parent = Array(parent(getproperty(ug_dist, name)))
    local_saved = load(grid_rank_file, string(name))
    if local_saved != dist_parent
        n_diff = sum(local_saved .!= dist_parent)
        @error "Rank $rank: Grid metric $name MISMATCH ($n_diff cells differ)"
        n_mismatches += n_diff
    end
end
if n_mismatches == 0
    @info "Rank $rank: Grid verification PASSED (all metrics match distributed grid)"
else
    @error "Rank $rank: Grid verification FAILED ($n_mismatches total mismatches)"
end
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)

################################################################################
# Partition FTS data
################################################################################

# Determine which FTS files to partition based on VELOCITY_SOURCE
vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
fts_fields = [
    ("u_from_$(vs_prefix)", "u"),
    ("v_from_$(vs_prefix)", "v"),
    ("w_from_$(vs_prefix)", "w"),
    ("eta", "η"),
]

stop_time = 1year  # for Cyclical time indexing

for (file_prefix, field_name) in fts_fields
    monthly_file = joinpath(monthly_dir, "$(file_prefix)_monthly.jld2")
    if !isfile(monthly_file)
        @warn "Rank $rank: FTS file not found, skipping: $monthly_file"
        continue
    end

    @info "Rank $rank: Loading global FTS '$field_name' from $monthly_file"
    flush(stdout); flush(stderr)

    # Load on serial CPU grid with halos filled
    cpu_fts = FieldTimeSeries(
        monthly_file, field_name;
        architecture = CPU(), grid = serial_grid,
        backend = InMemory(),
        time_indexing = Cyclical(stop_time),
    )
    fill_halo_regions!(cpu_fts)

    # Extract per-rank parent data for each snapshot
    Nt = length(cpu_fts.times)
    loc = instantiated_location(cpu_fts)
    rank_fts_file = joinpath(fts_partition_dir, "$(file_prefix)_monthly_rank$(rank).jld2")

    # Build a temporary local field on the dist_grid at this FTS location and
    # use total_size (interior + halos) to determine the correct local parent
    # shape, so it's location- and topology-aware automatically.
    # TODO: refactor partition_data.jl to use Oceananigans' own partition/set!
    # routines end-to-end instead of slicing parent arrays manually. The hand
    # rolled global_i/j_range logic is fragile and missed Face-vs-Center
    # semantics on fold-owning ranks (see commit history for the bug).
    local_field = Field(loc, dist_grid)
    lpx, lpy, lpz = total_size(local_field)
    fts_i_range = (1 + x_offset):(x_offset + lpx)
    fts_j_range = (1 + y_offset):(y_offset + lpy)
    @info "Rank $rank: '$field_name' local parent = ($lpx, $lpy, $lpz), fts_i_range = $fts_i_range, fts_j_range = $fts_j_range"
    flush(stdout); flush(stderr)

    @info "Rank $rank: Saving $Nt snapshots of '$field_name' to $rank_fts_file"
    flush(stdout); flush(stderr)

    jldopen(rank_fts_file, "w") do f
        f["location"] = loc
        f["times"] = collect(cpu_fts.times)
        f["field_name"] = field_name
        f["Hx"] = Hx
        f["Hy"] = Hy
        f["Hz"] = Hz
        f["local_Nx"] = local_Nx
        f["local_Ny"] = local_Ny
        f["Nz"] = Nz
        f["rank"] = rank
        f["partition"] = "$(px)x$(py)"

        for n in 1:Nt
            serial_parent = Array(parent(cpu_fts[n].data))
            if n == 1
                @info "Rank $rank: '$field_name' serial_parent size = $(size(serial_parent)), slice = ($fts_i_range, $fts_j_range, :), result size = $(size(serial_parent[fts_i_range, fts_j_range, :]))"
                flush(stdout); flush(stderr)
            end
            local_parent = serial_parent[fts_i_range, fts_j_range, :]
            f["data/$n"] = local_parent
        end
    end

    # Verify first snapshot against serial
    local_saved = load(rank_fts_file, "data/1")
    serial_parent = Array(parent(cpu_fts[1].data))
    serial_slice = serial_parent[fts_i_range, fts_j_range, :]
    if local_saved == serial_slice
        @info "Rank $rank: FTS '$field_name' verification PASSED"
    else
        n_diff = sum(local_saved .!= serial_slice)
        @error "Rank $rank: FTS '$field_name' verification FAILED ($n_diff cells differ)"
    end

    flush(stdout); flush(stderr)
    MPI.Barrier(MPI.COMM_WORLD)
end

################################################################################
# Summary
################################################################################

@info "Rank $rank: partition_data.jl complete"
@info "  Grid files: $grid_partition_dir/grid_rank*.jld2"
@info "  FTS files:  $fts_partition_dir/*_monthly_rank*.jld2"
flush(stdout); flush(stderr)
MPI.Barrier(MPI.COMM_WORLD)
