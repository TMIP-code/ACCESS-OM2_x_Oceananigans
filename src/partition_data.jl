"""
Pre-partition grid and FTS data for distributed (MPI) simulations.

Runs under MPI on CPU. Each rank writes its own JLD2 files containing
local grid arrays and FTS snapshots with correct halos, sliced directly
from the serial global data. This eliminates the need for runtime
`fold_set!` + `fill_halo_regions!` on distributed FTS fields.

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
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: on_architecture
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
time_window = get(ENV, "TIME_WINDOW", "1960-1979")

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

@info "Rank $rank: Local grid size = ($local_Nx, $local_Ny, $Nz)"
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
if VELOCITY_SOURCE == "totaltransport"
    fts_fields = [
        ("u_from_total_transport", "u"),
        ("v_from_total_transport", "v"),
        ("w_from_total_transport", "w"),
        ("eta", "η"),
    ]
elseif VELOCITY_SOURCE == "cgridtransports"
    fts_fields = [
        ("u_from_mass_transport", "u"),
        ("v_from_mass_transport", "v"),
        ("w_from_mass_transport", "w"),
        ("eta", "η"),
    ]
else
    fts_fields = [
        ("u_interpolated", "u"),
        ("v_interpolated", "v"),
        ("w", "w"),
        ("eta", "η"),
    ]
end

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
            nz_fts = size(serial_parent, 3)
            # For 2D fields (eta), z-dim may be 1 — no z-halos to slice
            if nz_fts <= 2Hz
                local_parent = serial_parent[global_i_range, global_j_range, :]
            else
                local_parent = serial_parent[global_i_range, global_j_range, :]
            end
            f["data/$n"] = local_parent
        end
    end

    # Verify first snapshot against serial
    local_saved = load(rank_fts_file, "data/1")
    serial_parent = Array(parent(cpu_fts[1].data))
    nz_fts = size(serial_parent, 3)
    serial_slice = serial_parent[global_i_range, global_j_range, :]
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
