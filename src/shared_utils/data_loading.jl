################################################################################
# Data loading helpers
#
# Extracted from shared_functions.jl — distributed-aware FieldTimeSeries/MLD
# loading and part-file I/O for split JLD2 output.
################################################################################

using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Fields: location, instantiated_location
using Oceananigans.Grids: on_architecture, znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.OrthogonalSphericalShellGrids: fold_set!
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory
using Oceananigans.Architectures: CPU, GPU, architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!

################################################################################
# Distributed-aware loading helpers
################################################################################

"""
    load_fts(arch, file, name, grid; backend, time_indexing, cpu_grid=nothing)

Load a `FieldTimeSeries` from a JLD2 file. For `Distributed` architectures, loading
directly fails because the `FieldTimeSeries` constructor passes global-sized file data to
`offset_data`, which expects local-sized data. Workaround: load on a serial CPU grid first,
then partition to distributed ranks via `fold_set!`.
"""
function load_fts(arch, file, name, grid; backend, time_indexing, cpu_grid = nothing)
    return FieldTimeSeries(file, name; architecture = arch, grid, backend, time_indexing)
end

function load_fts(arch::Distributed, file, name, grid; cpu_grid, backend, time_indexing)
    @info "Loading FTS '$name' via CPU grid for distributed partitioning"
    cpu_fts = FieldTimeSeries(
        file, name; architecture = CPU(), grid = cpu_grid,
        backend = InMemory(), time_indexing
    )
    # Pass boundary_conditions from the CPU FTS so the distributed FTS gets the correct
    # FPivotZipperBoundaryCondition sign (e.g., -1 for u, v velocities at the north fold).
    # Without this, the default BCs omit the sign and fill_halo_regions! produces wrong halos.
    dist_fts = FieldTimeSeries(
        instantiated_location(cpu_fts), grid, cpu_fts.times;
        backend = InMemory(), time_indexing,
        boundary_conditions = cpu_fts.boundary_conditions,
    )
    # Use fold_set! directly because DistributedTripolarField dispatch doesn't match
    # ImmersedBoundaryGrid-wrapped fields (Oceananigans bug: DistributedTripolarField
    # uses DistributedTripolarGrid but not DistributedTripolarGridOfSomeKind).
    conformal_mapping = grid.underlying_grid.conformal_mapping
    y_loc = instantiated_location(cpu_fts)[2]
    for n in eachindex(cpu_fts.times)
        fold_set!(dist_fts[n], Array(interior(cpu_fts[n])), conformal_mapping, typeof(y_loc))
    end
    fill_halo_regions!(dist_fts)
    return dist_fts
end

"""
    load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)

Load MLD data and create a vertical diffusivity field. For `Distributed` architectures,
keeps data on CPU so `set!` dispatches to `fold_set!` with global arrays.
"""
function load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)
    mld_ds = open_dataset(mld_file)
    mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
    z_center = znodes(grid, Center(), Center(), Center())
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    κVField = CenterField(grid)
    set!(κVField, κVML * is_mld + κVBG * .!is_mld)
    return κVField
end

function load_mld_diffusivity(arch::Distributed, grid, mld_file, κVML, κVBG, Nz)
    mld_ds = open_dataset(mld_file)
    mld_data = -replace(readcubedata(mld_ds.mld).data, NaN => 0.0)
    z_center = collect(znodes(grid, Center(), Center(), Center()))
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    κVField = CenterField(grid)
    # Use fold_set! directly (DistributedTripolarField dispatch doesn't match ImmersedBoundaryGrid)
    conformal_mapping = grid.underlying_grid.conformal_mapping
    fold_set!(κVField, Array(κVML * is_mld + κVBG * .!is_mld), conformal_mapping, Center)
    fill_halo_regions!(κVField)
    return κVField
end


################################################################################
# Part-file loading helpers for split JLD2 output
################################################################################

# Part files are monthly snapshots produced by file_splitting:
#   part 1 = snapshot at t=0
#   part 2 = snapshot at t=1 month
#   ...
#   part 13 = snapshot at t=1 year

"""
    load_serial_part(dir, field_name, duration_tag, part) -> (data, time)

Load a single serial part file (one monthly snapshot).
Returns the 3D data array and the snapshot time.
"""
function load_serial_part(dir, field_name, duration_tag, part)
    filepath = joinpath(dir, "$(field_name)_$(duration_tag)_part$(part).jld2")
    isfile(filepath) || error("Part file not found: $filepath")
    return jldopen(filepath, "r") do f
        iters = keys(f["timeseries/$(field_name)"])
        iter = first(filter(k -> k != "serialized", iters))
        data = f["timeseries/$(field_name)/$iter"]
        t = f["timeseries/t/$iter"]
        return data, t
    end
end

"""
    load_distributed_part(dir, field_name, duration_tag, part, px, py, Nx, Ny[, Nz]) -> (data, time)

Load and stitch distributed rank files for a single part into a global array.
Partition layout is px × py (e.g., 2×2 for 4 GPUs/CPUs).
`Nx, Ny` is the interior size (e.g., from `size(wet3D)`).
`Nz` is optional — the z-dimension is auto-detected from rank 0 data.
Handles 2D fields (e.g., eta) and fields with different z-sizes (e.g., w at Nz+1).
The distributed grid may include a fold row (Ny_full = Ny + 1 for tripolar grids);
the stitched result is trimmed to `(Nx, Ny, :)` to match serial interior output.
"""
function load_distributed_part(dir, field_name, duration_tag, part, px, py, Nx, Ny, Nz = nothing)
    nranks = px * py

    # Oceananigans rank2index uses column-major ordering:
    #   rank = i * Ry + j  (0-indexed, Rz=1)
    #   i = div(rank, py),  j = mod(rank, py)
    # So x-column i contains ranks {i*py, i*py+1, ..., i*py+py-1}
    # and y-row j contains ranks {j, j+py, j+2*py, ...}

    # Determine per-rank sizes from one rank per x-column and one per y-row
    # Also auto-detect z-size and dimensionality from rank 0
    rank_sizes_x = zeros(Int, px)
    rank_sizes_y = zeros(Int, py)
    nz_data = 0
    ndims_data = 0
    for i in 1:px
        rank = (i - 1) * py  # first rank in x-column i
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")
        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            sample = f["timeseries/$(field_name)/$iter"]
            rank_sizes_x[i] = size(sample, 1)
            if i == 1
                ndims_data = ndims(sample)
                nz_data = ndims_data >= 3 ? size(sample, 3) : 1
            end
        end
    end
    for j in 1:py
        rank = j - 1  # first rank in y-row j
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            rank_sizes_y[j] = size(f["timeseries/$(field_name)/$iter"], 2)
        end
    end

    Nx_full = sum(rank_sizes_x)
    Ny_full = sum(rank_sizes_y)
    x_offsets = cumsum([0; rank_sizes_x[1:(end - 1)]])
    y_offsets = cumsum([0; rank_sizes_y[1:(end - 1)]])

    global_data = zeros(Float64, Nx_full, Ny_full, nz_data)
    t = nothing

    for rank in 0:(nranks - 1)
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")

        # Oceananigans column-major: i = div(rank, py), j = mod(rank, py)
        i_rank = div(rank, py) + 1
        j_rank = mod(rank, py) + 1

        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            local_data = f["timeseries/$(field_name)/$iter"]
            if t === nothing
                t = f["timeseries/t/$iter"]
            end

            x_start = x_offsets[i_rank] + 1
            x_end = x_offsets[i_rank] + rank_sizes_x[i_rank]
            y_start = y_offsets[j_rank] + 1
            y_end = y_offsets[j_rank] + rank_sizes_y[j_rank]

            if ndims_data >= 3
                global_data[x_start:x_end, y_start:y_end, :] .= local_data
            else
                global_data[x_start:x_end, y_start:y_end, 1] .= local_data
            end
        end
    end

    # Trim to interior size (exclude fold row for tripolar grids)
    return global_data[1:Nx, 1:Ny, :], t
end
