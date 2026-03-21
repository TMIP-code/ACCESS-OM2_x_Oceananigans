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
    load_serial_snapshot(dir, field_name, duration_tag, iter_key) -> (data, time)

Load a single snapshot from a serial JLD2Writer file (one file per variable,
iterations stored as keys under `timeseries/{field}/{iter}`).
"""
function load_serial_snapshot(dir, field_name, duration_tag, iter_key)
    filepath = joinpath(dir, "$(field_name)_$(duration_tag).jld2")
    isfile(filepath) || error("Serial file not found: $filepath")
    return jldopen(filepath, "r") do f
        data = f["timeseries/$(field_name)/$iter_key"]
        t = f["timeseries/t/$iter_key"]
        return data, t
    end
end

"""
    list_iterations(dir, field_name, duration_tag) -> Vector{String}

List all iteration keys in a serial JLD2Writer file, sorted numerically.
"""
function list_iterations(dir, field_name, duration_tag)
    filepath = joinpath(dir, "$(field_name)_$(duration_tag).jld2")
    isfile(filepath) || error("File not found: $filepath")
    return jldopen(filepath, "r") do f
        iters = collect(keys(f["timeseries/$(field_name)"]))
        filter!(k -> k != "serialized", iters)
        sort!(iters; by = k -> parse(Int, k))
        return iters
    end
end

"""
    load_distributed_snapshot(dir, field_name, duration_tag, iter_key, px, py, Nx, Ny) -> (data, time)

Load and stitch distributed rank files for a single snapshot into a global array.
Each rank has its own JLD2Writer file: `{field}_{tag}_rank{r}.jld2`.
Partition layout is px × py. Oceananigans rank ordering: rank = i * Ry + j.
The stitched result is trimmed to `(Nx, Ny, :)` to match serial interior.
"""
function load_distributed_snapshot(dir, field_name, duration_tag, iter_key, px, py, Nx, Ny)
    nranks = px * py

    # Determine per-rank sizes and z-dimension from rank files
    rank_sizes_x = zeros(Int, px)
    rank_sizes_y = zeros(Int, py)
    nz_data = 0
    ndims_data = 0
    for i in 1:px
        rank = (i - 1) * py
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")
        jldopen(filepath, "r") do f
            sample = f["timeseries/$(field_name)/$iter_key"]
            rank_sizes_x[i] = size(sample, 1)
            if i == 1
                ndims_data = ndims(sample)
                nz_data = ndims_data >= 3 ? size(sample, 3) : 1
            end
        end
    end
    for j in 1:py
        rank = j - 1
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        jldopen(filepath, "r") do f
            rank_sizes_y[j] = size(f["timeseries/$(field_name)/$iter_key"], 2)
        end
    end

    Nx_full = sum(rank_sizes_x)
    Ny_full = sum(rank_sizes_y)
    x_offsets = cumsum([0; rank_sizes_x[1:(end - 1)]])
    y_offsets = cumsum([0; rank_sizes_y[1:(end - 1)]])

    global_data = zeros(Float64, Nx_full, Ny_full, nz_data)
    t = nothing

    for rank in 0:(nranks - 1)
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")

        i_rank = div(rank, py) + 1
        j_rank = mod(rank, py) + 1

        jldopen(filepath, "r") do f
            local_data = f["timeseries/$(field_name)/$iter_key"]
            if t === nothing
                t = f["timeseries/t/$iter_key"]
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

    return global_data[1:Nx, 1:Ny, :], t
end
