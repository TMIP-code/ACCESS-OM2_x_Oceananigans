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
function load_fts(arch, file, name, grid; backend, time_indexing, cpu_grid = nothing, partition_dir = nothing)
    return FieldTimeSeries(file, name; architecture = arch, grid, backend, time_indexing)
end

function load_fts(arch::Distributed, file, name, grid; cpu_grid, backend, time_indexing, partition_dir = nothing)
    # Try loading from pre-partitioned per-rank file first
    if partition_dir !== nothing
        # Derive rank file name from the global file name
        # e.g., "u_from_mass_transport_monthly.jld2" → "u_from_mass_transport_monthly_rank0.jld2"
        basename_global = basename(file)
        basename_rank = replace(basename_global, ".jld2" => "_rank$(arch.local_rank).jld2")
        rank_file = joinpath(partition_dir, basename_rank)
        if isfile(rank_file)
            @info "Loading pre-partitioned FTS '$name' from $rank_file"
            return load_fts_from_rank_file(rank_file, name, grid; backend, time_indexing)
        else
            @warn "Rank file not found: $rank_file — falling back to global loading"
        end
    end

    # Fallback: load global FTS on CPU, partition via fold_set!, fill halos
    @info "Loading FTS '$name' via CPU grid for distributed partitioning"
    cpu_fts = FieldTimeSeries(
        file, name; architecture = CPU(), grid = cpu_grid,
        backend = InMemory(), time_indexing
    )
    dist_fts = FieldTimeSeries(
        instantiated_location(cpu_fts), grid, cpu_fts.times;
        backend = InMemory(), time_indexing,
        boundary_conditions = cpu_fts.boundary_conditions,
    )
    conformal_mapping = grid.underlying_grid.conformal_mapping
    y_loc = instantiated_location(cpu_fts)[2]
    for n in eachindex(cpu_fts.times)
        fold_set!(dist_fts[n], Array(interior(cpu_fts[n])), conformal_mapping, typeof(y_loc))
    end
    fill_halo_regions!(dist_fts)
    return dist_fts
end

"""
    load_fts_from_rank_file(rank_file, name, grid; backend, time_indexing)

Load a FieldTimeSeries from a pre-partitioned per-rank JLD2 file.
The file contains parent data (including halos) for each time snapshot,
sliced from the serial global data by partition_data.jl.
No fill_halo_regions! needed — halos are already correct.
"""
function load_fts_from_rank_file(rank_file, name, grid; backend, time_indexing)
    loc, times, snapshots = jldopen(rank_file, "r") do f
        loc = Tuple(f["location"])
        times = f["times"]
        Nt = length(times)
        snaps = [f["data/$n"] for n in 1:Nt]
        (loc, times, snaps)
    end

    # Create empty FTS on the distributed grid
    dist_fts = FieldTimeSeries(loc, grid, times; backend = InMemory(), time_indexing)

    # Copy pre-partitioned parent data directly into each snapshot.
    # Use copyto! to handle CPU→GPU transfer when grid is on GPU.
    for n in eachindex(times)
        copyto!(parent(dist_fts[n].data), snapshots[n])
    end

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
    load_distributed_snapshot(dir, field_name, duration_tag, iter_key, px, py, Nx, Ny; halo=(0,0,0)) -> (data, time)

Load and stitch distributed rank files for a single snapshot into a global array.
Each rank has its own JLD2Writer file: `{field}_{tag}_rank{r}.jld2`.
Partition layout is px × py. Oceananigans rank ordering: rank = i * Ry + j.

If `halo` is nonzero, strips halo regions from each rank before stitching
(needed when data was saved with `with_halos=true`).
The stitched result is trimmed to `(Nx, Ny, :)` to match serial interior.
"""
function load_distributed_snapshot(dir, field_name, duration_tag, iter_key, px, py, Nx, Ny; halo = (0, 0, 0))
    nranks = px * py
    Hx, Hy, Hz = halo

    # Determine per-rank interior sizes and z-dimension from rank files
    rank_interior_nx = zeros(Int, px)
    rank_interior_ny = zeros(Int, py)
    nz_data = 0
    ndims_data = 0
    for i in 1:px
        rank = (i - 1) * py
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")
        jldopen(filepath, "r") do f
            sample = f["timeseries/$(field_name)/$iter_key"]
            rank_interior_nx[i] = size(sample, 1) - 2Hx
            if i == 1
                ndims_data = ndims(sample)
                nz_raw = ndims_data >= 3 ? size(sample, 3) : 1
                # Only strip z-halos for true 3D fields (nz > 2Hz).
                # 2D fields like eta are stored as (nx, ny, 1) — no z-halos.
                nz_data = nz_raw > 2Hz ? nz_raw - 2Hz : nz_raw
            end
        end
    end
    for j in 1:py
        rank = j - 1
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        jldopen(filepath, "r") do f
            rank_interior_ny[j] = size(f["timeseries/$(field_name)/$iter_key"], 2) - 2Hy
        end
    end

    Nx_full = sum(rank_interior_nx)
    Ny_full = sum(rank_interior_ny)
    x_offsets = cumsum([0; rank_interior_nx[1:(end - 1)]])
    y_offsets = cumsum([0; rank_interior_ny[1:(end - 1)]])

    global_data = zeros(Float64, Nx_full, Ny_full, nz_data)
    t = nothing

    for rank in 0:(nranks - 1)
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")

        i_rank = div(rank, py) + 1
        j_rank = mod(rank, py) + 1

        jldopen(filepath, "r") do f
            local_full = f["timeseries/$(field_name)/$iter_key"]
            if t === nothing
                t = f["timeseries/t/$iter_key"]
            end

            # Strip halos: interior at [Hx+1:end-Hx, Hy+1:end-Hy, Hz+1:end-Hz]
            # For 2D fields (nz_data ≤ 2Hz), no z-halos to strip
            nx_local = rank_interior_nx[i_rank]
            ny_local = rank_interior_ny[j_rank]
            nz_raw = size(local_full, 3)
            z_has_halos = nz_raw > 2Hz
            z_range = z_has_halos ? ((Hz + 1):(Hz + nz_data)) : (1:nz_data)
            local_data = if ndims_data >= 3
                local_full[(Hx + 1):(Hx + nx_local), (Hy + 1):(Hy + ny_local), z_range]
            else
                local_full[(Hx + 1):(Hx + nx_local), (Hy + 1):(Hy + ny_local)]
            end

            x_start = x_offsets[i_rank] + 1
            x_end = x_offsets[i_rank] + nx_local
            y_start = y_offsets[j_rank] + 1
            y_end = y_offsets[j_rank] + ny_local

            if ndims_data >= 3
                global_data[x_start:x_end, y_start:y_end, :] .= local_data
            else
                global_data[x_start:x_end, y_start:y_end, 1] .= local_data
            end
        end
    end

    return global_data[1:Nx, 1:Ny, :], t
end
