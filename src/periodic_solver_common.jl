"""
Common infrastructure for periodic steady-state solvers.

Included after `setup_model.jl` — assumes its globals are in scope:
  model, grid, arch, Δt, stop_time, prescribed_Δt, progress_message,
  year, on_architecture, compute_wet_mask
"""

using LinearAlgebra: norm
using Oceananigans.Simulations: reset!
using Oceananigans.DistributedComputations: Distributed
using Printf: @sprintf
using MPI

################################################################################
# Trace solver history configuration
################################################################################

TRACE_SOLVER_HISTORY = lowercase(get(ENV, "TRACE_SOLVER_HISTORY", "no")) == "yes"
trace_job_id = get(ENV, "PBS_JOBID", "interactive")

if TRACE_SOLVER_HISTORY
    trace_dir = solver_output_dir  # defined by caller before include
    mkpath(trace_dir)
    @info "TRACE_SOLVER_HISTORY enabled — saving iterates to $trace_dir (job_id=$trace_job_id)"
else
    trace_dir = ""
    @info "TRACE_SOLVER_HISTORY disabled (set TRACE_SOLVER_HISTORY=yes to enable)"
end
flush(stdout); flush(stderr)

################################################################################
# Simulation (no output writers)
################################################################################

include("setup_simulation.jl")

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

################################################################################
# Compute wet cell mask & preallocate CPU/GPU buffers
################################################################################

@info "Computing wet cell mask"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: $Nidx"
Nx′, Ny′, Nz′ = size(wet3D)
flush(stdout); flush(stderr)

# Aliases — Φ!_body uses the *_local names to make scatter/gather semantics explicit.
# In serial these alias the only-cell-mask variables; in distributed they are the
# rank-local versions (the global versions are built later on rank 0).
idx_local = idx
age3D_local_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_local_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))
age3D_cpu = age3D_local_cpu
age3D_gpu = age3D_local_gpu

################################################################################
# Cell volumes and volume-weighted norm
################################################################################

@info "Computing cell volumes for volume-weighted norm"
flush(stdout); flush(stderr)

grid_cpu = on_architecture(CPU(), grid)
v1D = interior(compute_volume(grid_cpu))[idx]

# make_vol_norm is defined in shared_functions.jl

vol_norm = make_vol_norm(v1D, year)

################################################################################
# MPI scatter/gather infrastructure for partitioned NK
################################################################################
#
# Rank 0 drives NonlinearSolve.solve on a global 1D wet-cell vector
# (`age_global`, length `Nidx_global`). Each Φ! invocation scatters that
# vector to per-rank slabs via `scatter!`, runs the 1-year simulation
# collectively across all ranks, and gathers the result back via `gather!`.
# In serial mode `Nidx_local == Nidx_global` and scatter!/gather! collapse to
# `copyto!`.

include("mpi_partition_io.jl")

# Local 1D wet-cell vector (every rank)
Nidx_local = Nidx
age_local_vec = Vector{Float64}(undef, Nidx_local)

if arch isa Distributed
    @assert px == 1 "Partitioned NK requires PARTITION_X=1 (got px=$px)"
    COMM = MPI.COMM_WORLD
    rank = MPI.Comm_rank(COMM)
else
    COMM = nothing
    rank = 0
end

if arch isa Distributed && rank == 0
    @info "[rank 0] Building partitioned-NK setup (global mask, v1D, permutation)"
    flush(stdout); flush(stderr)
    grid_cpu_global = load_tripolar_grid(grid_file, CPU())
    global_mask = compute_wet_mask(grid_cpu_global)
    wet3D_global = global_mask.wet3D
    idx_global = global_mask.idx
    Nidx_global = global_mask.Nidx
    Nx′_global, Ny′_global, Nz′_global = size(wet3D_global)

    # Replace rank-local v1D with global v1D for the volume-weighted norm.
    # vol_norm only fires on rank 0 once the rank-0/rank-1 split lands in step 4.
    v1D = interior(compute_volume(grid_cpu_global))[idx_global]
    vol_norm = make_vol_norm(v1D, year)

    Ny_global = size(wet3D_global, 2)
    partition_y_sizes = collect(Oceananigans.DistributedComputations.local_sizes(Ny_global, arch.partition.y))
    Ny_sum = sum(partition_y_sizes)
    Ny_sum != Ny_global && (partition_y_sizes[end] += Ny_global - Ny_sum)

    perm, counts, displs = build_global_permutation(wet3D_global, partition_y_sizes)
    @info "[rank 0] Global setup complete" Nidx_global Ny_global partition_y_sizes counts

    send_buf = Vector{Float64}(undef, Nidx_global)
    recv_buf = Vector{Float64}(undef, Nidx_global)
    age_global = Vector{Float64}(undef, Nidx_global)

    # Free big rank-0-only buffers; keep wet3D_global + idx_global for the final save
    grid_cpu_global = nothing
    flush(stdout); flush(stderr)
elseif arch isa Distributed
    # Rank > 0 stubs (unused; scatter!/gather! distributed paths skip the rank-0 blocks)
    wet3D_global = nothing
    idx_global = nothing
    Nidx_global = 0
    Nx′_global = 0; Ny′_global = 0; Nz′_global = 0
    perm = Int[]
    counts = Int[]
    displs = Int[]
    send_buf = Float64[]
    recv_buf = Float64[]
    age_global = Float64[]
else
    # Serial: global aliases local
    wet3D_global = wet3D
    idx_global = idx
    Nidx_global = Nidx_local
    Nx′_global, Ny′_global, Nz′_global = (Nx′, Ny′, Nz′)
    age_global = age_local_vec
    perm = Int[]
    counts = Int[]
    displs = Int[]
    send_buf = Float64[]
    recv_buf = Float64[]
end

################################################################################
# Initial age loading (INITIAL_AGE env var)
################################################################################

"""
    load_initial_age(idx, Nidx, outputdir, model_config; year)

Load the initial age vector based on the INITIAL_AGE environment variable.

Returns a Vector{Float64} of length `Nidx` (wet cells) in **seconds**.

- `INITIAL_AGE="TMage"` (default): load transport-matrix-computed steady-state age
- `INITIAL_AGE="0"`: zeros
  (tries ParU, UMFPACK, then generic full files in the matrices directory)
"""
function load_initial_age(idx, Nidx, outputdir, model_config; year)
    INITIAL_AGE = get(ENV, "INITIAL_AGE", "TMage")
    age_init_vec = zeros(Nidx)

    if INITIAL_AGE == "TMage"
        TM_SOURCE = get(ENV, "TM_SOURCE", "const")
        matrices_dir = joinpath(outputdir, "TM", model_config, TM_SOURCE)
        # Try candidate files in priority order
        candidates = [
            "steady_age_full_$(solver)_$(mp).jld2"
                for mp in ("raw", "dropzeros", "symfill", "symdrop")
                for solver in ("ParU", "UMFPACK", "Pardiso")
        ]
        loaded = false
        for candidate in candidates
            fpath = joinpath(matrices_dir, candidate)
            if isfile(fpath)
                @info "Loading TM age from $fpath"
                flush(stdout); flush(stderr)
                age_data = load(fpath, "age")
                # Matrix age files store age in years → convert to seconds
                age_init_vec .= view(age_data, idx) .* year
                @info "TM age loaded" max_years = maximum(abs, age_init_vec) / year mean_years = mean(age_init_vec) / year
                loaded = true
                break
            end
        end
        if !loaded
            @warn "INITIAL_AGE=TMage but no matrix age file found in $matrices_dir — starting from zeros"
        end
    elseif INITIAL_AGE != "0"
        # Treat as a file path (backwards-compatible with WARM_START_FILE concept)
        if isfile(INITIAL_AGE)
            @info "Loading initial age from file: $INITIAL_AGE"
            flush(stdout); flush(stderr)
            age_data = load(INITIAL_AGE, "age")
            # Detect units: if max age < 100_000, assume years; otherwise seconds
            max_val = maximum(abs, view(age_data, idx))
            if max_val < 100_000
                @info "Detected age in years — converting to seconds"
                age_init_vec .= view(age_data, idx) .* year
            else
                age_init_vec .= view(age_data, idx)
            end
            @info "Initial age loaded" max_years = maximum(abs, age_init_vec) / year mean_years = mean(age_init_vec) / year
        else
            @warn "INITIAL_AGE file not found: $INITIAL_AGE — starting from zeros"
        end
    else
        @info "Starting from zero initial guess (set INITIAL_AGE=TMage or path to warm-start)"
    end
    flush(stdout); flush(stderr)

    return age_init_vec
end

################################################################################
# Forward map Φ! and residual G!
################################################################################

g_call_count = Ref(0)
jvp_call_count = Ref(0)

"""
    Φ!_body(age_out, age_in; source_rate = 1.0)

1-year forward map body. `age_in` and `age_out` are global 1D wet-cell
vectors on rank 0 (length `Nidx_global`); on rank > 0 they are unused
dummies. Internally, the global vector is scattered to per-rank local 1D
buffers, packed into a local 3D field, run for 1 year, then the result is
extracted back to a local 1D buffer and gathered to the global vector on
rank 0. In serial mode `Nidx_local == Nidx_global` and scatter!/gather!
collapse to `copyto!`.

Set `source_rate = 0.0` to run the linear forward map (no constant interior
source), used for exact JVP computation.

The `Φ!` wrapper (defined alongside the rank-0 driver loop) prefixes an
`MPI.Bcast!` so rank > 0 worker loops join each `Φ!_body` call.
"""
function Φ!_body(age_out, age_in; source_rate = 1.0)
    g_call_count[] += 1
    call_num = g_call_count[]
    t_start = time()
    if rank == 0
        @info "Φ! call #$call_num starting (source_rate=$source_rate)" norm_age_years = norm(age_in) / year max_age_years = maximum(abs, age_in) / year
        flush(stdout); flush(stderr)
    end

    # Toggle the source rate on GPU/CPU before running
    copyto!(source_rate_arr, [source_rate])

    # Reset simulation for a fresh 1-year run
    reset!(simulation)
    simulation.stop_time = stop_time

    # Scatter rank-0 global vector → per-rank local 1D buffer (no-op in serial)
    scatter!(age_local_vec, age_in, arch)

    # Local CPU 1D vec → CPU 3D → GPU 3D
    fill!(age3D_local_cpu, 0)
    age3D_local_cpu[idx_local] .= age_local_vec
    copyto!(age3D_local_gpu, age3D_local_cpu)

    # Ensure all ranks are synchronised before setting initial condition
    arch isa Distributed && MPI.Barrier(COMM)

    # Set initial condition (local field set; no MPI inside)
    set!(model, age = age3D_local_gpu)

    if TRACE_SOLVER_HISTORY
        iter_str = @sprintf("%04d", call_num)
        trace_prefix = joinpath(trace_dir, "age_trace_iter_$(iter_str)_$(trace_job_id)")
        simulation.output_writers[:trace] = JLD2Writer(
            model, Dict("age" => model.tracers.age);
            schedule = TimeInterval(stop_time),
            filename = trace_prefix,
            overwrite_existing = true,
            array_type = Array{Float64},
        )
    end

    # Run 1-year simulation (collective across ranks in distributed)
    run!(simulation)

    if TRACE_SOLVER_HISTORY
        delete!(simulation.output_writers, :trace)
    end

    # Ensure all ranks finish simulation before extracting results
    arch isa Distributed && MPI.Barrier(COMM)

    # GPU field → CPU 3D → CPU local 1D vec
    copyto!(age3D_local_cpu, interior(model.tracers.age))
    age_local_vec .= view(age3D_local_cpu, idx_local)

    # Gather per-rank local 1D buffer → rank-0 global vector (no-op in serial)
    gather!(age_out, age_local_vec, arch)

    elapsed = time() - t_start
    if rank == 0
        @info "Φ! call #$call_num done ($(round(elapsed; digits = 1))s)"
        flush(stdout); flush(stderr)
    end
    return age_out
end

"""
    Φ!(age_out, age_in; source_rate = 1.0)

Thin wrapper around `Φ!_body` that, in distributed mode, prefixes an
`MPI.Bcast!` from rank 0 to wake the rank > 0 worker loop (defined in
`solve_periodic_NK.jl`). Only rank 0 ever calls this wrapper; rank > 0
calls `Φ!_body` directly from inside the worker loop after receiving the
matching Bcast.
"""
function Φ!(age_out, age_in; source_rate = 1.0)
    if arch isa Distributed
        # [continue_flag, source_rate] — rank > 0 worker loop reads this
        MPI.Bcast!([1.0, source_rate], 0, COMM)
    end
    return Φ!_body(age_out, age_in; source_rate = source_rate)
end

"""
    G!(dage, age, p)

1-year drift residual: G(x) = Φ(x) − x.
"""
function G!(dage, age, p)
    Φ!(dage, age) # dage <- Φ(age)       = age after 1 year
    dage .-= age  # dage <- Φ(age) - age = age drift after after 1 year
    @info "G! residual" vol_rms_drift_years = vol_norm(dage) max_drift_years = maximum(abs, dage) / year mean_drift_years = mean(abs, dage) / year
    flush(stdout); flush(stderr)
    return dage
end

################################################################################
# Exact JVP via source_rate toggle
################################################################################

"""
    jvp!(Jv, v, age, p)

Exact Jacobian-vector product: J_G · v = Φ(v; source_rate=0) − v.

Runs the linear forward map (source_rate=0.0, no constant interior source)
to compute the Jacobian of Φ applied to `v`.

`age` and `p` are unused but present for the NonlinearSolve JVP signature.
"""
function jvp!(Jv, v, age, p)
    Φ!(Jv, v; source_rate = 0.0)
    Jv .-= v
    return Jv
end
