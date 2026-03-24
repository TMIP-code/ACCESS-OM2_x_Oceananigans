"""
Common infrastructure for periodic steady-state solvers.

Included after `setup_model.jl` — assumes its globals are in scope:
  model, grid, arch, Δt, stop_time, prescribed_Δt, progress_message,
  year, on_architecture, compute_wet_mask
"""

using LinearAlgebra: norm
using Oceananigans.Simulations: reset!
using Printf: @sprintf

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

age3D_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))

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
    Φ!(age_out, age_in; source_rate = 1.0)

1-year forward map: runs the simulation for 1 year starting from `age_in`
(wet-cell vector) and writes the final age into `age_out` (wet-cell vector).

Set `source_rate = 0.0` to run the linear forward map (no constant interior
source), used for exact JVP computation.
"""
function Φ!(age_out, age_in; source_rate = 1.0)
    g_call_count[] += 1
    call_num = g_call_count[]
    t_start = time()
    @info "Φ! call #$call_num starting (source_rate=$source_rate)" norm_age_years = norm(age_in) / year max_age_years = maximum(abs, age_in) / year
    flush(stdout); flush(stderr)

    # Toggle the source rate on GPU/CPU before running
    copyto!(source_rate_arr, [source_rate])

    # Reset simulation for a fresh 1-year run
    reset!(simulation)
    simulation.stop_time = stop_time

    # CPU vec → CPU 3D → GPU 3D
    fill!(age3D_cpu, 0)
    age3D_cpu[idx] .= age_in
    copyto!(age3D_gpu, age3D_cpu)

    # Ensure all ranks are synchronised before setting initial condition
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)

    # Set initial condition and attach trace writer
    set!(model, age = age3D_gpu)

    if TRACE_SOLVER_HISTORY
        iter_str = @sprintf("%04d", call_num)
        trace_prefix = joinpath(trace_dir, "age_trace_iter_$(iter_str)_$(trace_job_id)")
        simulation.output_writers[:trace] = JLD2Writer(
            model, Dict("age" => model.tracers.age);
            schedule = TimeInterval(stop_time),
            filename = trace_prefix,
            overwrite_existing = true,
        )
    end

    # Run 1-year simulation
    run!(simulation)

    # Remove trace writer before next iteration
    if TRACE_SOLVER_HISTORY
        delete!(simulation.output_writers, :trace)
    end

    # Ensure all ranks finish simulation before extracting results
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)

    # GPU field → CPU 3D → CPU vec  (copyto! avoids intermediate Array allocation)
    copyto!(age3D_cpu, interior(model.tracers.age))
    age_out .= view(age3D_cpu, idx)

    elapsed = time() - t_start
    @info "Φ! call #$call_num done ($(round(elapsed; digits = 1))s)"
    flush(stdout); flush(stderr)
    return age_out
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
