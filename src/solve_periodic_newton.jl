"""
Solve for the periodic steady-state age using Newton-GMRES.

Finds x such that G(x) = Φ(x) - x = 0, where Φ(x) is the result of running
the model for 1 year from initial condition x. Uses a lump-and-spray
preconditioner (Bardin et al., 2014) and either matrix-based or finite-difference
JVP for GMRES.

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=04:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/solve_periodic_newton.jl")
```

Environment variables (in addition to setup_model.jl):
  JVP_METHOD – matrix | finitediff  (default: matrix)
               matrix:     approximate JVP using transport matrix M (fast, sparse matvec)
               finitediff: finite-difference JVP via AutoFiniteDiff (slow, extra G! evals)
"""

include("setup_model.jl")

using NonlinearSolve
using LinearAlgebra
using SparseArrays
using OceanTransportMatrixBuilder
using Oceananigans.Simulations: reset!
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using KernelAbstractions: @kernel, @index
import Pardiso
const nprocs = 12

################################################################################
# Configuration
################################################################################

JVP_METHOD = get(ENV, "JVP_METHOD", "matrix")
(JVP_METHOD ∈ ("matrix", "finitediff")) || error("JVP_METHOD must be one of: matrix, finitediff (got: $JVP_METHOD)")

matrices_dir = joinpath(outputdir, "matrices", "$(VELOCITY_SOURCE)_constant")

@info "Newton-GMRES periodic solver configuration"
@info "- JVP_METHOD  = $JVP_METHOD"
@info "- matrices_dir = $matrices_dir"
flush(stdout)

################################################################################
# Simulation (minimal output — no field writers during periodic solve)
################################################################################

@info "Creating simulation (no output writers)"
flush(stdout)

set!(model, age = Returns(0.0))

simulation = Simulation(model; Δt, stop_time)

function progress_message(sim)
    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year)
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout)
    return @info @sprintf(
        "  sim iter: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d), mean(age) = %.1e, wall: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age, idx_max.I..., mean_age, walltime
    )
end

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

################################################################################
# Load pre-built transport matrix M from disk
################################################################################

M_file = joinpath(matrices_dir, "M.jld2")
@info "Loading transport matrix from $M_file"
flush(stdout)
M = load(M_file, "M")
@info "Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
flush(stdout)

################################################################################
# Compute wet cell indexing
################################################################################

@info "Computing wet cell mask"
flush(stdout)

fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
wet3D = .!isnan.(interior(on_architecture(CPU(), fNaN)))
idx = findall(wet3D)
Nidx = length(idx)
@info "Number of wet cells: $Nidx (matrix size: $(size(M, 1)))"
@assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) != matrix rows ($(size(M, 1)))"
Nx′, Ny′, Nz′ = size(wet3D)
flush(stdout)

################################################################################
# Compute cell volumes
################################################################################

@info "Computing cell volumes"
flush(stdout)

@kernel function compute_volume!(vol, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds vol[i, j, k] = volume(i, j, k, grid, Center(), Center(), Center())
end

function compute_volume(grid)
    vol = CenterField(grid)
    (Nxv, Nyv, Nzv) = size(vol)
    kp = KernelParameters(1:Nxv, 1:Nyv, 1:Nzv)
    launch!(CPU(), grid, kp, compute_volume!, vol, grid)
    return vol
end

grid_cpu = on_architecture(CPU(), grid)
v1D = interior(compute_volume(grid_cpu))[idx]

################################################################################
# LUMP, SPRAY, and coarsened Jacobian
################################################################################

@info "Computing LUMP and SPRAY matrices"
flush(stdout)
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
Mc = LUMP * M * SPRAY
@info "Coarsened Jacobian Mc: $(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))"
flush(stdout)

################################################################################
# Preconditioner setup (Pardiso, CPU-only)
################################################################################

@info "Setting up preconditioner"
flush(stdout)

if Pardiso.isstructurallysymmetric(Mc)
    matrix_type = Pardiso.REAL_SYM
    @show pardiso_solver = MKLPardisoIterate(; nprocs, matrix_type)
else
    @warn "Mc is not structurally symmetric; using UMFPACKFactorization()"
    @show pardiso_solver = UMFPACKFactorization()
end

# P = S Qc⁻¹ L - I  (Bardin et al., 2014)
# Q = stop_time * M
# Qc = L Q S = stop_time * Mc
if !@isdefined(MyPreconditioner)
    struct MyPreconditioner
        prob
    end
end

Qc = stop_time * Mc
Plprob = LinearProblem(Qc, ones(size(Qc, 1)))
Plprob = init(Plprob, pardiso_solver, rtol = 1.0e-12)
Pl = MyPreconditioner(Plprob)

Base.eltype(::MyPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
    Pl.prob.b = LUMP * x
    solve!(Pl.prob) # solves Qc u = b = L x
    x .= SPRAY * Pl.prob.u .- x # x <- (S Qc⁻¹ L - I) x = P x
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
    Pl.prob.b = LUMP * x
    solve!(Pl.prob) # solves Qc u = b = L x
    y .= SPRAY * Pl.prob.u .- x # y <- (S Qc⁻¹ L - I) x = P x
    return y
end

Pr = I
precs = Returns((Pl, Pr))

@info "Preconditioner ready"
flush(stdout)

################################################################################
# Preallocate buffers for G!
################################################################################

age3D_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))

################################################################################
# G! function: 1-year drift (runs simulation on GPU, solver on CPU)
################################################################################

g_call_count = Ref(0)

function G!(dage, age, p)
    g_call_count[] += 1
    call_num = g_call_count[]
    t_start = time()
    @info "G! call #$call_num starting" norm_age = norm(age) max_age = maximum(abs, age)
    flush(stdout)

    # Reset simulation for a fresh 1-year run
    reset!(simulation)
    simulation.stop_time = stop_time

    # CPU vec -> CPU 3D -> preallocated GPU 3D (no new GPU allocation)
    fill!(age3D_cpu, 0)
    age3D_cpu[idx] .= age
    copyto!(age3D_gpu, age3D_cpu)

    # Set model field and run 1-year simulation
    set!(model, age = age3D_gpu)
    run!(simulation)

    # GPU field -> CPU 3D -> CPU vec (drift = final - initial)
    age3D_cpu .= Array(interior(model.tracers.age))
    dage .= view(age3D_cpu, idx) .- age

    elapsed = time() - t_start
    @info "G! call #$call_num done ($(round(elapsed; digits = 1))s)" norm_drift = norm(dage) max_drift = maximum(abs, dage)
    flush(stdout)
    return dage
end

################################################################################
# JVP setup
################################################################################

if JVP_METHOD == "matrix"
    @info "Using matrix-based JVP: J ≈ stop_time * M (sparse matvec)"
    flush(stdout)

    # The true Jacobian of G(x) = Φ(x) - x is J_G = exp(M*T) - I ≈ M*T
    # (first-order approximation). Using this avoids expensive G! evaluations
    # during GMRES iterations (sparse matvec vs full year simulation).
    MT = stop_time * M

    function approximate_jvp!(Jv, v, u, p)
        return mul!(Jv, MT, v)
    end

    f! = NonlinearFunction(G!; jvp = approximate_jvp!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-10),
    )

elseif JVP_METHOD == "finitediff"
    @info "Using finite-difference JVP (AutoFiniteDiff)"
    @warn "This requires extra G! evaluations per GMRES iteration — much slower than matrix JVP"
    flush(stdout)

    f! = NonlinearFunction(G!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-10),
        jvp_autodiff = AutoFiniteDiff(),
    )
end

################################################################################
# Nonlinear solve: Newton-GMRES
################################################################################

@info "Solving nonlinear problem with Newton-GMRES"
@info "- JVP method: $JVP_METHOD"
@info "- Preconditioner: lump-and-spray (Bardin et al., 2014)"
@info "- abstol = $(0.001 * stop_time)"
flush(stdout)

age_init_vec = zeros(Nidx)
nonlinearprob = NonlinearProblem(f!, age_init_vec, [])

@time sol = solve(
    nonlinearprob,
    newton_solver;
    show_trace = Val(true),
    reltol = Inf,
    abstol = 0.001 * stop_time,
    verbose = true,
)

@info "Newton-GMRES solve complete" retcode = sol.retcode total_G_calls = g_call_count[]
flush(stdout)

################################################################################
# Save result
################################################################################

@info "Saving steady-state age"
flush(stdout)

age_steady_3D = zeros(Float64, Nx′, Ny′, Nz′)
age_steady_3D[idx] .= sol.u

vol_mean = sum(sol.u .* v1D) / sum(v1D) / year
@info "Volume-weighted mean periodic steady age: $vol_mean years"

steady_dir = joinpath(outputdir, "age", run_mode_tag)
mkpath(steady_dir)
steady_file = joinpath(steady_dir, "steady_age_newton.jld2")
jldsave(steady_file; age = age_steady_3D, wet3D, idx)
@info "Saved steady-state age to $steady_file"
flush(stdout)

@info "solve_periodic_newton.jl complete"
flush(stdout)
