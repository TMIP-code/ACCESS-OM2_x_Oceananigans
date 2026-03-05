# MWE: NonlinearSolve allocates dense N×N Jacobian for Anderson acceleration
# via NLsolveJL and SIAMFANLEquationsJL wrappers, even though Anderson
# acceleration does not use a Jacobian.
#
# Bug locations in NonlinearSolve (v4.x):
#
# 1. NonlinearSolveNLsolveExt.jl (line ~27):
#    Always creates OnceDifferentiable(f!, u0, resid; autodiff) which allocates
#    a dense N×N Jacobian. NLsolve.jl's own nlsolve() correctly uses
#    NonDifferentiable (no Jacobian) for method=:anderson.
#
# 2. NonlinearSolveSIAMFANLEquationsExt.jl (line ~106):
#    Allocates FPS = zeros(N, N) unconditionally before branching on method.
#    For Anderson, aasol() only needs Vstore of size N×(3m+3), never uses FPS.
#
# Impact: For large N (e.g., N = 1.5M in our ocean model), the dense N×N
# allocation attempts ~18 TB → immediate OOM kill.
#
# Workaround: Call NLsolve.nlsolve() and SIAMFANLEquations.aasol() directly
# instead of going through NonlinearSolve wrappers.
#
# To reproduce: run this script. Increase N to see memory blow up.
#   julia --project bugs/NonlinearSolve_dense_jacobian_MWE.jl

using NonlinearSolve
using NLsolve
using SIAMFANLEquations

# --- Problem setup ---
N = 50_000 # increase to see memory blow up (N=50k → ~20 GB dense Jacobian)
G!(F, x, p) = (F .= sin.(x) .- x) # simple residual
x0 = ones(N)

# --- Bug 1: NLsolveJL via NonlinearSolve allocates dense N×N Jacobian ---
println("=== NLsolveJL via NonlinearSolve (allocates dense $N×$N Jacobian) ===")
prob = NonlinearProblem(NonlinearFunction(G!), x0, nothing)
try
    @time sol = solve(prob, NLsolveJL(; method = :anderson, m = 5); maxiters = 3)
    println("NLsolveJL via NonlinearSolve: OK (but wasted ~$(round(N^2 * 8 / 1.0e9, digits = 1)) GB on unused Jacobian)")
catch e
    println("NLsolveJL via NonlinearSolve: FAILED — $e")
end

# --- Bug 2: SIAMFANLEquationsJL via NonlinearSolve allocates dense N×N FPS ---
println("\n=== SIAMFANLEquationsJL via NonlinearSolve (allocates dense $N×$N FPS) ===")
try
    @time sol = solve(prob, SIAMFANLEquationsJL(; method = :anderson); maxiters = 3)
    println("SIAMFANLEquationsJL via NonlinearSolve: OK (but wasted ~$(round(N^2 * 8 / 1.0e9, digits = 1)) GB on unused FPS)")
catch e
    println("SIAMFANLEquationsJL via NonlinearSolve: FAILED — $e")
end

# --- Workaround 1: NLsolve directly (no Jacobian) ---
println("\n=== NLsolve.nlsolve directly (no Jacobian allocated) ===")
f!(F, x) = G!(F, x, nothing)
@time result = NLsolve.nlsolve(f!, x0; method = :anderson, m = 5, iterations = 3)
println("NLsolve direct: converged=$(NLsolve.converged(result))")

# --- Workaround 2: SIAMFANLEquations.aasol directly (no FPS allocated) ---
println("\n=== SIAMFANLEquations.aasol directly (no FPS allocated) ===")
FP!(G, x) = (G .= x .+ sin.(x) .- x) # fixed-point map: G(x) = x + f(x) for this toy problem
m = 5
Vstore = zeros(Float64, N, 3 * m + 3)
@time result2 = SIAMFANLEquations.aasol(FP!, x0, m, Vstore; maxit = 3)
println("SIAMFANLEquations direct: converged=$(result2.idid)")

# --- For comparison: SpeedMappingJL works fine (no Jacobian) ---
println("\n=== SpeedMappingJL via NonlinearSolve (no Jacobian — correct) ===")
using SpeedMapping
@time sol3 = solve(prob, SpeedMappingJL(); maxiters = 3)
println("SpeedMappingJL: retcode=$(sol3.retcode)")
