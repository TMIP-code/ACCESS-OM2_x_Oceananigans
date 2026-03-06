"""
Plot age diagnostic figures from solver trace history.

This is a standalone CPU script that loads the saved age fields from each
solver iteration (produced when TRACE_SOLVER_HISTORY=yes) and generates
zonal-average and horizontal-slice plots for each iterate, plus convergence
summary plots.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=02:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_trace_history.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  TRACE_PHASE      – start | end | both  (default: end)
                     Which phase of each iteration to plot diagnostics for.
"""

@info "Loading packages for trace history plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days

using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using Statistics
using TOML
using JLD2
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
else
    outputdir = profile["outputdir"]
end

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

NONLINEAR_SOLVER = get(ENV, "NONLINEAR_SOLVER", "anderson")
solver_subdir = NONLINEAR_SOLVER == "newton" ? "NK" : "AA"
trace_dir = joinpath(outputdir, "periodic", model_config, solver_subdir)
trace_plots_dir = joinpath(trace_dir, "plots")
mkpath(trace_plots_dir)

TRACE_JOB_ID = get(ENV, "TRACE_JOB_ID", "")

TRACE_PHASE = get(ENV, "TRACE_PHASE", "end")
(TRACE_PHASE ∈ ("start", "end", "both")) || error("TRACE_PHASE must be start, end, or both (got: $TRACE_PHASE)")

@info "Trace history plot configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- NONLINEAR_SOLVER = $NONLINEAR_SOLVER"
@info "- trace_dir        = $trace_dir"
@info "- TRACE_JOB_ID     = $(isempty(TRACE_JOB_ID) ? "(all jobs)" : TRACE_JOB_ID)"
@info "- TRACE_PHASE      = $TRACE_PHASE"
flush(stdout); flush(stderr)

isdir(trace_dir) || error("Trace directory not found: $trace_dir")

################################################################################
# Load grid
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

################################################################################
# Compute masks and volume
################################################################################

@info "Computing wet mask and cell volumes"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)

vol = compute_volume(grid)
vol_3D = Array(interior(vol))

################################################################################
# Discover and sort trace files
################################################################################

@info "Scanning trace directory for iteration files"
flush(stdout); flush(stderr)

all_files = readdir(trace_dir)
trace_files = filter(f -> startswith(f, "age_trace_iter_") && endswith(f, ".jld2"), all_files)
if !isempty(TRACE_JOB_ID)
    trace_files = filter(f -> occursin(TRACE_JOB_ID, f), trace_files)
end
sort!(trace_files)

@info "Found $(length(trace_files)) trace files"
flush(stdout); flush(stderr)

if isempty(trace_files)
    @warn "No trace files found in $trace_dir — nothing to plot"
    @info "plot_trace_history.jl complete (no files)"
    flush(stdout); flush(stderr)
    exit(0)
end

################################################################################
# Generate diagnostic plots for each trace file
################################################################################

phases_to_plot = TRACE_PHASE == "both" ? ["start", "end"] : [TRACE_PHASE]

for trace_file in trace_files
    filepath = joinpath(trace_dir, trace_file)

    # Load as FieldTimeSeries — contains 2 timesteps: t=0 (start) and t=stop_time (end)
    age_fts = FieldTimeSeries(filepath, "age")
    n_times = length(age_fts.times)

    # Extract iteration number from filename (e.g., age_trace_iter_0001.jld2 → 1)
    m = match(r"age_trace_iter_(\d+).*\.jld2", trace_file)
    iter_num = m === nothing ? 0 : parse(Int, m.captures[1])

    for phase in phases_to_plot
        if phase == "start" && n_times >= 1
            age_data = interior(age_fts[1])
        elseif phase == "end" && n_times >= 2
            age_data = interior(age_fts[n_times])
        elseif phase == "end" && n_times == 1
            age_data = interior(age_fts[1])
        else
            continue
        end

        age_years_3D = age_data ./ year

        # Summary statistics (wet cells only)
        age_wet_years = age_years_3D[idx]
        max_age_years = maximum(abs, age_wet_years)
        mean_age_years = mean(age_wet_years)
        @info "Plotting iter=$iter_num phase=$phase" max_age = round(max_age_years; digits = 2) mean_age = round(mean_age_years; digits = 2)
        flush(stdout); flush(stderr)

        # Adaptive colorrange
        cmax = max(ceil(max_age_years), 1.0)
        crange = (0, cmax)
        nlevels = min(20, max(10, Int(ceil(cmax))))
        clevels = range(0, cmax, length = nlevels + 1)

        label = @sprintf("trace_iter_%04d_%s", iter_num, phase)

        plot_age_diagnostics(
            age_years_3D, grid, wet3D, vol_3D, trace_plots_dir, label;
            colorrange = crange, levels = clevels,
        )
    end
end

################################################################################
# Summary convergence plot: max/mean age vs iteration
################################################################################

@info "Generating convergence summary plots"
flush(stdout); flush(stderr)

iters = Int[]
max_ages = Float64[]
mean_ages = Float64[]

for trace_file in trace_files
    filepath = joinpath(trace_dir, trace_file)
    m = match(r"age_trace_iter_(\d+).*\.jld2", trace_file)
    m === nothing && continue
    iter_num = parse(Int, m.captures[1])

    age_fts = FieldTimeSeries(filepath, "age")
    n_times = length(age_fts.times)
    age_data = interior(age_fts[n_times])  # end of simulation

    age_wet_years = age_data[idx] ./ year
    push!(iters, iter_num)
    push!(max_ages, maximum(abs, age_wet_years))
    push!(mean_ages, mean(age_wet_years))
end

if length(iters) >= 2
    fig = Figure(; size = (800, 500))
    ax = Axis(
        fig[1, 1];
        title = "Solver convergence — age statistics per iteration",
        xlabel = "Iteration (Φ! call number)",
        ylabel = "Age (years)",
    )
    lines!(ax, iters, max_ages; label = "max |age|", color = :red)
    lines!(ax, iters, mean_ages; label = "mean age", color = :blue)
    axislegend(ax)

    conv_file = joinpath(trace_plots_dir, "convergence_summary.png")
    @info "Saving $conv_file"
    save(conv_file, fig)
end

################################################################################
# Drift plot: |Φ(x) - x| per iteration (end - start within same file)
################################################################################

iters_drift = Int[]
max_drifts = Float64[]
mean_drifts = Float64[]

for trace_file in trace_files
    filepath = joinpath(trace_dir, trace_file)
    m = match(r"age_trace_iter_(\d+).*\.jld2", trace_file)
    m === nothing && continue

    age_fts = FieldTimeSeries(filepath, "age")
    n_times = length(age_fts.times)
    n_times < 2 && continue

    iter_num = parse(Int, m.captures[1])
    age_start = interior(age_fts[1])
    age_end = interior(age_fts[n_times])

    drift = (age_end .- age_start)[idx]
    push!(iters_drift, iter_num)
    push!(max_drifts, maximum(abs, drift) / year)
    push!(mean_drifts, mean(abs, drift) / year)
end

if length(iters_drift) >= 2
    fig = Figure(; size = (800, 500))
    ax = Axis(
        fig[1, 1];
        title = "Solver convergence — drift |Φ(x) - x| per iteration",
        xlabel = "Iteration (Φ! call number)",
        ylabel = "Drift (years)",
        yscale = log10,
    )
    lines!(ax, iters_drift, max_drifts; label = "max |drift|", color = :red)
    lines!(ax, iters_drift, mean_drifts; label = "mean |drift|", color = :blue)
    axislegend(ax)

    drift_file = joinpath(trace_plots_dir, "drift_convergence.png")
    @info "Saving $drift_file"
    save(drift_file, fig)
end

@info "plot_trace_history.jl complete"
flush(stdout); flush(stderr)
