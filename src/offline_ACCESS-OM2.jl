"""
To run this on Gadi interactively on the GPU queue, use

```
qsub -I -P y99 -l mem=47GB -q normal -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+gdata/ik11+scratch/y99 -o scratch_output/PBS/ -j oe
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99 -o scratch_output/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/offline_ACCESS-OM2.jl")
```
"""

# TODO: Check the minimum Δz
#   TODO: check that Δz from kmt is ≥ 0.2Δz from z levels (from ocean vgrid)

# TODO: Check

@info "Loading packages and functions"

using Oceananigans

# Comment/uncomment the following lines to enable/disable GPU
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = GPU()
else
    arch = CPU()
end
@info "Using $arch architecture"

using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode, get_active_cells_map
using Oceananigans.Simulations: reset!
using Oceananigans.OutputReaders: Cyclical, InMemory
using Oceananigans.Advection: div_Uc
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using Statistics
using LinearAlgebra
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using TOML
using JLD2
using Printf
using CairoMakie
using NonlinearSolve
using SpeedMapping
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using DifferentiationInterface: overloaded_input_type
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings
using OceanTransportMatrixBuilder
import Pardiso # import Pardiso instead of using (to avoid name clash?)
const nprocs = 12

# Determine which model profile to use. Priority:
# 1. ARGS[1] passed to julia
# 2. ENV["PARENTMODEL"]
# 3. `defaults.parentmodel` in LocalPreferences.toml
# 4. fallback to ACCESS-OM2-1
cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENTMODEL")
    ENV["PARENTMODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
    outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
    Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds
else
    outputdir = profile["outputdir"]
    Δt = profile["dt_seconds"] * second
end

mkpath(outputdir)
save_grid = false

# TODO: Maybe I should only use the supergrid for the locations
# of center/face/corner points but otherwise use the "standard"
# grid metrics available from the MOM outputs?
# -> No, instead I should use the same inputs! I think that is what I am doing now,
# but needs to be checked. I think best would be to read the input file location
# from the config file.

include("tripolargrid_reader.jl")

################################################################################
################################################################################
################################################################################

################################################################################
# Load grid from JLD2
################################################################################

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")
grid = load_tripolar_grid(grid_file)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

# # Plot bottom height
# h = on_architecture(CPU(), grid.immersed_boundary.bottom_height)
# fig = Figure()
# ax = Axis(fig[2, 1], aspect = 2.0)
# hm = surface!(
#     ax,
#     1:Nx, #view(grid.underlying_grid.λᶜᶜᵃ, 1:Nx, 1:Ny),
#     1:Ny, #view(grid.underlying_grid.φᶜᶜᵃ, 1:Nx, 1:Ny),
#     view(h.data, 1:Nx, 1:Ny, 1);
#     colormap = :viridis
# )
# Colorbar(fig[1, 1], hm, vertical = false, label = "Bottom height (m)")
# save(joinpath(outputdir, "bottom_height_heatmap_$(typeof(arch)).png"), fig)

################################################################################
################################################################################
################################################################################

@info "Loading velocities from disk"

u_file = joinpath(outputdir, "$(parentmodel)_u_ts.jld2")
v_file = joinpath(outputdir, "$(parentmodel)_v_ts.jld2")
η_file = joinpath(outputdir, "$(parentmodel)_eta_ts.jld2")

# Load FieldTimeSeries from disk using InMemory backend
# N_in_mem specifies how many timesteps to keep in memory at a time
# Not sure how many I should use.
N_in_mem = 4  # Keep 4 timesteps in memory (monthly data)

u_ts = FieldTimeSeries(u_file, "u"; backend = InMemory(N_in_mem), time_indexing = Cyclical(1year))
v_ts = FieldTimeSeries(v_file, "v"; backend = InMemory(N_in_mem), time_indexing = Cyclical(1year))
η_ts = FieldTimeSeries(η_file, "η"; backend = InMemory(N_in_mem), time_indexing = Cyclical(1year))

prescribed_Δt = u_ts.times[2] - u_ts.times[1]  # Infer from time spacing
fts_times = u_ts.times

@info "Velocities loaded (InMemory backend with $N_in_mem timesteps in memory)"

velocities = PrescribedVelocityFields(u = u_ts, v = v_ts, w = DiagnosticVerticalVelocity())
free_surface = PrescribedFreeSurface(displacement = η_ts)

################################################################################
################################################################################
################################################################################

# TODO implement PrescribedActiveTracers?

@info "Creating closures"

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

# Vertical diffusivity parameters
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# Load MLD to add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

implicit_vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κVField
)
explicit_vertical_diffusion = VerticalScalarDiffusivity(
    ExplicitTimeDiscretization();
    κ = κVField
)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

# Combine them all into a single diffusion closure
closure = (
    horizontal_diffusion,
    implicit_vertical_diffusion,
)
explicit_closure = (
    horizontal_diffusion,
    explicit_vertical_diffusion,
)

@info "Closures created"

################################################################################
################################################################################
################################################################################

@info "Model"

age_parameters = (;
    relaxation_timescale = 3Δt, # Relaxation timescale for removing age at surface
    source_rate = 1.0,          # Source for the age (1 second / second)
)

@inline age_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)
# TODO Do I really need a linear age source/sink? Maybe just rename ADc to age and reuse the same source/sink?
@inline linear_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)
# @inline age_jvp_source(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, 0.0)
# @inline age_source_sink(i, j, k, grid, clock, fields, params) = params.source_rate

age_dynamics = Forcing(
    age_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)
linear_dynamics = Forcing(
    linear_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)

forcing = (
    age = age_dynamics,
)
linear_forcing = (
    ADc = linear_dynamics,
)

# Tracer for autodiff tracing
ADc0 = CenterField(grid)
age0 = CenterField(grid)


# For building the Jacobian via autodiff I need to create a second model
# with linear forcings and explicit diffusion. So here I use common kwargs
# to make sure they share all the other pieces.
model_common_kwargs = (
    # tracer_advection = WENO(order = 5),
    tracer_advection = Centered(order = 2),
    # tracer_advection = UpwindBiased(),
    # timestepper = :SplitRungeKutta3, # <- to try and improve numerical stability over AB2
    velocities = velocities,
    free_surface = free_surface,
)
model_kwargs = (;
    model_common_kwargs...,
    tracers = (; age = age0),
    closure = closure,
    forcing = forcing,
)
jacobian_model_kwargs = (
    model_common_kwargs...,
    tracers = (; ADc = ADc0),
    closure = explicit_closure,
    forcing = linear_forcing,
)

model = HydrostaticFreeSurfaceModel(grid; model_kwargs...)

# TODO: This does not work with a MutableVerticalDiscretization, without a
# free_surface kwarg I guess. But I would rather a PrescribedFreeSurface() instead.
# I asked on Slack but this might be a tall order.

################################################################################
################################################################################
################################################################################

@info "Initial condition"

# set!(model, age = Returns(0.0), age_jvp = Returns(0.0)) # TODO: Unneccessary as fields are initialized to zero by default.
set!(model, age = Returns(0.0)) # TODO: Unneccessary as fields are initialized to zero by default.
# fill_halo_regions!(model.tracers.age)

fig, ax, plt = heatmap(
    make_plottable_array(model.tracers.age)[:, :, Nz];
    colormap = :viridis,
    axis = (; title = "Initial age at surface (years)"),
)
Colorbar(fig[1, 2], plt)
save(joinpath(outputdir, "initial_age_surface_$(typeof(arch)).png"), fig)

################################################################################
################################################################################
################################################################################

@info "Simulation"

stop_time = 12 * prescribed_Δt

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx = findmax(adapt(Array, sim.model.tracers.age) / year) # in years
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf(
        # "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d) wall time: %s\n",
        # iteration(sim), time(sim), sim.Δt, max_age, idx.I..., walltime
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age)/time = %.1e at (%d, %d, %d), mean(age) = %.1e, wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age / (time(sim) / year), idx.I..., mean_age, walltime
    )
end

# add_callback!(simulation, progress_message, TimeInterval(1year))
add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))
# add_callback!(simulation, zero_age_callback, IterationInterval(1))


output_fields = Dict(
    "age" => model.tracers.age,
    # "u" => model.velocities.u,
)

output_prefix = joinpath(outputdir, "offline_age_$(parentmodel)_$(typeof(arch))")

# simulation.output_writers[:fields] = NetCDFWriter(
simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(prescribed_Δt),
    # schedule = IterationInterval(1),
    filename = output_prefix,
    # dimensions=output_dims,
    # include_grid_metrics = true,
    # verbose = true,
    # array_type = Array{Float32},
    overwrite_existing = true,
)

run!(simulation)

################################################################################
################################################################################
################################################################################

# @info "Plotting"

# plottable_age = make_plottable_array(model.tracers.age)
# for k in 1:50
#     local fig, ax, plt = heatmap(
#         plottable_age[:, :, k] / year;
#         # on_architecture(CPU(), model.tracers.age.data.parent)[:, :, Nz];
#         nan_color = :black,
#         colorrange = (0, stop_time / year),
#         colormap = cgrad(cgrad(:tab20b; categorical = true)[1:16], categorical = true),
#         lowclip = :red,
#         highclip = :yellow,
#         axis = (; title = "Final age (years) at level k = $k"),
#     )
#     Colorbar(fig[1, 2], plt)
#     save(joinpath(outputdir, "final_age_k$(k)_$(parentmodel)_$(typeof(arch)).png"), fig)
# end


# # age_lazy = open_dataset(simulation.output_writers[:fields].filepath)["age"]
# age_lazy = FieldTimeSeries(simulation.output_writers[:fields].filepath, "age")
# # u_lazy = FieldTimeSeries(simulation.output_writers[:fields].filepath, "u")
# output_times = age_lazy.times

# set_theme!(Theme(fontsize = 30))

# # fig = Figure(size = (1200, 1200))
# fig = Figure(size = (1200, 600))

# n = Observable(1)
# k = 25
# agetitle = @lift "age and u on offline OM2 at k = $k, t = " * prettytime(output_times[$n])
# # utitle = @lift "u at k = $k, t = " * prettytime(output_times[$n])

# # agekₙ = @lift readcubedata(age_lazy[At(k = $k, times = [$n])]) # in years
# agekₙ = @lift make_plottable_array(age_lazy[$n])[:, :, k] / year # in years
# # ukₙ = @lift make_plottable_array(u_lazy[$n])[:, :, k] # in m/s

# ax = fig[1, 1] = Axis(
#     fig,
#     xlabel = "longitude index",
#     ylabel = "latitude index",
#     title = agetitle,
# )

# hm = heatmap!(
#     ax, agekₙ;
#     colorrange = (0, stop_time / year),
#     # extendhigh = auto,
#     # extendlow = auto,
#     # colorscale = SymLog(0.01),
#     colormap = :viridis,
#     nan_color = (:black, 1),
# )
# Colorbar(fig[1, 2], hm)

# ax = fig[2, 1] = Axis(
#     fig,
#     xlabel = "longitude index",
#     ylabel = "latitude index",
#     title = utitle,
# )

# hm = heatmap!(
#     ax, ukₙ;
#     colorrange = (-0.1, 0.1),
#     colormap = :RdBu,
#     nan_color = (:black, 1),
# )
# Colorbar(fig[2, 2], hm)

# frames = 1:length(output_times)

# @info "Making an animation..."

# Makie.record(fig, joinpath(outputdir, "offline_age_OM2_test_$(typeof(arch)).mp4"), frames, framerate = 25) do i
#     println("frame $i/$(length(frames))")
#     n[] = i
# end

################################################################################
################################################################################
################################################################################

foo


@info "Build matrix"

jacobian_model = HydrostaticFreeSurfaceModel(grid; jacobian_model_kwargs...)

@warn "Adding newton_div method to allow sparsity tracer to pass through WENO"
autodifftypes = Union{SparseConnectivityTracer.AbstractTracer, SparseConnectivityTracer.Dual, ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b) where {FT <: autodifftypes} = a / b


# TODO: Check if I really need to rewrite these. Not sure why but I think I had to.
# (These are copy-pasta from Oceananigans.)
@kernel function compute_hydrostatic_free_surface_GADc!(GADc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GADc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

Nx′, Ny′, Nz′ = size(ADc0)
N′ = Nx′ * Ny′ * Nz′
fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
wet3D = .!isnan.(interior(fNaN))
idx = findall(wet3D)
Nidx = length(idx)

ADc_advection = jacobian_model.advection[:ADc]
total_velocities = jacobian_model.transport_velocities
kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
active_cells_map = get_active_cells_map(grid, Val(:interior))

function mytendency!(GADcvec::Vector{T}, ADcvec::Vector{T}, clock) where {T}

    # Preallocate 3D array with type T and fill wet points
    # TODO find a way to preallocate 3D arrays outside of function?
    ADc3D = zeros(T, Nx′, Ny′, Nz′)
    ADc3D[idx] .= ADcvec

    # Preallocate Field with type T and fill it with 3D array
    # TODO find a way to preallocate fields outside of function?
    ADc = CenterField(grid, T)
    set!(ADc, ADc3D)

    # Preallocate "output" Field with type T
    GADc = CenterField(grid, T)

    # bits and pieces from model
    c_advection = jacobian_model.advection[:ADc]
    c_forcing = jacobian_model.forcing[:ADc]
    c_immersed_bc = immersed_boundary_condition(jacobian_model.tracers[:ADc])

    args = tuple(
        Val(1),
        Val(:ADc),
        c_advection,
        jacobian_model.closure,
        c_immersed_bc,
        jacobian_model.buoyancy,
        jacobian_model.biogeochemistry,
        jacobian_model.transport_velocities,
        jacobian_model.free_surface,
        (; ADc = ADc),
        jacobian_model.closure_fields,
        jacobian_model.auxiliary_fields,
        clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_GADc!,
        GADc,
        grid,
        args;
        active_cells_map
    )

    # Fill output vector with interior wet values
    GADcvec .= view(interior(GADc), idx)

    return GADcvec
end


@info "benchmark tendency function"
ADcvec = ones(Nidx)
GADcvec = ones(Nidx)
@time mytendency!(GADcvec, ADcvec, 0.0)
@time mytendency!(GADcvec, ADcvec, 0.0)

@info "Autodiff setup"

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

# strict mode false to allow different (preallocated) function
@time "Prepare Jacobian sparsity pattern" jac_prep_sparse = prepare_jacobian(
    mytendency!,
    GADcvec,
    sparse_forward_backend,
    ADcvec,
    Constant(0.0);
    strict = Val(false),
)

DualType = eltype(overloaded_input_type(jac_prep_sparse))
# Preallocate 3D array with type T and fill wet points
ADc3D_dual = zeros(DualType, Nx′, Ny′, Nz′)
# Preallocate Field with type T and fill it with 3D array
ADc_dual = CenterField(grid, DualType)
# Preallocate "output" Field with type T
Gc_dual = CenterField(grid, DualType)

function mytendency_preallocated!(GADcvec::Vector{DualType}, ADcvec::Vector{DualType}, clock)

    ADc3D_dual[idx] .= ADcvec
    set!(ADc_dual, ADc3D_dual)

    # bits and pieces from model
    c_advection = jacobian_model.advection[:ADc]
    c_forcing = jacobian_model.forcing[:ADc]
    c_immersed_bc = immersed_boundary_condition(jacobian_model.tracers[:ADc])

    args = tuple(
        Val(1),
        Val(:ADc),
        c_advection,
        jacobian_model.closure,
        c_immersed_bc,
        jacobian_model.buoyancy,
        jacobian_model.biogeochemistry,
        jacobian_model.transport_velocities,
        jacobian_model.free_surface,
        (; ADc = ADc_dual),
        jacobian_model.closure_fields,
        jacobian_model.auxiliary_fields,
        clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_GADc!,
        Gc_dual,
        grid,
        args;
        active_cells_map
    )

    # Fill output vector with interior wet values
    GADcvec .= view(interior(Gc_dual), idx)

    return GADcvec
end

@time "Prepare buffer for Jacobian" jac_buffer = similar(sparsity_pattern(jac_prep_sparse), eltype(ADcvec))

@info "Compute the Jacobian"

i = 1
@info "month = $i / 12"
@time "Compute Jacobian" jacobian!(
    mytendency_preallocated!,
    GADcvec,
    jac_buffer,
    jac_prep_sparse,
    sparse_forward_backend,
    ADcvec,
    Constant(fts_times[i])
)

M = jac_buffer / 12 # <- contains the first-month part of the Jacobian

for i in 2:12
    @info "month = $i / 12"
    jacobian!(
        mytendency_preallocated!,
        GADcvec,
        jac_buffer,
        jac_prep_sparse,
        sparse_forward_backend,
        ADcvec,
        Constant(fts_times[i])
    )
    M .+= jac_buffer / 12
end

# Show me the Jacobian!
display(M)
fig = Figure()
ax = Axis(fig[1, 1])
plt = spy!(
    0.5 .. size(M, 1) + 0.5,
    0.5 .. size(M, 2) + 0.5,
    M;
    colormap = :coolwarm,
    colorrange = maximum(abs.(M)) .* (-1, 1),
    markersize = size(M, 2) / 1000, # adjust marker size based on matrix size
)
ylims!(ax, size(M, 2) + 0.5, 0.5)
Colorbar(fig[1, 2], plt)
save(joinpath(outputdir, "$(parentmodel)_jacobian2.png"), fig)
# This above throws now? Not sure why, skipping for now (in a rush)

################################################################################
################################################################################
################################################################################

@info "Periodic-state solver"

@info "Extra for vector of volumes"

@kernel function compute_volume!(vol, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds vol[i, j, k] = volume(i, j, k, grid, Center(), Center(), Center())
end

function compute_volume(grid)
    vol = CenterField(grid)
    (Nx, Ny, Nz) = size(vol)
    kernel_parameters = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    launch!(CPU(), grid, kernel_parameters, compute_volume!, vol, grid)
    return vol
end

v1D = interior(compute_volume(grid))[idx]

age3D = zeros(Nx′, Ny′, Nz′)

function G!(dage, age, p) # SciML syntax
    @show "calling G!"
    @show extrema(age)
    model = simulation.model
    reset!(simulation)
    simulation.stop_time = stop_time
    age3D[idx] .= age
    age3D_arch = on_architecture(arch, age3D)
    set!(model, age = age3D_arch)
    # model.tracers.age.data .= age3D_arch
    run!(simulation)
    # dage .= model.tracers.age.data.parent
    # dage .= adapt(Array, interior(model.tracers.age))[wet3D]
    dage .= view(interior(model.tracers.age), idx)
    dage .-= age
    @show extrema(dage)
    return dage
end

# @time sol = solve(nonlinearprob, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs)), verbose = true, reltol=1e-10, abstol=Inf);
# @time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-12)); show_trace = Val(true), reltol = Inf, abstol = 1.0e-10norm(u0, Inf));
# @time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(rtol = 1.0e-12), jvp_autodiff = AutoFiniteDiff()); show_trace = Val(true), reltol = Inf, abstol = 1.0);
# @time sol! = solve(nonlinearprob!, SpeedMappingJL(); show_trace = Val(true), reltol = Inf, abstol = 0.001 * 12 * prescribed_Δt / year, verbose = true);

@info "LUMP and SPRAY matrices"
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
display(LUMP)
display(SPRAY)

@info "coarsened Jacobian"
Mc = LUMP * M * SPRAY
display(Mc)

@info "Setting up Pardiso solver"
matrix_type = Pardiso.REAL_SYM
@show solver = MKLPardisoIterate(; nprocs, matrix_type)

@info "Set up preconditioner problem"
# Left Preconditioner needs a new type
struct MyPreconditioner
    prob
end
# P = S Qc⁻¹ L - I
# Q = stop_time * M
# Qc = L Q S = stop_time * Mc
Qc = stop_time * Mc
Plprob = LinearProblem(Qc, ones(size(Qc, 1)))  # following Bardin et al. (2014)
Plprob = init(Plprob, solver, rtol = 1.0e-12)
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

# SciML syntax for left and right preconditioners
Pr = I
precs = Returns((Pl, Pr))

# For initial guess, use the coarsened solution
# Steady state solution is
# ∂age/∂t = 0 = F(age) = M age + source_rate
# since M is the Jacobian of the tendency function age -> F(age) = ∂age/∂t
# so age = M \ -1
# But here age is in units of years
# and ∂age/∂t is in units of year/s
# so M is in units of 1/s and u = M \ -1s/s is in units of s * s / s = s
init_prob_coarsened = LinearProblem(Mc, -ones(size(Mc, 1))) # initial guess for preconditioner solve (can be tuned)
init_prob_coarsened = init(init_prob_coarsened, solver, rtol = 1.0e-12) # initial guess for preconditioner solve (can be tuned)
@time "solve initial age" age_init_vec = SPRAY * solve!(init_prob_coarsened).u / year

fig, ax, plt = hist(age_init_vec)
save(joinpath(outputdir, "initial_steady_age_coarsened_histogram_$(parentmodel).png"), fig)

init_prob_full = LinearProblem(M, -ones(size(M, 1))) # initial guess for preconditioner solve (can be tuned)
init_prob_full = init(init_prob_full, solver, rtol = 1.0e-12) # initial guess for preconditioner solve (can be tuned)
@time "solve initial age full" age_init_vec = solve!(init_prob_full).u / year

fig, ax, plt = hist(age_init_vec)
save(joinpath(outputdir, "initial_steady_age_full_histogram_$(parentmodel).png"), fig)

foo

f! = NonlinearFunction(G!)
nonlinearprob! = NonlinearProblem(f!, age_init_vec, [])

@info "Solving nonlinear problem with GMRES and lump-and-spray preconditioner (Bardin et al., 2014)"
@time sol! = solve(nonlinearprob!,
    NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-10),
        jvp_autodiff = AutoFiniteDiff(relstep = 0.01)
    );
    show_trace = Val(true),
    reltol = Inf,
    abstol = 0.001 * stop_time,
    verbose = true
);
