"""
To run this on Gadi interactively on the GPU queue, use

```
qsub -I -P y99 -N RungeKutta3bug -l mem=47GB -q normal -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o scratch_output/PBS/ -j oe
qsub -I -P y99 -N RungeKutta3bug -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o scratch_output/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/RungeKutta3bug.jl")
```
"""

#########################################
@info "0. Loading packages and functions"
#########################################

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

using Statistics
using Printf
using Oceananigans.Grids: on_architecture

Nx = 360
Ny = 120
latitude = (-60, 60)
longitude = (-180, 180)

# A spherical domain
grid = LatitudeLongitudeGrid(
    arch;
    size = (Nx, Ny, 1),
    radius = 1,
    latitude = latitude,
    longitude = longitude,
    z = (-1, 0)
)

model = HydrostaticFreeSurfaceModel(
    grid;
    timestepper = :SplitRungeKutta3, # <- this causes the call to `model` to error
    tracers = :c,
    velocities = PrescribedVelocityFields(), # quiescent
    closure = HorizontalScalarDiffusivity(κ = 1)
)

# Tracer patch for visualization
Gaussian(λ, φ, L) = exp(-(λ^2 + φ^2) / 2L^2)

# Tracer patch parameters
L = 12 # degree
φ₀ = 0 # degrees

cᵢ(λ, φ, z) = Gaussian(λ, φ - φ₀, L)

set!(model, c = cᵢ)

c = model.tracers.c

φᵃᶜᵃ_max = maximum(abs, φnodes(grid, Center(), Center(), Center()))
Δx_min = grid.radius * cosd(φᵃᶜᵃ_max) * deg2rad(grid.Δλᶜᵃᵃ)
Δy_min = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
Δ_min = min(Δx_min, Δy_min)

# Time-scale for gravity wave propagation across the smallest grid cell
cell_diffusion_time_scale = Δ_min^2

simulation = Simulation(
    model;
    Δt = 0.1cell_diffusion_time_scale,
    stop_time = 1000cell_diffusion_time_scale,
)

function progress_message(sim)
    max_c = maximum(on_architecture(CPU(), sim.model.tracers.c))
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(c) = %.1e, wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_c, walltime)
end

add_callback!(simulation, progress_message, IterationInterval(100))

# output_fields = model.tracers

# output_prefix = "spot_tracer_diffusion_Nx$(grid.Nx)_Ny$(grid.Ny)"

# simulation.output_writers[:fields] = JLD2Writer(
#     model, output_fields,
#     schedule = TimeInterval(10cell_diffusion_time_scale),
#     filename = output_prefix,
#     overwrite_existing = true
# )

run!(simulation)

