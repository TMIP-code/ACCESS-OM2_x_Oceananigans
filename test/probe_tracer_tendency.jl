"""
Dump tracer-tendency `Gⁿ.age` + every input the tendency kernel reads, at
iter 0 (pre-step) and iter 1 (post first Euler step). Compare CPU 1×2 vs
GPU 1×2 dumps with `scripts/debugging/compare_tendency_probes.jl` to
discriminate:

  - identical inputs at iter 0 + different `Gⁿ.age` at iter 1
        → GPU-specific tendency-kernel bug
  - inputs already differ at iter 0
        → upstream bug (FTS loader, partition data, grid metrics, halos)

`Gⁿ.age` after `time_step!` still holds the tendency that was just
computed and applied — `cache_previous_tendencies!` copies it to `G⁻`
without zeroing `Gⁿ`, and nothing else writes to `Gⁿ.age` between the AB2
step and the next `compute_tracer_tendencies!`. So a callback at the end
of each iteration captures the same `Gⁿ.age` we would have captured with
a manual `time_step!` decomposition.

Outputs land in `{outputdir}/standardrun/{MC}/{px x py}/probe/`:
    probe_tendency_{cpu|gpu}_iter{N}{_rank{R}}{_noACM}.jld2

Set `PROBE_NSTEPS=10` to extend the dump to iter 0..10 (and watch the
bell-shape build up on GPU+1×2).

Usage (via PBS):
  PARTITION=1x2 JOB_CHAIN=probetend    bash scripts/test_driver.sh   # GPU
  PARTITION=1x2 JOB_CHAIN=probetendcpu bash scripts/test_driver.sh   # CPU
"""

include("../src/setup_model.jl")
include("../src/setup_simulation.jl")

using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Oceananigans.Architectures: architecture, on_architecture, child_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

if arch isa Distributed
    using MPI
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
else
    rank = 0
end

device_str = if arch isa Distributed
    child_architecture(arch) isa CPU ? "cpu" : "gpu"
else
    arch isa CPU ? "cpu" : "gpu"
end

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
probe_root = isempty(gpu_tag) ?
    joinpath(outputdir, "standardrun", model_config, "probe") :
    joinpath(outputdir, "standardrun", model_config, gpu_tag, "probe")
mkpath(probe_root)

rank_suffix = arch isa Distributed ? "_rank$(rank)" : ""

@info "PROBE_TEND: rank=$rank device=$device_str probe_root=$probe_root"

host(x) = Array(on_architecture(CPU(), x))

# u/v are TimeSeriesInterpolation (wrapping FTS); w is a materialised Field
# from DiagnosticVerticalVelocity. The FTS buffer holds all loaded snapshots
# (including halo cells), so we dump the whole thing — only ~Hy halo rows per
# snapshot are actually consumed at iter 0, but we get the iter≥1 inputs for free.
dump_velocity(v) = v isa TimeSeriesInterpolation ?
    host(parent(v.time_series.data)) : host(parent(v))

ug = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid

function dump_state(model, n::Integer)
    out_path = joinpath(
        probe_root,
        "probe_tendency_$(device_str)_iter$(n)$(rank_suffix)$(noACM_suffix()).jld2",
    )
    @info "PROBE_TEND: rank=$rank dumping iter=$n to $out_path"

    jldopen(out_path, "w") do f
        f["meta/iteration"] = model.clock.iteration
        f["meta/clock_time"] = model.clock.time
        f["meta/Δt"] = Δt
        f["meta/device"] = device_str
        f["meta/rank"] = rank
        f["meta/partition_x"] = px
        f["meta/partition_y"] = py
        f["meta/chi"] = model.timestepper.χ
        f["meta/active_cells_map"] = active_cells_map_enabled()

        # Tracer (parent: includes halos)
        f["age"] = host(parent(model.tracers.age))

        # Velocities
        f["u_fts"] = dump_velocity(model.velocities.u)
        f["v_fts"] = dump_velocity(model.velocities.v)
        f["w"] = host(parent(model.velocities.w))

        # Free-surface displacement (FTS-backed for our setup)
        eta_disp = model.free_surface.displacement
        if eta_disp isa TimeSeriesInterpolation
            f["eta_fts"] = host(parent(eta_disp.time_series.data))
            f["eta_clock_time"] = eta_disp.clock.time
        else
            f["eta_fts"] = host(parent(eta_disp))
        end

        # AB2 tendency state
        f["Gn_age"] = host(parent(model.timestepper.Gⁿ.age))
        f["Gm_age"] = host(parent(model.timestepper.G⁻.age))

        # z-star vertical-coordinate internals (MutableVerticalDiscretization)
        if hasproperty(ug.z, :σᶜᶜⁿ)
            f["sigma_cc"] = host(parent(ug.z.σᶜᶜⁿ))
        end
        if hasproperty(ug.z, :ηⁿ)
            f["eta_n"] = host(parent(ug.z.ηⁿ))
        end
        if hasproperty(ug.z, :∂t_σ)
            f["dt_sigma"] = host(parent(ug.z.∂t_σ))
        end

        # Grid metrics (constructed at partition time — the docs flag these
        # as candidates for a partition-construction off-by-one at the rank's
        # south boundary).
        for (name, prop) in [
                ("Dx_cca", :Δxᶜᶜᵃ), ("Dy_cca", :Δyᶜᶜᵃ), ("Az_cca", :Azᶜᶜᵃ),
                ("Dx_fca", :Δxᶠᶜᵃ), ("Dy_fca", :Δyᶠᶜᵃ), ("Az_fca", :Azᶠᶜᵃ),
                ("Dx_cfa", :Δxᶜᶠᵃ), ("Dy_cfa", :Δyᶜᶠᵃ), ("Az_cfa", :Azᶜᶠᵃ),
            ]
            if hasproperty(ug, prop)
                metric = getproperty(ug, prop)
                metric isa AbstractArray && (f["metric/$name"] = host(parent(metric)))
            end
        end

        # Immersed-boundary bottom topography
        if model.grid isa ImmersedBoundaryGrid
            ib = model.grid.immersed_boundary
            if hasproperty(ib, :bottom_height)
                f["bottom_height"] = host(parent(ib.bottom_height.data))
            end
        end
    end
    flush(stdout); flush(stderr)
    return out_path
end

nsteps = parse(Int, get(ENV, "PROBE_NSTEPS", "1"))
@info "PROBE_TEND: rank=$rank PROBE_NSTEPS=$nsteps (dumping iter 0..$nsteps)"

# Initial state (pre any time_step!). update_state! has already run inside
# the model constructor, so velocities, w-from-continuity, and tracer halos
# are filled.
dump_state(model, 0)

for n in 1:nsteps
    @info "PROBE_TEND: rank=$rank time_step!($n)"
    time_step!(model, Δt)
    dump_state(model, n)
end

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
@info "PROBE_TEND: rank=$rank all probes complete."
flush(stdout); flush(stderr)
