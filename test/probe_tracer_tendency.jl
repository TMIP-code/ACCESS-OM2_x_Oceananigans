"""
Dump tracer-tendency `Gⁿ.age` + every input the tendency kernel reads, plus
the intra-step age field at three points: before `ab2_step_tracers!`, after
the explicit Euler kernel `_ab2_step_tracer_field!`, and after
`implicit_step!` (the implicit vertical-diffusion solve).

Why these three: the previous round of comparison showed CPU 1×2 and GPU
1×2 produce **bit-identical `Gⁿ.age`** but **different `age` after the full
step** — so the bug is between Gⁿ-write and the final age. Splitting
`ab2_step_tracers!` into its two halves isolates whether the divergence
comes from the explicit AB2 update kernel or from `implicit_step!`.

Dumps land in `{outputdir}/standardrun/{MC}/{px x py}/probe/`:
    probe_tendency_{cpu|gpu}_iter{N}{_rank{R}}{_noACM}.jld2          # full state
    probe_age_{cpu|gpu}_iter{N}_post_explicit{_rank{R}}{_noACM}.jld2 # after _ab2_step_tracer_field!
    probe_age_{cpu|gpu}_iter{N}_post_implicit{_rank{R}}{_noACM}.jld2 # after implicit_step!

The full-state dumps now also include the vertical-diffusion `κ` field
(`closure[2].κ`) and any non-nothing entries of `model.closure_fields`.

`PROBE_NSTEPS` controls how many manual-decomposed steps are run
(default 1 — covers iter 0 → iter 1, the step in which the docs say the
GPU seam bug first fires).

Usage (via PBS):
  PARTITION=1x2 JOB_CHAIN=probetend    bash scripts/test_driver.sh   # GPU
  PARTITION=1x2 JOB_CHAIN=probetendcpu bash scripts/test_driver.sh   # CPU
"""

include("../src/setup_model.jl")
include("../src/setup_simulation.jl")

using Oceananigans.TimeSteppers:
    update_state!,
    cache_previous_tendencies!,
    tick!,
    step_closure_prognostics!
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    compute_tracer_tendencies!,
    compute_momentum_flux_bcs!,
    compute_free_surface_tendency!,
    step_free_surface!,
    compute_transport_velocities!,
    ab2_step_velocities!,
    ab2_step_grid!,
    correct_barotropic_mode!,
    mask_immersed_horizontal_velocities!,
    _ab2_step_tracer_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Oceananigans.Architectures: architecture, on_architecture, child_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Fields: AbstractField
using Oceananigans.Utils: launch!

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

dump_velocity(v) = v isa TimeSeriesInterpolation ?
    host(parent(v.time_series.data)) : host(parent(v))

ug = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid

# Diffusivity field used by implicit vertical diffusion. closure ==
# (horizontal_diffusion, implicit_vertical_diffusion) in this setup, so the
# vertical one is closure[2] and its κ is the (per-tracer) field.
function dump_kappa_v!(f)
    closure = model.closure
    vd = closure isa Tuple ? closure[2] : closure
    if hasproperty(vd, :κ)
        κ = vd.κ
        if κ isa AbstractField
            f["κV"] = host(parent(κ))
        elseif κ isa NamedTuple
            for (name, fld) in pairs(κ)
                fld isa AbstractField && (f["κV_$(name)"] = host(parent(fld)))
            end
        elseif κ isa Number
            f["κV_scalar"] = Float64(κ)
        end
    end
    return nothing
end

function dump_closure_fields!(f)
    cf = model.closure_fields
    cf === nothing && return nothing
    iter_pairs = cf isa Tuple ? enumerate(cf) :
        cf isa NamedTuple ? pairs(cf) :
        [(1, cf)]
    for (key, entry) in iter_pairs
        entry === nothing && continue
        if entry isa AbstractField
            f["closure_fields/$key"] = host(parent(entry))
        elseif entry isa NamedTuple
            for (sub_key, fld) in pairs(entry)
                fld isa AbstractField && (f["closure_fields/$key/$sub_key"] = host(parent(fld)))
            end
        end
    end
    return nothing
end

function dump_full_state(model, n::Integer)
    out_path = joinpath(
        probe_root,
        "probe_tendency_$(device_str)_iter$(n)$(rank_suffix)$(noACM_suffix()).jld2",
    )
    @info "PROBE_TEND: rank=$rank full dump iter=$n → $out_path"

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

        f["age"] = host(parent(model.tracers.age))

        f["u_fts"] = dump_velocity(model.velocities.u)
        f["v_fts"] = dump_velocity(model.velocities.v)
        f["w"] = host(parent(model.velocities.w))

        eta_disp = model.free_surface.displacement
        if eta_disp isa TimeSeriesInterpolation
            f["eta_fts"] = host(parent(eta_disp.time_series.data))
            f["eta_clock_time"] = eta_disp.clock.time
        else
            f["eta_fts"] = host(parent(eta_disp))
        end

        f["Gn_age"] = host(parent(model.timestepper.Gⁿ.age))
        f["Gm_age"] = host(parent(model.timestepper.G⁻.age))

        if hasproperty(ug.z, :σᶜᶜⁿ)
            f["sigma_cc"] = host(parent(ug.z.σᶜᶜⁿ))
        end
        if hasproperty(ug.z, :ηⁿ)
            f["eta_n"] = host(parent(ug.z.ηⁿ))
        end
        if hasproperty(ug.z, :∂t_σ)
            f["dt_sigma"] = host(parent(ug.z.∂t_σ))
        end

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

        if model.grid isa ImmersedBoundaryGrid
            ib = model.grid.immersed_boundary
            if hasproperty(ib, :bottom_height)
                f["bottom_height"] = host(parent(ib.bottom_height.data))
            end
        end

        dump_kappa_v!(f)
        dump_closure_fields!(f)
    end
    flush(stdout); flush(stderr)
    return out_path
end

# Lightweight mid-step dump: just age + Gⁿ (small files for quick diffing).
function dump_age_snapshot(model, n::Integer, stage::AbstractString)
    out_path = joinpath(
        probe_root,
        "probe_age_$(device_str)_iter$(n)_$(stage)$(rank_suffix)$(noACM_suffix()).jld2",
    )
    @info "PROBE_TEND: rank=$rank age snapshot iter=$n stage=$stage → $out_path"
    jldopen(out_path, "w") do f
        f["meta/iteration"] = model.clock.iteration
        f["meta/clock_time"] = model.clock.time
        f["meta/stage"] = stage
        f["meta/Δt"] = Δt
        f["age"] = host(parent(model.tracers.age))
        f["Gn_age"] = host(parent(model.timestepper.Gⁿ.age))
    end
    flush(stdout); flush(stderr)
    return out_path
end

"""
Replicate `time_step!` for `HydrostaticFreeSurfaceModel + QAB2 +
PrescribedVelocityFields + PrescribedFreeSurface`, but split
`ab2_step_tracers!` into its explicit and implicit halves and dump the
age field between them.
"""
function step_with_intra_dumps!(model, Δt, iter_n::Integer)
    @info "PROBE_TEND: rank=$rank manual step iter $iter_n → $(iter_n + 1)"

    FT = eltype(model.grid)
    χ_orig = model.timestepper.χ
    euler = (model.clock.iteration == 0) || (Δt != model.clock.last_Δt)
    χ_step = euler ? convert(FT, -0.5) : χ_orig
    model.timestepper.χ = χ_step

    # maybe_prepare_first_time_step!: at iter 0, QAB2 calls update_state!.
    if model.clock.iteration == 0
        update_state!(model, [])
    end

    # Pre-tracer-tendency block of hydrostatic_ab2_step! — most no-ops for
    # PrescribedVelocityFields + PrescribedFreeSurface, but we run them so
    # the model state matches exactly what `compute_tracer_tendencies!`
    # would normally see.
    compute_momentum_flux_bcs!(model)
    compute_free_surface_tendency!(model.grid, model, model.free_surface)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    compute_transport_velocities!(model, model.free_surface)
    ab2_step_velocities!(model.velocities, model, Δt, χ_step)
    mask_immersed_horizontal_velocities!(model.velocities)
    let u = model.velocities.u, v = model.velocities.v
        fill_halo_regions!((u, v), model.clock, fields(model); async = true)
    end

    compute_tracer_tendencies!(model)
    ab2_step_grid!(model.grid, model, model.vertical_coordinate, Δt, χ_step)
    correct_barotropic_mode!(model, Δt)

    # Manual split of ab2_step_tracers!. Only `age` is in our setup.
    tracer_index = 1
    tracer_name = :age
    Gⁿ = model.timestepper.Gⁿ[tracer_name]
    G⁻ = model.timestepper.G⁻[tracer_name]
    tracer_field = model.tracers[tracer_name]
    grid = model.grid

    launch!(
        architecture(grid), grid, :xyz,
        _ab2_step_tracer_field!, tracer_field, grid, convert(FT, Δt), χ_step, Gⁿ, G⁻
    )
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)

    dump_age_snapshot(model, iter_n, "post_explicit")

    implicit_step!(
        tracer_field,
        model.timestepper.implicit_solver,
        model.closure,
        model.closure_fields,
        Val(tracer_index),
        model.clock,
        fields(model),
        Δt
    )
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)

    dump_age_snapshot(model, iter_n, "post_implicit")

    # Finish time_step!: cache, tick, closure, update_state.
    cache_previous_tendencies!(model)
    tick!(model.clock, Δt)
    step_closure_prognostics!(model, Δt)
    update_state!(model, [])
    model.clock.last_Δt = Δt
    model.timestepper.χ = χ_orig
    return nothing
end

nsteps = parse(Int, get(ENV, "PROBE_NSTEPS", "1"))
@info "PROBE_TEND: rank=$rank PROBE_NSTEPS=$nsteps (full dumps at iter 0..$nsteps; intra dumps at each step)"

dump_full_state(model, 0)

for n in 1:nsteps
    step_with_intra_dumps!(model, Δt, n - 1)
    dump_full_state(model, n)
end

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
@info "PROBE_TEND: rank=$rank all probes complete."
flush(stdout); flush(stderr)
