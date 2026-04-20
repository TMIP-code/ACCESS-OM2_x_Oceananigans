#=
Temporal blocking for tracer advance on distributed tripolar grids.

Ported from the validated MWE:
  /g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar_distributed.jl
  /g/data/y99/bp3051/.julia/dev/Oceananigans/temporal_blocking_tripolar.md

Requires halo size `H ≥ K + 1` on each partitioned horizontal
direction; Centered(order=2) + AB2 only. Replaces `run!` / `time_step!`
— `Simulation` callbacks do not fire inside the batch, so callers must
pass any FTS-backed field updates through `fts_updates`.
=#

using Oceananigans.BoundaryConditions: fill_halo_regions!, _fill_bottom_and_top_halo!
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: tick!, update_state!, maybe_prepare_first_time_step!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_tracer_tendencies!,
    compute_tracer_flux_bcs!,
    _ab2_step_tracer_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.Fields: instantiated_location
using Oceananigans.OutputReaders: Time

extended_kernel_parameters(K, k, Nx, Ny, Nz) =
let margin = K - k + 1
    KernelParameters((1 - margin):(Nx + margin), (1 - margin):(Ny + margin), 1:Nz)
end

function compute_tracer_tendencies_in_halo!(model, kp)
    # Direct call bypasses complete_communication_and_compute_tracer_buffer!,
    # which on distributed models syncs MPI halos AND recomputes the buffer
    # region with standard :xyz KPs — overwriting extended-KP work.
    compute_hydrostatic_tracer_tendencies!(model, kp; active_cells_map = nothing)
    compute_tracer_flux_bcs!(model)
    return nothing
end

function ab2_step_tracers_in_halo!(tracers, model, Δt, χ, kp)
    grid = model.grid
    FT = eltype(grid)
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        G⁻ = model.timestepper.G⁻[tracer_name]
        tracer_field = tracers[tracer_name]
        launch!(
            architecture(grid), grid, kp,
            _ab2_step_tracer_field!, tracer_field, grid,
            convert(FT, Δt), χ, Gⁿ, G⁻,
        )
        implicit_step!(
            tracer_field, model.timestepper.implicit_solver,
            model.closure, model.closure_fields,
            Val(tracer_index), model.clock, fields(model), Δt,
        )
    end
    return nothing
end

# Reapply the z-BC over an xy range extended by `margin` halo cells on each
# side. The standard z-fill covers only the interior (Nx, Ny); corner cells
# at (x-halo, y-halo, z-halo) are otherwise untouched by any fill and the
# stencil there diverges exponentially across sub-steps. Local operation.
function fill_z_halos_over_extended_xy!(tracers, grid, Nx, Ny, margin)
    arch = architecture(grid)
    kp = KernelParameters((1 - margin):(Nx + margin), (1 - margin):(Ny + margin))
    for name in propertynames(tracers)
        c = tracers[name]
        bottom_bc = c.boundary_conditions.bottom
        top_bc = c.boundary_conditions.top
        loc = instantiated_location(c)
        launch!(
            arch, grid, kp, _fill_bottom_and_top_halo!,
            c.data, bottom_bc, top_bc, loc, grid, (),
        )
    end
    return nothing
end

"""
    multi_time_step!(model, Δt, Nx, Ny, Nz; K, fts_updates=(),
                     sync_gc_nbatches=0, batch_index=Ref(0))

Run `K` tracer sub-steps behind a single MPI halo exchange.

`fts_updates` is a tuple of `(target_field, source_fts)` pairs that are
`set!` at each sub-step (after `tick!`) and halo-filled with
`only_local_halos=true`. Use this to replace any `IterationInterval(1)`
Simulation callback that prescribes a field from an FTS.

`sync_gc_nbatches` (default 0 = disabled) fires `GC.gc(false)` once every
`sync_gc_nbatches` calls, immediately after the batch-end `update_state!`
collective. The shared `batch_index::Ref{Int}` lets the caller carry the
counter across calls and reset it between warmup and timed regions.
Note: the unit is **batches**, not raw steps — N batches = N·K raw steps.
"""
function multi_time_step!(
        model, Δt, Nx, Ny, Nz; K, fts_updates = (),
        sync_gc_nbatches::Int = 0, batch_index::Ref{Int} = Ref(0)
    )
    FT = eltype(model.grid)
    Δt_FT = convert(FT, Δt)

    maybe_prepare_first_time_step!(model, [])

    for k in 1:K
        euler = (Δt_FT != model.clock.last_Δt)
        χ = ifelse(euler, convert(FT, -0.5), model.timestepper.χ)
        kp = extended_kernel_parameters(K, k, Nx, Ny, Nz)

        compute_tracer_tendencies_in_halo!(model, kp)
        ab2_step_tracers_in_halo!(model.tracers, model, Δt_FT, χ, kp)

        # Cache Gⁿ → G⁻ including halos (parent arrays, not just :xyz).
        for name in propertynames(model.tracers)
            parent(model.timestepper.G⁻[name]) .= parent(model.timestepper.Gⁿ[name])
        end

        tick!(model.clock, Δt_FT)

        # Per-sub-step FTS-backed field updates (η, κV, optional T/S).
        # Local only — no MPI.
        if !isempty(fts_updates)
            t_now = Time(model.clock.time)
            for (target, source) in fts_updates
                set!(target, source[t_now])
                fill_halo_regions!(target; only_local_halos = true)
            end
        end

        if k < K
            # Local halo fill — applies BC-derived halos (Periodic, Fold,
            # z-Bounded) without triggering MPI halo communication.
            fill_halo_regions!(model.tracers; only_local_halos = true)

            # Corner-cell fix: reapply z-BC over the extended xy range for
            # the next sub-step. Without it, corner halos go stale and
            # blow up over K sub-steps.
            margin_next = K - (k + 1) + 1
            fill_z_halos_over_extended_xy!(model.tracers, model.grid, Nx, Ny, margin_next)
        end
    end

    # The one MPI halo exchange for the batch.
    update_state!(model)
    for name in propertynames(model.tracers)
        fill_halo_regions!(model.timestepper.G⁻[name])
    end

    # Synchronized GC at a known collective-sync point. Every rank reaches
    # this line at the same wall-clock instant (update_state! just synced),
    # so GC.gc(false) on all ranks costs max(GC_i) instead of sum(GC_i).
    batch_index[] += 1
    if sync_gc_nbatches > 0 && batch_index[] % sync_gc_nbatches == 0
        GC.gc(false)
    end

    return nothing
end
