################################################################################
# Simulation setup and age validation
#
# Extracted from shared_functions.jl — progress callback, simulation
# configuration for standard age runs, and post-simulation age diagnostics.
################################################################################

using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Architectures: CPU, GPU, architecture, child_architecture
using Oceananigans.Grids: znodes
using Oceananigans.OutputWriters: JLD2Writer
using Adapt: adapt
using Statistics: mean
using Printf: @sprintf

################################################################################
# Progress message callback for simulations
################################################################################

function progress_message(sim)
    if architecture(sim.model.grid) isa Distributed
        # Lightweight progress for distributed runs: avoid GPU→CPU transfer and
        # MPI-synchronous findmax/mean which can deadlock across ranks.
        flush(stdout); flush(stderr)
        return @info @sprintf(
            "  sim iter: %04d, time: %1.3f yr, Δt: %.2e yr, wall: %s\n",
            iteration(sim), time(sim) / year, sim.Δt / year, prettytime(sim.run_wall_time)
        )
    end

    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year)
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout); flush(stderr)
    return @info @sprintf(
        "  sim iter: %04d, time: %1.3f yr, Δt: %.2e yr, max(age) = %.1e yr at (%d, %d, %d), mean(age) = %.1e yr, wall: %s\n",
        iteration(sim), time(sim) / year, sim.Δt / year, max_age, idx_max.I..., mean_age, walltime
    )
end


################################################################################
# Simulation setup for standard age runs
################################################################################

"""
    setup_age_simulation(model, Δt, stop_time, outputdir, model_config, duration_tag;
                          output_interval, progress_interval)

Create a Simulation with progress callback and JLD2 output writer for a standard
age simulation. Returns `(simulation, age_output_dir)`.
"""
function setup_age_simulation(
        model, Δt, stop_time, outputdir, model_config, duration_tag;
        output_interval, progress_interval
    )
    simulation = Simulation(model; Δt, stop_time)
    add_callback!(simulation, progress_message, TimeInterval(progress_interval))

    px = parse(Int, get(ENV, "PARTITION_X", "1"))
    py = parse(Int, get(ENV, "PARTITION_Y", "1"))
    gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
    age_output_dir = isempty(gpu_tag) ?
        joinpath(outputdir, "standardrun", model_config) :
        joinpath(outputdir, "standardrun", model_config, gpu_tag)
    mkpath(age_output_dir)

    # One JLD2Writer per field → one file per variable per rank
    output_fields = Dict(
        :age => model.tracers.age,
        :u => model.velocities.u,
        :v => model.velocities.v,
        :w => model.velocities.w,
        :eta => model.free_surface.displacement,
    )
    for (name, field) in output_fields
        simulation.output_writers[name] = JLD2Writer(
            model, Dict(string(name) => field);
            schedule = TimeInterval(output_interval),
            filename = joinpath(age_output_dir, "$(name)_$(duration_tag)"),
            overwrite_existing = true,
            with_halos = true,
            including = [],  # workaround for #5410: serializeproperty! deadlocks on distributed
        )
    end

    # Save z-star internal fields (∂t_σ, σᶜᶜⁿ, ηⁿ) via manual callback —
    # these are raw OffsetArrays in grid.z, not Fields, so JLD2Writer can't handle them.
    ug = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid
    if ug.z isa MutableVerticalDiscretization
        grid_arch = Oceananigans.Architectures.architecture(model.grid)
        zstar_rank = grid_arch isa Distributed ? grid_arch.local_rank : -1
        zstar_suffix = zstar_rank >= 0 ? "_rank$(zstar_rank)" : ""
        # Remove stale files from previous runs
        for name in ("dt_sigma", "sigma_cc", "eta_n")
            old = joinpath(age_output_dir, "$(name)_$(duration_tag)$(zstar_suffix).jld2")
            isfile(old) && rm(old)
        end
        function save_zstar_fields(sim)
            t = time(sim)
            iter = iteration(sim)
            for (name, arr) in [("dt_sigma", ug.z.∂t_σ), ("sigma_cc", ug.z.σᶜᶜⁿ), ("eta_n", ug.z.ηⁿ)]
                filepath = joinpath(age_output_dir, "$(name)_$(duration_tag)$(zstar_suffix).jld2")
                jldopen(filepath, "a") do f
                    f["timeseries/$(name)/$(iter)"] = Array(parent(arr))
                    f["timeseries/t/$(iter)"] = t
                end
            end
            return
        end
        add_callback!(simulation, save_zstar_fields, TimeInterval(output_interval))
    end

    # Manual callback: save actual model field data with halos directly from parent(field.data).
    # JLD2Writer wraps fields in anonymous ComputedFields whose fill_halo_regions! uses default
    # sign=+1 BCs (since the wrapper isn't named :u/:v, it can't dispatch to sign=-1).
    # This callback captures the TRUE model field halos for diagnostic comparison.
    # Only enabled on CPU (scalar indexing and Array(parent(...)) fail on GPU).
    grid_arch = Oceananigans.Architectures.architecture(model.grid)
    is_cpu = child_architecture(grid_arch) isa CPU
    manual_rank = grid_arch isa Distributed ? grid_arch.local_rank : -1
    manual_suffix = manual_rank >= 0 ? "_rank$(manual_rank)" : ""
    manual_fields = Dict(
        "u" => model.velocities.u,
        "v" => model.velocities.v,
        "w" => model.velocities.w,
        "eta" => model.free_surface.displacement,
    )
    if is_cpu
        # Remove stale manual/fts files from previous runs
        for name in keys(manual_fields)
            for suffix in ("_manual_", "_fts_")
                old = joinpath(age_output_dir, "$(name)$(suffix)$(duration_tag)$(manual_suffix).jld2")
                isfile(old) && rm(old)
            end
        end
        # Save bottom height (from PartialCellBottom) for diagnostics
        if model.grid isa ImmersedBoundaryGrid
            ib = model.grid.immersed_boundary
            if hasproperty(ib, :bottom_height)
                bottom_path = joinpath(age_output_dir, "bottom_$(duration_tag)$(manual_suffix).jld2")
                isfile(bottom_path) && rm(bottom_path)
                jldopen(bottom_path, "w") do f
                    f["bottom"] = Array(parent(ib.bottom_height.data))
                end
                @info "Saved bottom_height: size=$(size(parent(ib.bottom_height.data)))"
            end
        end
        # Save fields via three methods for comparison:
        # (a) Pointwise TSI evaluation at every index (including halos)
        # (b) FTS snapshot parent data (raw halos from load_fts)
        # (c) Regular parent(field.data) for non-TSI fields (w, etc.)
        manual_tmp = Dict{String, Field}()
        manual_fts = Dict{String, Any}()
        for (name, field) in manual_fields
            if field isa Oceananigans.OutputReaders.TimeSeriesInterpolation
                loc = Oceananigans.Fields.location(field)
                manual_tmp[name] = Field{loc[1], loc[2], loc[3]}(model.grid)
                manual_fts[name] = field.time_series
                @info "Manual callback '$name': TSI field, tmp axes=$(axes(manual_tmp[name].data)), FTS data axes=$(axes(field.time_series.data)[1:3])"
            end
        end
        flush(stdout); flush(stderr)
        function save_manual_fields(sim)
            t = time(sim)
            iter = iteration(sim)
            for (name, field) in manual_fields
                if haskey(manual_tmp, name)
                    tmp = manual_tmp[name]
                    # (a) Pointwise TSI evaluation (no @inbounds — let errors surface)
                    for k in axes(tmp.data, 3), j in axes(tmp.data, 2), i in axes(tmp.data, 1)
                        tmp.data[i, j, k] = field[i, j, k]
                    end
                    data_eval = Array(parent(tmp.data))
                    # (b) FTS snapshot 1 parent data (raw halos from load_fts)
                    fts = manual_fts[name]
                    data_fts = Array(parent(fts[1].data))
                    # Save both
                    filepath_eval = joinpath(age_output_dir, "$(name)_manual_$(duration_tag)$(manual_suffix).jld2")
                    jldopen(filepath_eval, "a") do f
                        f["timeseries/$(name)/$(iter)"] = data_eval
                        f["timeseries/t/$(iter)"] = t
                    end
                    filepath_fts = joinpath(age_output_dir, "$(name)_fts_$(duration_tag)$(manual_suffix).jld2")
                    jldopen(filepath_fts, "a") do f
                        f["timeseries/$(name)/$(iter)"] = data_fts
                        f["timeseries/t/$(iter)"] = t
                    end
                else
                    # (c) Regular field (w, etc.)
                    data = Array(parent(field.data))
                    filepath = joinpath(age_output_dir, "$(name)_manual_$(duration_tag)$(manual_suffix).jld2")
                    jldopen(filepath, "a") do f
                        f["timeseries/$(name)/$(iter)"] = data
                        f["timeseries/t/$(iter)"] = t
                    end
                end
            end
            return
        end
        add_callback!(simulation, save_manual_fields, TimeInterval(output_interval))
    end # is_cpu

    @info "Simulation configured: stop_time=$(stop_time / year) yr, output_dir=$age_output_dir"
    flush(stdout); flush(stderr)

    return simulation, age_output_dir
end


################################################################################
# Age field validation (post-simulation diagnostics)
################################################################################

"""
    validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label="simulation")

Run 5 diagnostic tests on the age field after a simulation:
1. Max age bound check (should not exceed elapsed time by >10%)
2. Surface age near zero (surface relaxation working)
3. Non-negativity (advection scheme oscillations)
4. Depth-averaged profile (per-level statistics)
5. Hotspot inspection (neighbors of max age cell)
"""
function validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label = "simulation")
    @info "Validating age field after $label"
    flush(stdout); flush(stderr)

    age_data = Array(interior(model.tracers.age))
    elapsed_time = time(simulation)

    (; wet3D, idx, Nidx) = compute_wet_mask(grid)
    Nx′, Ny′, Nz′ = size(wet3D)
    age_wet = age_data[idx]

    # ── Test 1: Max age bound ────────────────────────────────────────────────
    max_age_val = maximum(age_wet)
    max_age_ratio = max_age_val / elapsed_time
    @info "Max age bound check:" max_age_years = max_age_val / year ratio_to_elapsed = max_age_ratio
    if max_age_ratio > 1.1
        @warn "Max age exceeds 1.1× elapsed time — possible numerical overshoot or bug"
    end

    # ── Test 2: Surface age should be near zero ──────────────────────────────
    surface_mask = wet3D[:, :, end]
    surface_ages = age_data[:, :, end][surface_mask]
    max_surface_age = maximum(abs, surface_ages)
    mean_surface_age = mean(surface_ages)
    @info "Surface age:" max_days = max_surface_age / day mean_days = mean_surface_age / day
    if max_surface_age > 1day
        @warn "Surface age exceeds 1 day — surface relaxation may not be working correctly"
    end

    # ── Test 3: Non-negativity ───────────────────────────────────────────────
    n_negative = count(x -> x < 0, age_wet)
    min_age_val = minimum(age_wet)
    @info "Non-negativity:" n_negative fraction = n_negative / Nidx min_age_days = min_age_val / day
    if n_negative > 0
        @warn "Found $n_negative negative age values (min = $(min_age_val / day) days) — advection scheme ($ADVECTION_SCHEME) may produce oscillations"
    end

    # ── Test 4: Depth-averaged profile ───────────────────────────────────────
    z_centers = Array(znodes(grid, Center(), Center(), Center()))
    @info "Volume-weighted mean age by depth level:"
    flush(stdout); flush(stderr)
    for k in Nz′:-1:1
        level_mask = wet3D[:, :, k]
        n_wet = count(level_mask)
        if n_wet == 0
            continue
        end
        level_ages = age_data[:, :, k][level_mask]
        level_mean = mean(level_ages)
        level_max = maximum(level_ages)
        level_min = minimum(level_ages)
        z_val = z_centers[k]
        @info @sprintf(
            "  k=%3d  z=%7.0fm  mean=%6.2f yr  max=%6.2f yr  min=%6.2f yr  n_wet=%d",
            k, z_val, level_mean / year, level_max / year, level_min / year, n_wet
        )
    end
    flush(stdout); flush(stderr)

    # ── Test 5: Hotspot inspection ───────────────────────────────────────────
    max_idx = argmax(age_data)
    mi, mj, mk = Tuple(max_idx)
    @info "Max age hotspot at (i=$mi, j=$mj, k=$mk):" age_years = age_data[mi, mj, mk] / year z_meters = z_centers[mk]
    for dk in -1:1, dj in -1:1, di in -1:1
        ni, nj, nk = mi + di, mj + dj, mk + dk
        if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′ && wet3D[ni, nj, nk]
            @info @sprintf("  neighbor (%+d,%+d,%+d): age = %6.2f yr", di, dj, dk, age_data[ni, nj, nk] / year)
        end
    end
    flush(stdout); flush(stderr)

    # ── Summary ──────────────────────────────────────────────────────────────
    @info "Validation summary:" max_age_years = max_age_val / year mean_wet_age_years = mean(age_wet) / year max_surface_days = max_surface_age / day n_negative n_wet_cells = Nidx
    flush(stdout); flush(stderr)

    return nothing
end
