"""
Plot velocity and age outputs from run_ACCESS-OM2.jl simulation.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q normal -l walltime=01:00:00 -l ncpus=12 \
-l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_outputs.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)

Alternatively, pass the JLD2 output filepath as ARGS[1].
"""

@info "Loading packages for plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries
using CairoMakie
using Statistics
using TOML
using JLD2

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if length(ARGS) >= 2
    ARGS[2]
elseif haskey(ENV, "PARENT_MODEL")
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
arch_str = "CPU"

# Determine JLD2 output filepath
if !isempty(ARGS)
    output_filepath = ARGS[1]
else
    age_output_dir = joinpath(outputdir, "age", model_config)
    output_filepath = joinpath(age_output_dir, "age_1year.jld2")
end

@info "Plotting outputs from: $output_filepath"
flush(stdout); flush(stderr)

################################################################################
# Load output lazily from disk
################################################################################

@info "Loading output lazily from disk for visualization"
flush(stdout); flush(stderr)
u_lazy = FieldTimeSeries(output_filepath, "u")
v_lazy = FieldTimeSeries(output_filepath, "v")
w_lazy = FieldTimeSeries(output_filepath, "w")
η_lazy = FieldTimeSeries(output_filepath, "η")

################################################################################
# Plot u, v, w, η for each output time
################################################################################

for itime in eachindex(u_lazy.times)
    itime_str = "$itime/$(length(u_lazy.times))"
    @info "Visualizing output $itime_str"
    flush(stdout); flush(stderr)
    for k in 25:25
        local fig = Figure(size = (1200, 2400))
        local ax = Axis(fig[1, 1], title = "C-grid u[k=$k, output=$itime_str]")
        local velocity2D = view(interior(u_lazy[itime]), :, :, k)
        local maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        local hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[1, 2], hm)
        ax = Axis(fig[2, 1], title = "C-grid v[k=$k, output=$itime_str]")
        velocity2D = view(interior(v_lazy[itime]), :, :, k)
        maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[2, 2], hm)
        ax = Axis(fig[3, 1], title = "C-grid w[k=$k, output=$itime_str]")
        velocity2D = view(interior(w_lazy[itime]), :, :, k + 1)
        maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[3, 2], hm)
        ax = Axis(fig[4, 1], title = "C-grid η[output=$itime_str]")
        velocity2D = interior(η_lazy[itime], :, :, 1)
        maxvelocity = quantile(abs.(velocity2D[.!isnan.(velocity2D)]), 0.9)
        hm = heatmap!(ax, velocity2D; colormap = :RdBu_9, colorrange = maxvelocity .* (-1, 1), nan_color = :black)
        Colorbar(fig[4, 2], hm)
        k_dir = joinpath(outputdir, "velocities", "uvweta", model_config, "k$(k)")
        mkpath(k_dir)
        @show fig_file_name = joinpath(k_dir, "CGrid_velocities_final_k$(k)_output$(itime)_$(arch_str).png")
        flush(stdout); flush(stderr)
        save(fig_file_name, fig)
        fig = nothing
        GC.gc()
    end
end

@info "Plotting complete"
flush(stdout); flush(stderr)
