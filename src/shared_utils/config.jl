################################################################################
# Config env var helpers
#
# Extracted from shared_functions.jl — configuration parsing and project setup.
################################################################################

using TOML

"""Parse and validate the 4 core config env vars."""
function parse_config_env()
    VS = get(ENV, "VELOCITY_SOURCE", "cgridtransports")
    WF = get(ENV, "W_FORMULATION", "wdiagnosed")
    AS = get(ENV, "ADVECTION_SCHEME", "centered2")
    TS = get(ENV, "TIMESTEPPER", "AB2")
    VS ∈ ("bgridvelocities", "cgridtransports", "totaltransport") || error("VELOCITY_SOURCE must be bgridvelocities, cgridtransports, or totaltransport (got: $VS)")
    WF ∈ ("wdiagnosed", "wprescribed") || error("W_FORMULATION must be wdiagnosed or wprescribed (got: $WF)")
    AS ∈ ("centered2", "weno3", "weno5") || error("ADVECTION_SCHEME must be centered2, weno3, or weno5 (got: $AS)")
    TS ∈ ("AB2", "SRK2", "SRK3", "SRK4", "SRK5") || error("TIMESTEPPER must be AB2, SRK2, SRK3, SRK4, or SRK5 (got: $TS)")
    return (; VELOCITY_SOURCE = VS, W_FORMULATION = WF, ADVECTION_SCHEME = AS, TIMESTEPPER = TS)
end

"""Build the unified MODEL_CONFIG directory tag from parsed config + optional env flags."""
function build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)
    wf_tag = W_FORMULATION
    if W_FORMULATION == "wprescribed"
        pw = get(ENV, "PRESCRIBED_W_SOURCE", "parent")
        wf_tag = pw == "diagnosed" ? "wprediag" : "wparent"
    end
    mc = "$(VELOCITY_SOURCE)_$(wf_tag)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"
    lowercase(get(ENV, "GM_REDI", "no")) == "yes" && (mc = "$(mc)_GMREDI")
    lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes" && (mc = "$(mc)_mkappaV")
    return mc
end

"""Convert ADVECTION_SCHEME string to Oceananigans advection object."""
function advection_from_scheme(s::AbstractString)
    return s == "centered2" ? Centered(order = 2) :
        s == "weno3" ? WENO(order = 3) :
        s == "weno5" ? WENO(order = 5) :
        error("Unknown ADVECTION_SCHEME: $s")
end

"""Convert TIMESTEPPER string to Oceananigans timestepper Symbol."""
function timestepper_from_string(s::AbstractString)
    return s == "AB2" ? :QuasiAdamsBashforth2 :
        s == "SRK2" ? :SplitRungeKutta2 :
        s == "SRK3" ? :SplitRungeKutta3 :
        s == "SRK4" ? :SplitRungeKutta4 :
        s == "SRK5" ? :SplitRungeKutta5 :
        error("Unknown TIMESTEPPER: $s")
end

"""
    load_project_config(; parentmodel_arg_index = 1)

Load project configuration from LocalPreferences.toml, ARGS, or ENV.
Priority: ARGS[parentmodel_arg_index] > ENV["PARENT_MODEL"] > TOML defaults > "ACCESS-OM2-1".

Returns a NamedTuple with fields:
  parentmodel, experiment, time_window,
  experiment_dir, monthly_dir, yearly_dir,
  outputdir, Δt_seconds, profile
"""
function load_project_config(; parentmodel_arg_index = 1)
    cfg_file = "LocalPreferences.toml"
    cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

    parentmodel = if length(ARGS) >= parentmodel_arg_index && !isempty(ARGS[parentmodel_arg_index])
        ARGS[parentmodel_arg_index]
    elseif haskey(ENV, "PARENT_MODEL")
        ENV["PARENT_MODEL"]
    else
        get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
    end

    profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
    if profile === nothing
        @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
        Δt = parentmodel == "ACCESS-OM2-1" ? 5400.0 :
            parentmodel == "ACCESS-OM2-025" ? 1800.0 : 400.0
    else
        Δt = Float64(profile["dt_seconds"])
    end

    # Experiment and time window
    default_experiments = Dict(
        "ACCESS-OM2-1" => "1deg_jra55_iaf_omip2_cycle6",
        "ACCESS-OM2-025" => "025deg_jra55_iaf_omip2_cycle6",
    )
    experiment = get(ENV, "EXPERIMENT", get(default_experiments, parentmodel, ""))
    isempty(experiment) && error("No default EXPERIMENT for $parentmodel; set EXPERIMENT env var")
    time_window = get(ENV, "TIME_WINDOW", "1960-1979")

    # Centralized path construction
    experiment_dir = normpath(joinpath(@__DIR__, "..", "..", "preprocessed_inputs", parentmodel, experiment))
    time_window_dir = joinpath(experiment_dir, time_window)
    monthly_dir = joinpath(time_window_dir, "monthly")
    yearly_dir = joinpath(time_window_dir, "yearly")

    # outputdir now includes experiment/time_window
    if profile === nothing
        outputdir = normpath(joinpath(@__DIR__, "..", "..", "outputs", parentmodel, experiment, time_window))
    else
        outputdir = joinpath(profile["outputdir"], experiment, time_window)
    end

    @info "GIT_COMMIT   = $(get(ENV, "GIT_COMMIT", "unknown"))"
    @info "EXPERIMENT   = $experiment"
    @info "TIME_WINDOW  = $time_window"

    return (;
        parentmodel, experiment, time_window,
        experiment_dir, monthly_dir, yearly_dir,
        outputdir, Δt_seconds = Δt, profile,
    )
end
