################################################################################
# Config env var helpers
#
# Extracted from shared_functions.jl — configuration parsing and project setup.
################################################################################

using TOML
using Oceananigans.Grids: znodes, Center
using Oceananigans.Architectures: on_architecture, CPU

"""
    require_env(var; help="")

Read a required production-config env var or error with a clear message
pointing the user at `scripts/env_defaults.sh`. Use for any var documented
in `env_defaults.sh` / `model_configs/*.sh`. For benchmark / per-script
knobs (NYEARS, BENCHMARK_STEPS, etc.), prefer `get(ENV, var, default)`.
"""
function require_env(var::AbstractString; help::AbstractString = "")
    return get(ENV, var) do
        msg = "Required env var '$var' is not set."
        isempty(help) || (msg *= " " * help)
        msg *= "\nSource scripts/env_defaults.sh before running julia, or set the var explicitly."
        error(msg)
    end
end

"""Parse and validate the 4 core config env vars."""
function parse_config_env()
    VS = require_env("VELOCITY_SOURCE")
    WF = require_env("W_FORMULATION")
    AS = require_env("ADVECTION_SCHEME")
    TS = require_env("TIMESTEPPER")
    VS ∈ ("cgridtransports", "totaltransport") || error("VELOCITY_SOURCE must be cgridtransports or totaltransport (got: $VS)")
    WF ∈ ("wdiagnosed", "wprescribed") || error("W_FORMULATION must be wdiagnosed or wprescribed (got: $WF)")
    AS ∈ ("centered2", "weno3", "weno5", "upwind1") || error("ADVECTION_SCHEME must be centered2, weno3, weno5, or upwind1 (got: $AS)")
    TS ∈ ("AB2", "SRK2", "SRK3", "SRK4", "SRK5") || error("TIMESTEPPER must be AB2, SRK2, SRK3, SRK4, or SRK5 (got: $TS)")
    return (; VELOCITY_SOURCE = VS, W_FORMULATION = WF, ADVECTION_SCHEME = AS, TIMESTEPPER = TS)
end

"""
    parse_kappa_env() -> (; κH, κVML, κVBG)

Parse the diffusivity env vars (m²/s): horizontal scalar diffusivity `KAPPA_H`
(also reused for the GM-Redi isopycnal κ), and the vertical diffusivity in the
mixed layer `KAPPA_V_ML` and ocean interior `KAPPA_V_BG`. These have per-model
defaults in `model_configs/*.sh` (scaled by resolution; see README).
"""
function parse_kappa_env()
    κH = parse(Float64, require_env("KAPPA_H"))
    κVML = parse(Float64, require_env("KAPPA_V_ML"))
    κVBG = parse(Float64, require_env("KAPPA_V_BG"))
    all(>(0), (κH, κVML, κVBG)) || error("KAPPA_H/KAPPA_V_ML/KAPPA_V_BG must be positive (got κH=$κH, κVML=$κVML, κVBG=$κVBG)")
    return (; κH, κVML, κVBG)
end

"""
    parse_lump_and_spray(s = get(ENV, "LUMP_AND_SPRAY", "no"))
        -> (; di, dj, dk, on, tag, dir_suffix)

Parse the `LUMP_AND_SPRAY` env var into coarsening factors and naming tags.

Accepted values:
- `"no"` — no coarsening: `(di=0, dj=0, dk=0, on=false, tag="prec", dir_suffix="")`.
- `"<A>x<B>"` (positive ints) — coarsen horizontally only: `(di=A, dj=B, dk=1,
  on=true, tag="Q{A}x{B}", dir_suffix="_Q{A}x{B}")`.

The `tag` replaces today's `lumpspray_tag = "LSprec" | "prec"` everywhere it's
used to compose output filenames. The `dir_suffix` is appended to the NK
subdir name (e.g. `NK` → `NK_Q5x5`) and to the matrix subdir holding the
coarsened-matrix artefacts (`Mc.jld2`, `LUMP.jld2`, `SPRAY.jld2`).

The legacy `"yes"` alias is no longer accepted (use `"2x2"` for the previous
hardcoded default).
"""
function parse_lump_and_spray(s::AbstractString = require_env("LUMP_AND_SPRAY"))
    sl = lowercase(s)
    if sl == "no"
        return (; di = 0, dj = 0, dk = 0, on = false, tag = "prec", dir_suffix = "")
    elseif sl == "yes"
        error(
            "LUMP_AND_SPRAY=yes is no longer supported; use 'no' or '<int>x<int>' " *
                "(e.g. '2x2' for the previous hardcoded default).",
        )
    end
    m = match(r"^(\d+)x(\d+)$", sl)
    m === nothing &&
        error("LUMP_AND_SPRAY must be 'no' or '<int>x<int>' (got: $s)")
    di = parse(Int, m.captures[1])
    dj = parse(Int, m.captures[2])
    (di > 0 && dj > 0) ||
        error("LUMP_AND_SPRAY factors must be positive integers (got: $s)")
    tag = "Q$(di)x$(dj)"
    return (; di, dj, dk = 1, on = true, tag, dir_suffix = "_$tag")
end

"""
    parse_omega(s = require_env("OMEGA"))
        -> (; kind::Symbol, depth_m::Float64, tag::String, suffix::String)

Parse the `OMEGA` env var that selects the age-source region.

Accepted values:
- `"all"` (default) — source applied everywhere. `kind=:all`, `suffix=""`.
- `"z<D>"` — source applied only where `z_center < -D` m (i.e. deeper than
  `D` metres). `kind=:zdeep`, `depth_m=Float64(D)`, `suffix="_z<D>"`.

The `suffix` is appended to age output filenames (and derived plots) so that
multiple OMEGA values can coexist in the same directory. `OMEGA=all`
produces no suffix, preserving filenames from before this feature existed.
"""
function parse_omega(s::AbstractString = require_env("OMEGA"))
    sl = lowercase(s)
    sl == "all" && return (; kind = :all, depth_m = NaN, tag = "all", suffix = "")
    m = match(r"^z(\d+(?:\.\d+)?)$", sl)
    m === nothing && error("OMEGA must be 'all' or 'z<depth>' (got: $s)")
    D = parse(Float64, m.captures[1])
    digits = m.captures[1]
    return (; kind = :zdeep, depth_m = D, tag = "z$(digits)", suffix = "_z$(digits)")
end

"""Convenience: return the OMEGA filename-suffix string for the current env."""
omega_filename_suffix() = parse_omega().suffix

"""
    build_omega_k_mask(grid, omega; arch)

Return a length-`Nz` `Vector{Float64}` (on `arch`) where `mask[k] = 1.0` if
cell-centre at level `k` is inside the OMEGA source region, else `0.0`.

On the tripolar grid `z` at `(Center, Center, Center)` is independent of
`(i, j)`, so a 1-D k-mask is exact.
"""
function build_omega_k_mask(grid, omega; arch)
    Nz = size(grid, 3)
    cpu_mask = ones(Float64, Nz)
    if omega.kind === :zdeep
        zc = Array(znodes(grid, Center(), Center(), Center()))
        for k in 1:Nz
            cpu_mask[k] = zc[k] < -omega.depth_m ? 1.0 : 0.0
        end
    end
    return on_architecture(arch, cpu_mask)
end


"""
Return `true` if the immersed-boundary grid should be built with
`active_cells_map = true` (the default), `false` otherwise.

Controlled by the `ACTIVE_CELLS_MAP` env var (yes | no, default yes). Setting
it to "no" also causes `noACM_suffix()` to return "_noACM" so output files
don't collide with the default runs.
"""
function active_cells_map_enabled()
    return lowercase(require_env("ACTIVE_CELLS_MAP")) ≠ "no"
end

"""Suffix to tack onto duration tags / file names when ACTIVE_CELLS_MAP=no."""
function noACM_suffix()
    return active_cells_map_enabled() ? "" : "_noACM"
end

"""Return the sorted list of positive divisors of `n`."""
function _divisors(n::Integer)
    n ≥ 1 || error("_divisors requires n ≥ 1 (got: $n)")
    ds = Int[]
    i = 1
    while i * i ≤ n
        if mod(n, i) == 0
            push!(ds, i)
            i * i == n || push!(ds, n ÷ i)
        end
        i += 1
    end
    return sort!(ds)
end

"""Convert ADVECTION_SCHEME string to Oceananigans advection object."""
function advection_from_scheme(s::AbstractString)
    return s == "centered2" ? Centered(order = 2) :
        s == "weno3" ? WENO(order = 3) :
        s == "weno5" ? WENO(order = 5) :
        s == "upwind1" ? UpwindBiased(order = 1) :
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
  parentmodel, experiment,
  time_window, mld_time_window,
  experiment_dir, monthly_dir, yearly_dir,
  mld_monthly_dir, mld_yearly_dir,
  outputdir, Δt_seconds, profile

When `MLD_TIME_WINDOW` is set in the environment (regardless of value), the
MLD-related directories are derived from it (decoupled from `TIME_WINDOW`,
which then refers to the transport-window only) and `outputdir` is routed
under a `test/TR{TIME_WINDOW}_MLD{MLD_TIME_WINDOW}/` subtree to keep test
runs isolated from production trees. When `MLD_TIME_WINDOW` is unset,
behaviour is fully back-compatible.
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
    Δt_base = Δt

    # TIMESTEP_MULT: scale Δt for the tracer-advection stability sweep.
    # The dynamical-core Δt_base from the parent model is the wrong constraint
    # for a passive-tracer offline simulation (see docs/timestep_multiplier.md).
    M_str = require_env("TIMESTEP_MULT")
    M = tryparse(Int, M_str)
    M === nothing && error("TIMESTEP_MULT must be a positive integer (got: \"$M_str\")")
    M ≥ 1 || error("TIMESTEP_MULT must be ≥ 1 (got: $M)")
    year_seconds = 365.25 * 86400
    N_base = round(Int, year_seconds / Δt_base)
    if mod(N_base, M) ≠ 0
        all_divisors = _divisors(N_base)
        practical_M_max = floor(Int, 18 * 3600 / Δt_base)
        valid_practical = filter(≤(practical_M_max), all_divisors)
        next_idx = findfirst(>(practical_M_max), all_divisors)
        next_hint = next_idx === nothing ? "" :
            " Next valid value is $(all_divisors[next_idx]) (= 1 month per step)."
        error(
            "TIMESTEP_MULT=$M is not a divisor of N_base=$N_base (= year/Δt_base " *
                "for $parentmodel). Valid multipliers ≤ $practical_M_max " *
                "(Δt ≤ 18 h): {$(join(valid_practical, ", "))}." * next_hint
        )
    end
    Δt = M * Δt_base

    # Experiment and time window
    default_experiments = Dict(
        "ACCESS-OM2-1" => "1deg_jra55_iaf_omip2_cycle6",
        "ACCESS-OM2-025" => "025deg_jra55_iaf_omip2_cycle6",
    )
    experiment = get(ENV, "EXPERIMENT", get(default_experiments, parentmodel, ""))
    isempty(experiment) && error("No default EXPERIMENT for $parentmodel; set EXPERIMENT env var")
    time_window = require_env("TIME_WINDOW")

    # MLD time window: explicit env var only — `haskey` (not `get` with default)
    # so we can detect "set vs unset" and route outputs accordingly.
    mld_explicit = haskey(ENV, "MLD_TIME_WINDOW") && !isempty(ENV["MLD_TIME_WINDOW"])
    mld_time_window = mld_explicit ? ENV["MLD_TIME_WINDOW"] : time_window

    # Centralized path construction
    experiment_dir = normpath(joinpath(@__DIR__, "..", "..", "preprocessed_inputs", parentmodel, experiment))
    time_window_dir = joinpath(experiment_dir, time_window)
    monthly_dir = joinpath(time_window_dir, "monthly")
    yearly_dir = joinpath(time_window_dir, "yearly")

    mld_time_window_dir = joinpath(experiment_dir, mld_time_window)
    mld_monthly_dir = joinpath(mld_time_window_dir, "monthly")
    mld_yearly_dir = joinpath(mld_time_window_dir, "yearly")

    # outputdir: test/ subtree when MLD_TIME_WINDOW is explicitly set,
    # otherwise unchanged from production layout.
    output_tag = mld_explicit ?
        joinpath("test", "TR$(time_window)_MLD$(mld_time_window)") :
        time_window
    if profile === nothing
        outputdir = normpath(joinpath(@__DIR__, "..", "..", "outputs", parentmodel, experiment, output_tag))
    else
        outputdir = joinpath(profile["outputdir"], experiment, output_tag)
    end

    @info "GIT_COMMIT       = $(get(ENV, "GIT_COMMIT", "unknown"))"
    @info "TIMESTEP_MULT    = $M  (Δt_base = $Δt_base s → Δt = $Δt s)"
    @info "EXPERIMENT       = $experiment"
    @info "TIME_WINDOW      = $time_window"
    @info "MLD_TIME_WINDOW  = $mld_time_window$(mld_explicit ? "" : "  (default = TIME_WINDOW; not set)")"
    @info "experiment_dir   = $experiment_dir"
    @info "monthly_dir      = $monthly_dir"
    @info "yearly_dir       = $yearly_dir"
    @info "mld_monthly_dir  = $mld_monthly_dir"
    @info "mld_yearly_dir   = $mld_yearly_dir"
    @info "outputdir        = $outputdir"

    return (;
        parentmodel, experiment,
        time_window, mld_time_window,
        experiment_dir, monthly_dir, yearly_dir,
        mld_monthly_dir, mld_yearly_dir,
        outputdir, Δt_seconds = Δt, profile,
    )
end
