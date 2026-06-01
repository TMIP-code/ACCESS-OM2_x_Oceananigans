# Plot AABW + NADW transport timeseries for each configured resolution.
#
# 2x1 figure per resolution:
#   Top: AABW (1 line) — `min ψ` over (lat < 0°, ρ ≥ 1036 kg/m³) on the GLOBAL
#        density-space ψ. Monthly faint trace + yearly bold trace.
#   Bottom: NADW (4 lines) — `max ψ` on the ATLANTIC-only ψ, four boxes:
#        {26.5°N or 26°N–65°N} × {all ρ or ρ ∈ [1035, 1037] kg/m³}.
#        Yearly traces only (monthlies would be too busy with 4 lines).
#
# Both panels share decadal vspans at 1968–1977 and 1999–2008 (the paper's
# two ventilation windows).
#
# Usage:
#   julia --project src/plot_AABW_NADW_timeseries.jl

using Pkg
Pkg.activate(".")

using NCDatasets
using CairoMakie
using Dates
using Statistics

include("shared_utils/water_mass_metrics.jl")

# ── Configuration ─────────────────────────────────────────────────────────

PROJECT = get(ENV, "PROJECT", "y99")
datadir = "/scratch/$PROJECT/TMIP/data"

models = [
    ("ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6"),
    ("ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6"),
]

# Decadal windows that bracket the cross-resolution ventilation paper's analysis
decadal_windows = [
    (1968, 1977, :royalblue, "1968–1977"),
    (1999, 2008, :firebrick, "1999–2008"),
]

# 4 NADW metric specifications: (label, kwargs to rho_nadw_metric, line color)
nadw_specs = [
    (
        "26.5°N, all ρ",
        (lat_min = 26.0, lat_max = 27.0), :black,
    ),
    (
        "26.5°N, ρ ∈ [1035, 1037] kg/m³",
        (lat_min = 26.0, lat_max = 27.0, rho_min = 1035.0, rho_max = 1037.0), :firebrick,
    ),
    (
        "26°N–65°N, all ρ",
        (lat_min = 26.0, lat_max = 65.0), :royalblue,
    ),
    (
        "26°N–65°N, ρ ∈ [1035, 1037] kg/m³",
        (lat_min = 26.0, lat_max = 65.0, rho_min = 1035.0, rho_max = 1037.0), :darkorange,
    ),
]

# Cross-resolution combined-figure styling — one entry per model in `models`
combined_styles = [
    (color = :black, linestyle = :solid),
    (color = :darkorange, linestyle = :solid),
]
combined_data = NamedTuple[]

# ── Main loop ─────────────────────────────────────────────────────────────

for (model, experiment) in models
    global_file = joinpath(datadir, model, experiment, "rhospace", "psi_tot_global.nc")
    atlantic_file = joinpath(datadir, model, experiment, "rhospace", "psi_tot_atlantic.nc")

    if !isfile(global_file)
        @warn "Skipping $model: $global_file not found"
        continue
    end
    if !isfile(atlantic_file)
        @warn "Skipping $model: $atlantic_file not found (run BASIN=atlantic compute first)"
        continue
    end

    @info "Processing $model"

    # Load global ψ for AABW
    psi_glo, lat_glo, potrho_glo, dates_glo = load_rho_psi(global_file)
    monthly_time_glo = [Dates.year(t) + (Dates.month(t) - 0.5) / 12 for t in dates_glo]

    aabw_monthly = rho_aabw_metric(
        psi_glo, lat_glo, potrho_glo;
        lat_max = 0.0, rho_threshold = 1036.0
    )
    aabw_years, aabw_yearly = yearly_mean_of_monthly(dates_glo, aabw_monthly)
    # AABW is the negative cell — negate for plotting as positive transport.
    aabw_monthly_plot = -aabw_monthly
    aabw_yearly_plot = -aabw_yearly

    # Load Atlantic ψ for NADW
    psi_atl, lat_atl, potrho_atl, dates_atl = load_rho_psi(atlantic_file)

    length(dates_glo) == length(dates_atl) && all(dates_glo .== dates_atl) ||
        error(
        "Time axis mismatch between $global_file and $atlantic_file — " *
            "rebuild psi_tot_atlantic.nc"
    )

    nadw_series = map(nadw_specs) do (label, kwargs, color)
        monthly = rho_nadw_metric(psi_atl, lat_atl, potrho_atl; kwargs...)
        yrs, yearly = yearly_mean_of_monthly(dates_atl, monthly)
        (label = label, color = color, years = yrs, yearly = yearly)
    end

    # ── Figure ────────────────────────────────────────────────────────────

    fig = Figure(; size = (1000, 750), fonts = (; regular = "DejaVu Sans"))

    ax_aabw = Axis(
        fig[1, 1];
        ylabel = "AABW transport (Sv)",
        title = "$model — AABW (global ψ) and NADW (Atlantic ψ) " *
            "density-space transport timeseries",
        xticks = 1960:10:2020,
        limits = (nothing, nothing, 0, nothing)
    )
    ax_nadw = Axis(
        fig[2, 1];
        xlabel = "Year",
        ylabel = "NADW transport (Sv)",
        xticks = 1960:10:2020,
        limits = (nothing, nothing, 0, nothing)
    )
    linkxaxes!(ax_aabw, ax_nadw)
    hidexdecorations!(ax_aabw; ticks = false, grid = false)

    # Decadal vspans on both panels
    for (y1, y2, color, _) in decadal_windows
        vspan!(ax_aabw, y1 - 0.5, y2 + 0.5; color = (color, 0.12))
        vspan!(ax_nadw, y1 - 0.5, y2 + 0.5; color = (color, 0.12))
    end

    # AABW top panel
    lines!(
        ax_aabw, monthly_time_glo, aabw_monthly_plot;
        color = (:gray60, 0.5), linewidth = 0.5, label = "Monthly"
    )
    lines!(
        ax_aabw, aabw_years, aabw_yearly_plot;
        color = :black, linewidth = 2,
        label = "Yearly mean of monthly metric: min ψ, lat < 0°, ρ ≥ 1036 kg/m³"
    )
    axislegend(ax_aabw; position = :rt, labelsize = 9, framevisible = true)

    # NADW bottom panel — 4 yearly traces
    for s in nadw_series
        lines!(
            ax_nadw, s.years, s.yearly;
            color = s.color, linewidth = 2, label = s.label
        )
    end
    axislegend(ax_nadw; position = :rb, labelsize = 9, framevisible = true, nbanks = 2)

    # Save
    outdir = joinpath(@__DIR__, "..", "outputs", model, experiment, "AABW_NADW")
    mkpath(outdir)
    outfile = joinpath(outdir, "AABW_NADW_rhospace_timeseries.png")
    save(outfile, fig; px_per_unit = 3)
    @info "Saved: $outfile"

    # Compact magnitude summary for the user's sanity check
    @info "  AABW range (Sv):  min=$(round(minimum(aabw_yearly_plot); digits = 2))  " *
        "max=$(round(maximum(aabw_yearly_plot); digits = 2))"
    for s in nadw_series
        finite = filter(isfinite, s.yearly)
        if !isempty(finite)
            @info "  NADW [$(s.label)]:  min=$(round(minimum(finite); digits = 2))  " *
                "max=$(round(maximum(finite); digits = 2))  Sv"
        end
    end

    # Stash data for cross-resolution combined figures (NADW: 4th spec only,
    # i.e. 26°N–65°N × ρ ∈ [1035, 1037]).
    push!(
        combined_data, (
            model = model,
            aabw_years = aabw_years,
            aabw_yearly_plot = aabw_yearly_plot,
            nadw_years = nadw_series[4].years,
            nadw_yearly = nadw_series[4].yearly,
        )
    )
end

# ── Cross-resolution combined figures ────────────────────────────────────

if length(combined_data) == length(models)
    combined_outdir = joinpath(@__DIR__, "..", "outputs", "cross_resolution", "AABW_NADW")
    mkpath(combined_outdir)

    # Style helpers
    fontsize = 14
    decadal_vspans!(ax) = for (y1, y2, color, _) in decadal_windows
        vspan!(ax, y1 - 0.5, y2 + 0.5; color = (color, 0.1))
    end

    # 1) AABW + NADW (2×1)
    fig_an = Figure(; size = (650, 550), fontsize = fontsize)
    ax_a = Axis(
        fig_an[1, 1];
        ylabel = "AABW (Sv)",
        title = "AABW (global ψ) & NADW (Atlantic ψ, 26°N–65°N, ρ∈[1035,1037] kg/m³)",
        titlesize = fontsize,
        xticks = 1960:10:2020, limits = (nothing, nothing, 0, nothing),
    )
    ax_n = Axis(
        fig_an[2, 1];
        xlabel = "Year", ylabel = "NADW (Sv)",
        xticks = 1960:10:2020, limits = (nothing, nothing, 0, nothing),
    )
    linkxaxes!(ax_a, ax_n)
    hidexdecorations!(ax_a; ticks = false, grid = false)
    decadal_vspans!(ax_a); decadal_vspans!(ax_n)
    for (d, s) in zip(combined_data, combined_styles)
        lines!(
            ax_a, d.aabw_years, d.aabw_yearly_plot;
            color = s.color, linestyle = s.linestyle, linewidth = 2, label = d.model,
        )
        lines!(
            ax_n, d.nadw_years, d.nadw_yearly;
            color = s.color, linestyle = s.linestyle, linewidth = 2, label = d.model,
        )
    end
    axislegend(ax_a; position = :rt, labelsize = fontsize - 2, framevisible = true)
    rowgap!(fig_an.layout, 6)

    outfile_an = joinpath(combined_outdir, "AABW_NADW_combined_rhospace_timeseries.png")
    save(outfile_an, fig_an; px_per_unit = 3)
    @info "Saved: $outfile_an"

    # 2) AABW only (1×1)
    fig_a = Figure(; size = (650, 350), fontsize = fontsize)
    ax = Axis(
        fig_a[1, 1];
        xlabel = "Year", ylabel = "AABW (Sv)",
        title = "AABW transport (global ψ, min ψ over lat<0°, ρ≥1036 kg/m³)",
        titlesize = fontsize,
        xticks = 1960:10:2020, limits = (nothing, nothing, 0, nothing),
    )
    decadal_vspans!(ax)
    for (d, s) in zip(combined_data, combined_styles)
        lines!(
            ax, d.aabw_years, d.aabw_yearly_plot;
            color = s.color, linestyle = s.linestyle, linewidth = 2, label = d.model,
        )
    end
    axislegend(ax; position = :rt, labelsize = fontsize - 2, framevisible = true)

    outfile_a = joinpath(combined_outdir, "AABW_combined_rhospace_timeseries.png")
    save(outfile_a, fig_a; px_per_unit = 3)
    @info "Saved: $outfile_a"
end
