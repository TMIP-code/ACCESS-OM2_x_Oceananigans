# Plot depth-space AABW transport timeseries with selected TIME_WINDOWs highlighted
#
# Usage:
#   julia --project=. src/plot_AABW_timeseries.jl
#
# Reads psi_tot_year.nc from /scratch/y99/TMIP/data/{model}/{experiment}/depthspace/
# Outputs plots to outputs/{model}/{experiment}/AABW/

using Pkg
Pkg.activate(".")

using NCDatasets
using CairoMakie
using Statistics

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT = get(ENV, "PROJECT", "y99")
datadir = "/scratch/$PROJECT/TMIP/data"

models = [
    ("ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6"),
    ("ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6"),
]

# Fixed time windows (same for both models)
fixed_windows = [
    ("1958-1987", "First 30yr", :royalblue),
    ("1989-2018", "Last 30yr", :firebrick),
    ("1958-1977", "First 20yr", :dodgerblue),
    ("1999-2018", "Last 20yr", :orangered),
    ("1958-1967", "First 10yr", :steelblue),
    ("2009-2018", "Last 10yr", :tomato),
]

# ── Helper functions ───────────────────────────────────────────────────────

function parse_window(tw::String)
    if contains(tw, "-")
        parts = split(tw, "-")
        return parse(Int, parts[1]), parse(Int, parts[2])
    else
        y = parse(Int, tw)
        return y, y
    end
end

function compute_aabw_timeseries(filepath)
    ds = NCDataset(filepath)
    psi = ds["psi_tot"][:, :, :]  # NCDatasets: (yu_ocean, st_ocean, year)
    yu = ds["yu_ocean"][:]
    st = ds["st_ocean"][:]
    years = ds["year"][:]
    close(ds)

    # Unit conversion: ty_trans in kg/s → divide by 1e9 for Sv-equivalent
    to_sv = 1.0e-9

    # Metric 1: min(psi) for lat ≤ -60°S, all depths
    lat_mask_60S = yu .<= -60.0
    psi_60S = psi[lat_mask_60S, :, :]
    aabw_60S = [minimum(skipmissing(psi_60S[:, :, i])) for i in axes(psi_60S, 3)] .* to_sv

    # Metric 2: min(psi) for lat ≤ 0° (Southern Hemisphere), depth ≥ 3000m
    lat_mask_SH = yu .<= 0.0
    depth_mask_deep = st .>= 3000.0
    psi_deep_SH = psi[lat_mask_SH, depth_mask_deep, :]
    aabw_deep = [minimum(skipmissing(psi_deep_SH[:, :, i])) for i in axes(psi_deep_SH, 3)] .* to_sv

    # Metric 3: min(psi) for lat ≤ -50°S, all depths (original metric)
    lat_mask_50S = yu .<= -50.0
    psi_50S = psi[lat_mask_50S, :, :]
    aabw_50S = [minimum(skipmissing(psi_50S[:, :, i])) for i in axes(psi_50S, 3)] .* to_sv

    return years, aabw_60S, aabw_deep, aabw_50S
end

function find_rolling_extrema(years, vals, window_length)
    n = length(vals)
    if window_length == 1
        imin = argmin(vals)  # most negative = strongest AABW
        imax = argmax(vals)  # least negative = weakest AABW
        return (years[imin], years[imin], vals[imin]),
            (years[imax], years[imax], vals[imax])
    end

    nw = n - window_length + 1
    rolling_mean = [mean(vals[i:(i + window_length - 1)]) for i in 1:nw]
    imin = argmin(rolling_mean)
    imax = argmax(rolling_mean)

    strongest = (years[imin], years[imin + window_length - 1], rolling_mean[imin])
    weakest = (years[imax], years[imax + window_length - 1], rolling_mean[imax])
    return strongest, weakest
end

# ── Main ───────────────────────────────────────────────────────────────────

for (model, experiment) in models
    infile = joinpath(datadir, model, experiment, "depthspace", "psi_tot_year.nc")
    if !isfile(infile)
        @warn "Skipping $model: $infile not found"
        continue
    end

    @info "Processing $model"
    years, aabw_60S, aabw_deep, aabw_50S = compute_aabw_timeseries(infile)

    # Three metrics with their labels and series
    metrics = [
        (aabw_60S, "min ψ, lat ≤ 60°S", "AABW_60S"),
        (aabw_deep, "min ψ, lat ≤ 0°, depth ≥ 3000m", "AABW_deep"),
        (aabw_50S, "min ψ, lat ≤ 50°S", "AABW_50S"),
    ]

    for (aabw, metric_label, metric_tag) in metrics
        # Find AABW-dependent windows
        aabw_windows = Tuple{String, String, Symbol}[]
        for (wlen, wlen_label) in [(10, "10yr"), (3, "3yr"), (1, "1yr")]
            strongest, weakest = find_rolling_extrema(years, aabw, wlen)
            y1s, y2s, vs = strongest
            y1w, y2w, vw = weakest

            tw_strong = y1s == y2s ? "$y1s" : "$y1s-$y2s"
            tw_weak = y1w == y2w ? "$y1w" : "$y1w-$y2w"

            push!(aabw_windows, (tw_strong, "Max AABW $wlen_label", :darkgreen))
            push!(aabw_windows, (tw_weak, "Min AABW $wlen_label", :darkorange))

            @info "  [$metric_tag] $wlen_label: strongest=$tw_strong ($(round(vs; digits = 2)) Sv), weakest=$tw_weak ($(round(vw; digits = 2)) Sv)"
        end

        all_windows = vcat(fixed_windows, aabw_windows)

        # ── Plot ───────────────────────────────────────────────────────────

        fig = Figure(; size = (900, 500), fonts = (; regular = "DejaVu Sans"))

        ax = Axis(
            fig[1, 1];
            xlabel = "Year",
            ylabel = "AABW transport (Sv)",
            title = "$model — Depth-space AABW transport ($metric_label)",
            xticks = 1960:10:2020,
            limits = (nothing, nothing, 0, nothing),
        )

        # Plot timeseries (negate: min(ψ) is negative, show as positive transport)
        aabw_plot = .-aabw
        lines!(ax, years, aabw_plot; color = :black, linewidth = 2, label = "Yearly mean")

        # Highlight time windows as shaded vertical spans
        all_windows = vcat(fixed_windows, aabw_windows)
        for (tw, label, color) in all_windows
            window_start, window_end = parse_window(tw)
            vspan!(ax, window_start - 0.5, window_end + 0.5; color = (color, 0.15))
        end

        # Brackets below the timeseries at staggered y levels
        # Fixed:  30yr→y=1, 20yr→y=2, 10yr→y=3
        # AABW:   10yr→y=4, 3yr→y=5, 1yr→y=6
        # Deep metric (larger values) gets 2× y to keep brackets below the data
        yscale = metric_tag == "AABW_deep" ? 2 : 1
        bracket_data = Tuple{String, String, Symbol, Int}[]  # (tw, label, color, y)
        for (tw, label, color) in fixed_windows
            window_start, window_end = parse_window(tw)
            span = window_end - window_start + 1
            by = (span >= 25 ? 1 : span >= 15 ? 2 : 3) * yscale
            push!(bracket_data, (tw, label, color, by))
        end
        for (i, (tw, label, color)) in enumerate(aabw_windows)
            # aabw_windows order: 10yr max, 10yr min, 3yr max, 3yr min, 1yr max, 1yr min
            by = (i <= 2 ? 4 : i <= 4 ? 5 : 6) * yscale
            push!(bracket_data, (tw, label, color, by))
        end

        for (tw, label, color, by) in bracket_data
            window_start, window_end = parse_window(tw)
            bracket!(
                ax, window_start - 0.5, by, window_end + 0.5, by;
                offset = 2, width = 5, text = label, style = :square, orientation = :down,
                color = :black, textcolor = :black, fontsize = 10
            )
        end

        # Save
        outdir = joinpath(@__DIR__, "..", "outputs", model, experiment, "AABW")
        mkpath(outdir)

        outfile = joinpath(outdir, "$(metric_tag)_depthspace_timeseries.png")
        save(outfile, fig; px_per_unit = 3)
        @info "  Saved: $outfile"

        # Save window summary
        summary_file = joinpath(outdir, "$(metric_tag)_windows.txt")
        open(summary_file, "w") do io
            println(io, "# AABW-dependent TIME_WINDOWs for $model ($experiment)")
            println(io, "# Metric: $metric_label")
            println(io, "#")
            for (tw, label, _) in aabw_windows
                println(io, "$label: $tw")
            end
        end
        @info "  Saved: $summary_file"
    end
end
