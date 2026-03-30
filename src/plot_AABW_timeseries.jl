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
    years = ds["year"][:]
    close(ds)

    # AABW metric: min(psi) for lat <= -50, over all depths
    lat_mask = yu .<= -50.0
    psi_south = psi[lat_mask, :, :]  # (lat_subset, depth, year)

    aabw = [minimum(skipmissing(psi_south[:, :, i])) for i in axes(psi_south, 3)]
    # Convert to Sverdrups (psi is in m³/s, 1 Sv = 1e9 m³/s... but check units)
    # MOM ty_trans is in kg/s; dividing by ρ₀ ≈ 1035 gives m³/s; then /1e6 for Sv
    # Actually ty_trans in MOM5 is already in Sv (10⁹ kg/s)... let's check magnitude
    # Values around -15e9 suggest units are kg/s → divide by 1e9 for Sv-equivalent
    aabw_sv = aabw ./ 1.0e9

    return years, aabw_sv
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
    years, aabw = compute_aabw_timeseries(infile)

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

        @info "  $wlen_label: strongest=$tw_strong ($(round(vs; digits = 2)) Sv), weakest=$tw_weak ($(round(vw; digits = 2)) Sv)"
    end

    all_windows = vcat(fixed_windows, aabw_windows)

    # ── Plot ───────────────────────────────────────────────────────────

    fig = Figure(; size = (1200, 700), fonts = (; regular = "DejaVu Sans"))

    ax = Axis(
        fig[1, 1];
        xlabel = "Year",
        ylabel = "AABW transport (Sv)",
        title = "$model — Depth-space AABW transport (min ψ, lat ≤ 50°S)",
        xticks = 1955:5:2020,
    )

    # Plot timeseries
    lines!(ax, years, aabw; color = :black, linewidth = 2, label = "Yearly mean")

    # Highlight time windows as shaded vertical spans
    for (tw, label, color) in all_windows
        y1, y2 = parse_window(tw)
        # Shade the window
        vspan!(ax, y1 - 0.5, y2 + 0.5; color = (color, 0.15))
        # Add label at top
        text!(
            ax, (y1 + y2) / 2, 1.0;
            text = label,
            align = (:center, :top),
            fontsize = 8,
            color = color,
            space = :relative,
        )
    end

    # Add a legend-like annotation for the window categories
    Legend(
        fig[2, 1],
        [PolyElement(; color = (c, 0.3)) for (_, _, c) in all_windows],
        ["$label ($tw)" for (tw, label, _) in all_windows];
        orientation = :horizontal,
        nbanks = 3,
        framevisible = false,
        labelsize = 9,
        patchsize = (15, 10),
    )

    # Save
    outdir = joinpath(@__DIR__, "..", "outputs", model, experiment, "AABW")
    mkpath(outdir)

    outfile = joinpath(outdir, "AABW_depthspace_timeseries.png")
    save(outfile, fig; px_per_unit = 3)
    @info "  Saved: $outfile"

    # Also save the window summary to a text file for reference
    summary_file = joinpath(outdir, "AABW_windows.txt")
    open(summary_file, "w") do io
        println(io, "# AABW-dependent TIME_WINDOWs for $model ($experiment)")
        println(io, "# Metric: min(ψ_depthspace) for lat ≤ 50°S")
        println(io, "#")
        for (tw, label, _) in aabw_windows
            println(io, "$label: $tw")
        end
    end
    @info "  Saved: $summary_file"
end
