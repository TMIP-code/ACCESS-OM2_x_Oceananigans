# Plot AABW transport timeseries with selected TIME_WINDOWs highlighted.
#
# Controlled by SPACE env var:
#   SPACE=depth (default) — read depth-space psi_tot_year.nc + psi_tot.nc
#                           from {datadir}/{model}/{experiment}/depthspace/
#                           and apply depth-based metrics (depth cutoffs).
#   SPACE=rho             — read density-space psi_tot_global.nc + psi_tot_rollingyear_global.nc
#                           from {datadir}/{model}/{experiment}/rhospace/
#                           and apply density-based metrics (σ₀ threshold).
#
# Usage:
#   julia --project=. src/plot_AABW_timeseries.jl
#   SPACE=rho julia --project=. src/plot_AABW_timeseries.jl

using Pkg
Pkg.activate(".")

using NCDatasets
using CairoMakie
using Dates
using Statistics

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT = get(ENV, "PROJECT", "y99")
datadir = "/scratch/$PROJECT/TMIP/data"

SPACE = get(ENV, "SPACE", "depth")
SPACE in ("depth", "rho") || error("SPACE must be 'depth' or 'rho', got '$SPACE'")

models = if SPACE == "rho"
    [
        ("ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6"),
        ("ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6"),
        ("ACCESS-OM2-01", "01deg_jra55v140_iaf_cycle3"),
    ]
else
    [
        ("ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6"),
        ("ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6"),
    ]
end

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

    # Metric 1: Upper cell — min(psi) for lat ≤ -60°S, depth < 3000m
    lat_mask_60S = yu .<= -60.0
    depth_mask_shallow = st .< 3000.0
    psi_upper = psi[lat_mask_60S, depth_mask_shallow, :]
    aabw_upper = [minimum(skipmissing(psi_upper[:, :, i])) for i in axes(psi_upper, 3)] .* to_sv

    # Metric 2: Deep cell — min(psi) for lat ≤ 0°, depth ≥ 3000m
    lat_mask_SH = yu .<= 0.0
    depth_mask_deep = st .>= 3000.0
    psi_deep_SH = psi[lat_mask_SH, depth_mask_deep, :]
    aabw_deep = [minimum(skipmissing(psi_deep_SH[:, :, i])) for i in axes(psi_deep_SH, 3)] .* to_sv

    # Metric 3: min(psi) for lat ≤ -50°S, all depths (legacy metric)
    lat_mask_50S = yu .<= -50.0
    psi_50S = psi[lat_mask_50S, :, :]
    aabw_50S = [minimum(skipmissing(psi_50S[:, :, i])) for i in axes(psi_50S, 3)] .* to_sv

    return years, aabw_upper, aabw_deep, aabw_50S
end

function compute_aabw_monthly(filepath)
    ds = NCDataset(filepath)
    psi = ds["psi_tot"][:, :, :]  # (yu_ocean, st_ocean, time)
    yu = ds["yu_ocean"][:]
    st = ds["st_ocean"][:]
    time_vals = ds["time"][:]  # DateTime values
    close(ds)

    to_sv = 1.0e-9
    nt = size(psi, 3)

    # Convert DateTime to fractional year for plotting
    time_frac = [Dates.year(t) + (Dates.month(t) - 0.5) / 12 for t in time_vals]

    # Upper cell: lat ≤ -60°S, depth < 3000m
    lat_mask_60S = yu .<= -60.0
    depth_mask_shallow = st .< 3000.0
    psi_upper = psi[lat_mask_60S, depth_mask_shallow, :]
    aabw_upper = [minimum(skipmissing(psi_upper[:, :, i])) for i in 1:nt] .* to_sv

    # Deep cell: lat ≤ 0°, depth ≥ 3000m
    lat_mask_SH = yu .<= 0.0
    depth_mask_deep = st .>= 3000.0
    psi_deep_SH = psi[lat_mask_SH, depth_mask_deep, :]
    aabw_deep = [minimum(skipmissing(psi_deep_SH[:, :, i])) for i in 1:nt] .* to_sv

    # Legacy 50S: lat ≤ -50°S, all depths
    lat_mask_50S = yu .<= -50.0
    psi_50S = psi[lat_mask_50S, :, :]
    aabw_50S = [minimum(skipmissing(psi_50S[:, :, i])) for i in 1:nt] .* to_sv

    return time_frac, time_vals, aabw_upper, aabw_deep, aabw_50S
end

"""
    yearly_mean_of_monthly(time_vals, monthly_vals)

Compute mean(metric(month)) per year — i.e., apply the spatial metric to each month
first, then average over the 12 months. This differs from metric(mean(month)) which
averages the field first then applies the spatial metric.
"""
function yearly_mean_of_monthly(time_vals, monthly_vals)
    yrs = [Dates.year(t) for t in time_vals]
    unique_yrs = sort(unique(yrs))
    yearly_means = Float64[]
    for y in unique_yrs
        idx = findall(yrs .== y)
        push!(yearly_means, mean(monthly_vals[idx]))
    end
    return unique_yrs, yearly_means
end

"""
Load density-space ψ from a NetCDF written by `compute_MOC_rho_timeseries.py`.
Returns (psi, lat, potrho, time_vals) with ψ already in Sv.
"""
function load_rho_psi(filepath)
    ds = NCDataset(filepath, "r"; maskingvalue = NaN)
    psi = ds["psi_tot"][:, :, :]    # (grid_yu_ocean, potrho, time)
    lat = ds["grid_yu_ocean"][:]
    potrho = ds["potrho"][:]
    time_vals = ds["time"][:]
    close(ds)
    return psi, lat, potrho, time_vals
end

"""
Apply the density-space AABW metric to each time step of `psi`:
min ψ over (lat < LAT_MAX, σ₀ ≥ RHO_THRESHOLD). ψ is already in Sv.
"""
function rho_aabw_metric(psi, lat, potrho; lat_max, rho_threshold)
    lat_mask = lat .< lat_max
    rho_mask = potrho .>= rho_threshold
    sub = psi[lat_mask, rho_mask, :]
    return [minimum(skipmissing(sub[:, :, i])) for i in axes(sub, 3)]
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
    # Load data + build per-metric yearly/monthly series.
    metrics = if SPACE == "depth"
        infile = joinpath(datadir, model, experiment, "depthspace", "psi_tot_year.nc")
        isfile(infile) || (@warn "Skipping $model: $infile not found"; continue)

        @info "Processing $model (depth-space)"
        years, aabw_upper, aabw_deep, aabw_50S = compute_aabw_timeseries(infile)

        monthly_file = joinpath(datadir, model, experiment, "depthspace", "psi_tot.nc")
        monthly_time, monthly_dates, monthly_upper, monthly_deep, monthly_50S =
            compute_aabw_monthly(monthly_file)

        [
            (
                years, monthly_time, monthly_dates, aabw_upper, monthly_upper,
                "min ψ, lat ≤ 60°S, depth < 3000m", "AABW_upper", "Depth-space",
            ),
            (
                years, monthly_time, monthly_dates, aabw_deep, monthly_deep,
                "min ψ, lat ≤ 0°, depth ≥ 3000m", "AABW_deep", "Depth-space",
            ),
            (
                years, monthly_time, monthly_dates, aabw_50S, monthly_50S,
                "min ψ, lat ≤ 50°S", "AABW_50S", "Depth-space",
            ),
        ]
    else  # rho
        monthly_file = joinpath(datadir, model, experiment, "rhospace", "psi_tot_global.nc")
        isfile(monthly_file) || (@warn "Skipping $model: $monthly_file not found"; continue)

        @info "Processing $model (density-space)"
        psi_m, lat, potrho, dates_m = load_rho_psi(monthly_file)
        monthly_time = [Dates.year(t) + (Dates.month(t) - 0.5) / 12 for t in dates_m]

        # Candidate metric: min ψ for lat < 0, σ₀ ≥ 1036
        aabw_m = rho_aabw_metric(psi_m, lat, potrho; lat_max = 0.0, rho_threshold = 1036.0)
        years, aabw_y = yearly_mean_of_monthly(dates_m, aabw_m)

        [
            (
                years, monthly_time, dates_m, aabw_y, aabw_m,
                "min ψ, lat < 0°, σ₀ ≥ 1036 kg/m³", "AABW_rho_lt0_1036", "Density-space",
            ),
        ]
    end

    for metric in metrics
        years, monthly_time, monthly_dates, aabw, aabw_monthly,
            metric_label, metric_tag, space_label = metric

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
            title = "$model — $space_label AABW transport ($metric_label)",
            xticks = 1960:10:2020,
            limits = (nothing, nothing, 0, nothing),
        )

        # Plot timeseries (negate: min(ψ) is negative, show as positive transport)
        monthly_plot = .-aabw_monthly
        lines!(ax, monthly_time, monthly_plot; color = (:gray60, 0.5), linewidth = 0.5, label = "Monthly")

        # Yearly series (for depth-space this is metric(mean) from the yearly file;
        # for rho-space we derive it as mean(metric) — see dispatch above)
        yearly_label = SPACE == "depth" ? "metric(mean)" : "mean(metric)"
        lines!(ax, years, .-aabw; color = :black, linewidth = 2, label = yearly_label)

        # Additional mean(metric) trace for depth-space (differs from the yearly file)
        if SPACE == "depth"
            mean_metric_years, mean_metric_vals = yearly_mean_of_monthly(monthly_dates, aabw_monthly)
            lines!(ax, mean_metric_years, .-mean_metric_vals; color = :red, linewidth = 2, label = "mean(metric)")
        end

        axislegend(ax; position = :rt, framevisible = true, labelsize = 10)

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

        space_suffix = SPACE == "depth" ? "depthspace" : "rhospace"
        outfile = joinpath(outdir, "$(metric_tag)_$(space_suffix)_timeseries.png")
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
