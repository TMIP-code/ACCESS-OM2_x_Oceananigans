"""
Plot per-rank partitioned FTS files against the global FTS to validate
`partition_data.jl`.

Loads (CPU, no model construction):
- Global monthly FTS from `preprocessed_inputs/{PM}/{EXP}/{TW}/monthly/*.jld2`
- Per-rank partition files from
  `preprocessed_inputs/{PM}/{EXP}/{TW}/partitions/{P}/*_monthly_rank*.jld2`

For each field (u, v, w, eta), and for the first time snapshot, plots
serial parent (with halos), per-rank parent (with halos), and the diff
against the corresponding slice of the serial parent.

Environment variables:
  PARENT_MODEL     – default: ACCESS-OM2-1
  EXPERIMENT       – default: from env_defaults
  TIME_WINDOW      – default: 1960-1979
  PARTITION        – e.g. "1x2", "2x2" (REQUIRED)
  VELOCITY_SOURCE  – cgridtransports | totaltransport | bgridvelocities (default cgridtransports)

Usage:
    PARTITION=1x2 julia --project src/plot_partitioned_fts.jl
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory, Cyclical
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Units: day, days, second, seconds
using CairoMakie
using JLD2
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, monthly_dir) = load_project_config()
VELOCITY_SOURCE = get(ENV, "VELOCITY_SOURCE", "cgridtransports")

PARTITION = get(ENV, "PARTITION", "1x2")
px, py = parse.(Int, split(PARTITION, "x"))
nranks = px * py

partition_dir = joinpath(dirname(monthly_dir), "partitions", PARTITION)
isdir(partition_dir) || error("Partition dir not found: $partition_dir — run partition step first")

plot_dir = joinpath(experiment_dir, "plots", "partition_check_$(PARTITION)")
mkpath(plot_dir)

@info "Configuration"
@info "  PARENT_MODEL  = $parentmodel"
@info "  PARTITION     = $PARTITION  ($nranks ranks)"
@info "  monthly_dir   = $monthly_dir"
@info "  partition_dir = $partition_dir"
@info "  plot_dir      = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Field list
################################################################################

if VELOCITY_SOURCE == "totaltransport"
    fts_fields = [
        ("u_from_total_transport", "u"),
        ("v_from_total_transport", "v"),
        ("w_from_total_transport", "w"),
        ("eta", "η"),
    ]
elseif VELOCITY_SOURCE == "cgridtransports"
    fts_fields = [
        ("u_from_mass_transport", "u"),
        ("v_from_mass_transport", "v"),
        ("w_from_mass_transport", "w"),
        ("eta", "η"),
    ]
else
    fts_fields = [
        ("u_interpolated", "u"),
        ("v_interpolated", "v"),
        ("w", "w"),
        ("eta", "η"),
    ]
end

################################################################################
# Load grid (CPU only) so we can build the serial FTS to read parent arrays
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading serial grid from $grid_file"
flush(stdout); flush(stderr)
serial_grid = load_tripolar_grid(grid_file, CPU())

################################################################################
# Helpers
################################################################################

function plot_field(data, title, filepath; symmetric = true)
    nx, ny = size(data, 1), size(data, 2)
    fig = Figure(; size = (1000, 600))
    ax = Axis(
        fig[1, 1];
        title,
        xlabel = "i (with halos)", ylabel = "j (with halos)",
        backgroundcolor = :lightgray,
    )
    valid = filter(!isnan, vec(data))
    if isempty(valid) || maximum(abs, valid) == 0
        vmax = 1.0e-10
    else
        vmax = maximum(abs, valid)
    end
    if symmetric
        hm = heatmap!(
            ax, 1:nx, 1:ny, data;
            colorrange = (-vmax, vmax), colormap = :balance, nan_color = :lightgray,
        )
    else
        hm = heatmap!(ax, 1:nx, 1:ny, data; nan_color = :lightgray)
    end
    Colorbar(fig[1, 2], hm)
    save(filepath, fig)
    return nothing
end

################################################################################
# Loop over fields
################################################################################

for (file_prefix, field_name) in fts_fields
    monthly_file = joinpath(monthly_dir, "$(file_prefix)_monthly.jld2")
    if !isfile(monthly_file)
        @warn "Skipping $field_name — global FTS not found: $monthly_file"
        continue
    end

    @info "─── $field_name ───"

    # Load the serial FTS so we can get parent(field) including halos
    cpu_fts = FieldTimeSeries(
        monthly_file, field_name;
        architecture = CPU(), grid = serial_grid,
        backend = InMemory(),
        time_indexing = Cyclical(1 * 365.25days),
    )
    fill_halo_regions!(cpu_fts)

    serial_parent = Array(parent(cpu_fts[1].data))
    nx_s, ny_s, nz_s = size(serial_parent)
    k_plot = nz_s == 1 ? 1 : nz_s - 7   # near-surface, skipping top halo
    @info "  serial parent size = ($nx_s, $ny_s, $nz_s), plotting k=$k_plot"

    # Plot serial
    plot_field(
        serial_parent[:, :, k_plot],
        "$field_name SERIAL parent (k=$k_plot, size=$(size(serial_parent)))",
        joinpath(plot_dir, "$(field_name)_serial_k$(k_plot).png"),
    )

    # Loop over ranks
    for r in 0:(nranks - 1)
        rank_file = joinpath(partition_dir, "$(file_prefix)_monthly_rank$(r).jld2")
        if !isfile(rank_file)
            @warn "  rank $r: file not found ($rank_file)"
            continue
        end

        # Read the per-rank partition file
        rank_data, Hx_r, Hy_r, local_Nx, local_Ny = jldopen(rank_file, "r") do f
            (
                f["data/1"],
                f["Hx"],
                f["Hy"],
                f["local_Nx"],
                f["local_Ny"],
            )
        end
        nx_r, ny_r, nz_r = size(rank_data)

        # Recover (rx, ry) from rank index for the rank's logical position.
        # Convention used by partition_data.jl: rank = rx * py + ry
        rx = r ÷ py
        ry = r % py
        # Compute x_offset / y_offset using the same convention as
        # partition_data.jl (sum of local sizes of preceding ranks).
        # Here we assume even partitioning (cell counts = grid.Nx ÷ px, grid.Ny ÷ py)
        # because partition_data.jl writes local_Nx / local_Ny per rank but the
        # offsets are reconstructed from local_size accumulation across ranks.
        # For a 1x2 / 2x2 setup these are simply ry * local_Ny and rx * local_Nx.
        x_offset = rx * local_Nx
        y_offset = ry * local_Ny

        global_i_range = (1 + x_offset):(local_Nx + x_offset + 2 * Hx_r)
        global_j_range = (1 + y_offset):(local_Ny + y_offset + 2 * Hy_r)

        # Slice the serial parent the same way partition_data.jl does
        # so the diff captures any inconsistency in the saved file.
        serial_slice = serial_parent[global_i_range, global_j_range, :]
        nx_ss, ny_ss, nz_ss = size(serial_slice)

        @info @sprintf(
            "  rank %d (rx=%d, ry=%d): rank_data=%s serial_slice=%s",
            r, rx, ry, size(rank_data), size(serial_slice),
        )

        # Adjust k for plotting if z dim differs (eta has nz=1)
        k_r = nz_r == 1 ? 1 : nz_r - 7

        # Plot rank data
        plot_field(
            rank_data[:, :, k_r],
            "$field_name rank $r ($rx,$ry) (k=$k_r, size=$(size(rank_data)))",
            joinpath(plot_dir, "$(field_name)_rank$(r)_k$(k_r).png"),
        )

        # Plot serial slice for the same rank
        plot_field(
            serial_slice[:, :, k_r],
            "$field_name SERIAL SLICE for rank $r (k=$k_r, size=$(size(serial_slice)))",
            joinpath(plot_dir, "$(field_name)_serialslice_rank$(r)_k$(k_r).png"),
        )

        # Diff (only if shapes match)
        if size(rank_data) == size(serial_slice)
            diff = rank_data[:, :, k_r] .- serial_slice[:, :, k_r]
            max_diff = maximum(abs, filter(!isnan, vec(diff)); init = 0.0)
            @info @sprintf("    max|diff|=%.3e", max_diff)
            plot_field(
                diff,
                "$field_name DIFF rank $r (k=$k_r, max|diff|=$(@sprintf("%.2e", max_diff)))",
                joinpath(plot_dir, "$(field_name)_diff_rank$(r)_k$(k_r).png"),
            )
        else
            @error "    SHAPE MISMATCH: rank $r data $(size(rank_data)) vs serial slice $(size(serial_slice)) — diff plot skipped"
        end
    end
    flush(stdout); flush(stderr)
end

@info "All plots saved to $plot_dir"
flush(stdout); flush(stderr)
