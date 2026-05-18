"""
Probe (u, v, η, T, S, κV) for the OM2-025 / 1999-2008 TRAF blow-up.

The cell (i = 1288, j = 1047, k = 36) is where the first Φ! call diverged
to max(age) = 6.8e+25 yr by sim iter 122. Cell location: high latitude,
33 cells south of the j = Ny = 1080 tripolar fold (so the fold mapping
itself doesn't apply at this row, but stencil reach into the halos above
might if the field uses a wide stencil).

For each FTS this script:
  1. Loads forward (IAF) values and a copy.
  2. Applies `reverse_fts_time!` to the copy with the same `flip_sign`
     setup_model.jl uses under TRAF (true for u/v, false otherwise).
  3. Prints values at the bad cell + relevant face neighbours for ALL
     12 months, for both the forward and the reversed FTS.
  4. Verifies the mirror identity:
        rev[i, j, k, m] ≟ sgn · fwd[i, j, k, N + 1 - m]
     and flags any deviation. (No fold sign-flip applies because the
     cell is in the interior, well below the fold halo.)
  5. Also reports NaN / Inf counts in the parent array across all
     months, to catch any single bad value.

Face neighbours we probe:
  - u lives at (Face, Center, Center): print u[1288, 1047, 36] (west face
    of cell 1288) AND u[1289, 1047, 36] (east face of cell 1288).
  - v lives at (Center, Face, Center): print v[1288, 1047, 36] (south
    face of cell 1047) AND v[1288, 1048, 36] (north face of cell 1047).
  - η lives at (Center, Center, Nothing): only one cell, (1288, 1047).
"""

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
using Oceananigans.Fields: interior
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
using Printf

include("../src/shared_functions.jl")

cfg = load_project_config()
(; parentmodel, experiment, time_window, experiment_dir, monthly_dir, mld_monthly_dir) = cfg

@info "Probe configuration"
@info "- PARENT_MODEL = $parentmodel"
@info "- EXPERIMENT   = $experiment"
@info "- TIME_WINDOW  = $time_window"

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
Nx, Ny, Nz = size(grid)
@info "Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"

# Cells of interest. The center cell of the blow-up is (1288, 1047, 36).
const I = 1288
const J = 1047
const K = 36

# Per-FTS probe configuration: (filename, name, flip_sign, dir, probe-cells)
# Probe cells are (i, j, k) tuples in the field's own interior indexing.
specs = [
    (
        file = "u_from_total_transport_monthly.jld2",
        name = "u",
        flip_sign = true,
        dir = monthly_dir,
        probes = [(I, J, K), (I + 1, J, K)],   # west, east face of x-cell I
    ),
    (
        file = "v_from_total_transport_monthly.jld2",
        name = "v",
        flip_sign = true,
        dir = monthly_dir,
        probes = [(I, J, K), (I, J + 1, K)],   # south, north face of y-cell J
    ),
    (
        file = "eta_monthly.jld2",
        name = "η",
        flip_sign = false,
        dir = monthly_dir,
        probes = [(I, J, 1)],                  # η has only k=1 in interior
    ),
    (
        file = "temp_monthly.jld2",
        name = "T",
        flip_sign = false,
        dir = monthly_dir,
        probes = [(I, J, K)],
    ),
    (
        file = "salt_monthly.jld2",
        name = "S",
        flip_sign = false,
        dir = monthly_dir,
        probes = [(I, J, K)],
    ),
    (
        file = "kappa_v_monthly.jld2",
        name = "κV",
        flip_sign = false,
        dir = mld_monthly_dir,
        probes = [(I, J, K)],
    ),
]

backend = InMemory()
time_indexing = Cyclical(1year)

for spec in specs
    println()
    println("="^90)
    println("FTS: $(spec.name)  ($(spec.file))  flip_sign=$(spec.flip_sign)")
    println("="^90)

    path = joinpath(spec.dir, spec.file)
    isfile(path) || error("Missing FTS file: $path")

    fwd = load_fts(CPU(), path, spec.name, grid; backend, time_indexing)
    rev = load_fts(CPU(), path, spec.name, grid; backend, time_indexing)
    reverse_fts_time!(rev; flip_sign = spec.flip_sign)

    N = length(fwd.times)
    sgn = spec.flip_sign ? -1 : 1

    # NaN / Inf scan across the full parent (interior + halos) for all months
    nan_count = 0
    inf_count = 0
    big_count = 0
    bigthresh = 1.0e10
    for m in 1:N
        p = parent(fwd[m])
        nan_count += count(isnan, p)
        inf_count += count(isinf, p)
        big_count += count(x -> abs(x) > bigthresh, p)
    end
    println()
    println("NaN/Inf/large scan over all 12 months (parent including halos):")
    println(@sprintf("  NaN count: %d", nan_count))
    println(@sprintf("  Inf count: %d", inf_count))
    println(@sprintf("  |val| > %.0e count: %d", bigthresh, big_count))

    for probe in spec.probes
        i, j, k = probe
        # Bounds check
        Ifwd = interior(fwd[1])
        ni, nj, nk = size(Ifwd)
        if !(1 ≤ i ≤ ni && 1 ≤ j ≤ nj && 1 ≤ k ≤ nk)
            println()
            println("  Probe ($i, $j, $k) is OUT OF BOUNDS for interior size ($ni, $nj, $nk) — skipping")
            continue
        end

        println()
        println("  Probe cell ($i, $j, $k):")
        println("    m  | fwd[m]          | rev[m]          | sgn·fwd[N+1-m] | rev == mirror ?")
        println("    ---+-----------------+-----------------+----------------+----------------")
        for m in 1:N
            mirror_m = N + 1 - m
            fwd_val = interior(fwd[m])[i, j, k]
            rev_val = interior(rev[m])[i, j, k]
            mirror_val = sgn * interior(fwd[mirror_m])[i, j, k]
            match = isapprox(rev_val, mirror_val; atol = 1.0e-15, rtol = 1.0e-12)
            tag = match ? "✓" : @sprintf("✗ Δ=%+.3e", rev_val - mirror_val)
            println(@sprintf("    %2d | %+15.7e | %+15.7e | %+15.7e | %s", m, fwd_val, rev_val, mirror_val, tag))
        end
    end

    fwd = nothing
    rev = nothing
    GC.gc()
end

println()
println("="^90)
println("Done")
println("="^90)
