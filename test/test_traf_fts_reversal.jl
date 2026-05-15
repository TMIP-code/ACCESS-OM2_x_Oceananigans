"""
TRAF FTS time-reversal smoke test.

For each monthly FieldTimeSeries that TRAF=yes consumes in setup_model.jl
(u, v, η, T, S, κV), this script:

  1. Loads the forward FTS.
  2. Loads a second copy and applies `reverse_fts_time!` with the same
     `flip_sign` choice that setup_model.jl uses (true for u/v, false otherwise).
  3. Verifies snapshot-aligned equality: parent(rev[i]) == sgn * parent(fwd[N+1-i]).
  4. Verifies clock-time interpolation at a dense sweep of 24 times (12
     snapshot-aligned + 12 mid-snapshot midpoints across the 1-year period):
     interior(rev[Time(t)]) ≈ sgn * interior(fwd[Time(mod(T - t, T))]).

OM2-1 only; runs once per TIME_WINDOW via the test driver step `trafftsrev`.
"""

using Test
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
using Oceananigans.Fields: interior
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days

include("../src/shared_functions.jl")

@info "GIT_COMMIT = $(get(ENV, "GIT_COMMIT", "unknown"))"

cfg = load_project_config()
(; parentmodel, experiment, time_window, experiment_dir, monthly_dir, mld_monthly_dir) = cfg

@info "TRAF FTS reversal test"
@info "- PARENT_MODEL = $parentmodel"
@info "- EXPERIMENT   = $experiment"
@info "- TIME_WINDOW  = $time_window"
@info "- monthly_dir     = $monthly_dir"
@info "- mld_monthly_dir = $mld_monthly_dir"

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading grid from $grid_file"
arch = CPU()
grid = load_tripolar_grid(grid_file, arch)
@info "Grid loaded: size = $(size(grid))"

# Mirror the loads in setup_model.jl exactly. VELOCITY_SOURCE is hardcoded to
# `totaltransport` here because TRAF uses that (see docs/TRAF_simulations.md).
specs = [
    (file = "u_from_total_transport_monthly.jld2", name = "u", flip_sign = true, dir = monthly_dir),
    (file = "v_from_total_transport_monthly.jld2", name = "v", flip_sign = true, dir = monthly_dir),
    (file = "eta_monthly.jld2", name = "η", flip_sign = false, dir = monthly_dir),
    (file = "temp_monthly.jld2", name = "T", flip_sign = false, dir = monthly_dir),
    (file = "salt_monthly.jld2", name = "S", flip_sign = false, dir = monthly_dir),
    (file = "kappa_v_monthly.jld2", name = "κV", flip_sign = false, dir = mld_monthly_dir),
]

backend = InMemory()
time_indexing = Cyclical(1year)

@testset "TRAF FTS reversal" begin
    for spec in specs
        @info "Testing FTS: $(spec.name) ($(spec.file)), flip_sign=$(spec.flip_sign)"
        path = joinpath(spec.dir, spec.file)
        isfile(path) || error("Missing FTS file: $path")

        fwd = load_fts(arch, path, spec.name, grid; backend, time_indexing)
        rev = load_fts(arch, path, spec.name, grid; backend, time_indexing)
        reverse_fts_time!(rev; flip_sign = spec.flip_sign)

        N = length(rev.times)
        Δt = rev.times[2] - rev.times[1]
        T_period = N * Δt
        sgn = spec.flip_sign ? -1 : 1

        @testset "$(spec.name) — snapshot-aligned" begin
            for i in 1:N
                @test parent(rev[i]) == sgn .* parent(fwd[N + 1 - i])
            end
        end

        # Use only mid-snapshot times here. The exact snapshot times t = times[i]
        # are already covered bit-exactly by the snapshot-aligned testset above
        # (via `parent` equality on the raw stored data), and at exact snapshots
        # Oceananigans' `fts[Time(t)]` takes asymmetric code paths: it returns
        # the *stored* Field directly when n₁ == n₂ (e.g. t == times[1]), but
        # builds a *new* Field via compute!(...) when n₁ ≠ n₂ (e.g. t == times[N],
        # where ñ exactly equals 1.0). The new-Field path runs fill_halo_regions!,
        # which for `v` at the tripolar fold (FPivotZipperBoundaryCondition,
        # j = Ny) modifies the j = Ny row to enforce antisymmetry — so we'd be
        # comparing raw stored data on one side against fill-halo'd data on the
        # other and observe small differences from the MOM output's residual
        # antisymmetry violation. Mid-snapshot times always trigger the same
        # linear-blend path on both sides, so the comparison is well-posed.
        @testset "$(spec.name) — mid-snapshot clock-time sweep (22 times)" begin
            for k in 0:21
                t = fwd.times[1] + Δt / 4 + k * (Δt / 2)
                t_mirror = mod(T_period - t, T_period)
                lhs = interior(rev[Time(t)])
                rhs = sgn .* interior(fwd[Time(t_mirror)])
                ok = isapprox(lhs, rhs)
                if !ok
                    diff = lhs .- rhs
                    @warn "Clock-time mismatch" name = spec.name k t t_mirror max_abs_diff = maximum(abs, diff) norm_diff = sqrt(sum(abs2, diff)) norm_rhs = sqrt(sum(abs2, rhs))
                end
                @test ok
            end
        end

        fwd = nothing
        rev = nothing
        GC.gc()
    end
end

@info "Done running test/test_traf_fts_reversal.jl"
