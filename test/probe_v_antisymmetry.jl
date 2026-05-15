"""
Diagnostic: examine v[i, Ny, k] antisymmetry across the tripolar fold.

Loads month 1 of the forward v FTS, then:
  1. Reports parent array shape and indexes used.
  2. For interior cells at j = Ny (the fold row), checks
     v[i, Ny, k] + v[Nx+1-i, Ny, k] against zero for all (i, k).
  3. Calls `fill_halo_regions!` on a copy of v[1] and re-checks.
  4. Reports max-abs diff between original and post-fill_halo.
  5. Spot-checks a single column.
"""

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days

include("../src/shared_functions.jl")

cfg = load_project_config()
(; experiment_dir, monthly_dir) = cfg

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
Nx, Ny, Nz = size(grid)
Hx, Hy, Hz = (grid.Hx, grid.Hy, grid.Hz)
@info "Grid" Nx Ny Nz Hx Hy Hz

v_path = joinpath(monthly_dir, "v_from_total_transport_monthly.jld2")
v_ts = load_fts(CPU(), v_path, "v", grid; backend = InMemory(), time_indexing = Cyclical(1year))
@info "v boundary_conditions" v_ts.boundary_conditions

v1 = v_ts[1]
p = parent(v1)
@info "parent(v1) size" size_p = size(p) axes_p = axes(p)

# Helper to read interior at (i, j, k) with i,j,k in 1..Nx, 1..Ny, 1..Nz
# Interior in parent indexing: parent[i + Hx, j + Hy, k + Hz]
@inline pinterior(p, i, j, k) = p[i + Hx, j + Hy, k + Hz]

# Antisymmetry at j = Ny (the fold row)
# Expected: v[i, Ny, k] = -v[Nx + 1 - i, Ny, k]  for all i, k
# => v[i, Ny, k] + v[Nx + 1 - i, Ny, k] = 0
function antisymmetry_residual(p)
    max_res = 0.0
    sum_res = 0.0
    n_nonzero = 0
    for k in 1:Nz
        for i in 1:Nx
            i_pair = Nx + 1 - i
            a = pinterior(p, i, Ny, k)
            b = pinterior(p, i_pair, Ny, k)
            r = a + b
            ar = abs(r)
            if ar > 1.0e-14
                n_nonzero += 1
            end
            max_res = max(max_res, ar)
            sum_res += r^2
        end
    end
    return (; max_res, l2_res = sqrt(sum_res), n_nonzero, total = Nx * Nz)
end

before = antisymmetry_residual(p)
@info "Antisymmetry residual at j=Ny BEFORE fill_halo_regions!" before

# Spot-check a few cells
@info "Spot check (i, Nx+1-i, k=25)" v1_Ny_3 = pinterior(p, 3, Ny, 25) v1_Ny_Nm2 = pinterior(p, Nx + 1 - 3, Ny, 25) sum = pinterior(p, 3, Ny, 25) + pinterior(p, Nx + 1 - 3, Ny, 25)

# Make a copy of v1 and apply fill_halo
v1_copy = Field((Center(), Face(), Center()), grid; boundary_conditions = v_ts.boundary_conditions)
parent(v1_copy) .= parent(v1)
fill_halo_regions!(v1_copy)
p_after = parent(v1_copy)

after = antisymmetry_residual(p_after)
@info "Antisymmetry residual at j=Ny AFTER fill_halo_regions!" after

# Compute the diff between before and after fill_halo, at j=Ny
diff_at_fold = 0.0
n_changed = 0
for k in 1:Nz
    for i in 1:Nx
        d = abs(pinterior(p_after, i, Ny, k) - pinterior(p, i, Ny, k))
        if d > 1.0e-14
            n_changed += 1
        end
        diff_at_fold = max(diff_at_fold, d)
    end
end
@info "Interior j=Ny modified by fill_halo_regions!" max_change = diff_at_fold n_changed total = Nx * Nz

# Compare halo cells too — are the on-disk halos consistent with what fill_halo computes?
# Halo at j = Ny+1 (just above the fold): parent[i + Hx, Ny + Hy + 1, k + Hz]
@inline phalo_north(p, i, h, k) = p[i + Hx, Ny + Hy + h, k + Hz]

halo_max_change = 0.0
halo_n_changed = 0
for h in 1:Hy
    for k in 1:Nz
        for i in 1:Nx
            d = abs(phalo_north(p_after, i, h, k) - phalo_north(p, i, h, k))
            if d > 1.0e-14
                halo_n_changed += 1
            end
            halo_max_change = max(halo_max_change, d)
        end
    end
end
@info "North halos modified by fill_halo_regions!" halo_max_change halo_n_changed total = Hy * Nx * Nz

@info "Done"
