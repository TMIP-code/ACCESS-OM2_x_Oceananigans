# MWE: PartlyInMemory time-index OOB with PrescribedFreeSurface + z-star
#
# Bug: update_model_field_time_series! updates the η FTS for model time tⁿ,
# but step_free_surface! advances η's clock to tⁿ⁺¹ before _update_zstar_scaling!
# accesses it. When tⁿ is just before a snapshot boundary but tⁿ⁺¹ is just past
# it, the InMemory buffer hasn't been reloaded and memory_index returns OOB.
#
# The Cyclical wrapping at t=0 reloads the buffer to start=12 (holding indices
# 12,1,2,3). When the simulation later needs index 4 (at ~76 days), memory_index
# computes mod1(4-11, 12) = 5, which exceeds the 4-slot buffer.
#
# Run: julia --check-bounds=yes --project mwe_fts_z_indexing_bug.jl
# Or:  CUDA_VISIBLE_DEVICES="" julia --check-bounds=yes --project mwe_fts_z_indexing_bug.jl

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OutputReaders: Cyclical, InMemory, OnDisk
using Oceananigans.Units: seconds, days

year = 365.25days
month = year / 12

arch = CPU()

# ── Grid: RectilinearGrid with z-star ────────────────────────────────────
Nz = 4
z_faces = collect(range(-100, 0, length = Nz + 1))
grid = RectilinearGrid(arch;
    size = (4, 4, Nz),
    x = (0, 100),
    y = (0, 100),
    z = MutableVerticalDiscretization(z_faces),
    halo = (4, 4, 4),
    topology = (Periodic, Periodic, Bounded),
)

# ── Write η FieldTimeSeries to JLD2 ─────────────────────────────────────
prescribed_dt = 1month
fts_times = collect(((1:12) .- 0.5) * prescribed_dt)
stop_time = 12 * prescribed_dt

tmpdir = mktempdir()
η_file = joinpath(tmpdir, "η.jld2")

η_ondisk = FieldTimeSeries{Center, Center, Face}(grid, fts_times;
    indices = (:, :, Nz + 1),
    backend = OnDisk(), path = η_file, name = "η",
    time_indexing = Cyclical(stop_time))
for n in 1:12
    η_field = Field{Center, Center, Face}(grid; indices = (:, :, Nz + 1))
    set!(η_field, sin(2pi * n / 12))
    set!(η_ondisk, η_field, n)
end

# ── Reload with InMemory(4) ─────────────────────────────────────────────
η_ts = FieldTimeSeries(η_file, "η";
    architecture = arch, grid,
    backend = InMemory(4),
    time_indexing = Cyclical(year))

# ── Model with PrescribedFreeSurface + z-star ────────────────────────────
free_surface = PrescribedFreeSurface(displacement = η_ts)

model = HydrostaticFreeSurfaceModel(grid;
    tracers = :c,
    free_surface,
    velocities = PrescribedVelocityFields(; formulation = DiagnosticVerticalVelocity()),
)

set!(model, c = 1.0)

# ── Simulation: runs ~76 days before hitting the OOB ─────────────────────
simulation = Simulation(model; Δt = 5400seconds, stop_time)

@info "Running — expect BoundsError at ~76 days (when η buffer needs index 4)"
run!(simulation)

@info "No BoundsError — bug is fixed!"
