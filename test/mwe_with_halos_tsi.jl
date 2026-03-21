"""
MWE: `with_halos=true` in JLD2Writer doesn't preserve halos for TSI fields.

For a regular Field, `parent(field)` returns the full data buffer (Nx+2Hx, Ny+2Hy, Nz+2Hz).
For a TimeSeriesInterpolation field (from PrescribedVelocityFields), `construct_output`
wraps it in a new Field via `Field(tsi; indices, compute=false)`. The resulting `parent()`
returns only interior-sized data.

Run:
    julia --project test/mwe_with_halos_tsi.jl
"""

using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OutputWriters: construct_output, fetch_output
using Oceananigans.Units: seconds

Nx, Ny, Nz = 12, 13, 4
H = 3
grid = RectilinearGrid(CPU(); size = (Nx, Ny, Nz), x = (0, 1), y = (0, 1), z = (-1, 0), halo = (H, H, H))

expected_parent = (Nx + 2H, Ny + 2H, Nz + 2H)

# --- Test 1: Regular CenterField ---
c = CenterField(grid)
set!(c, (x, y, z) -> x + y + z)

output_c = construct_output(c, (:, :, :), true)  # with_halos=true
fetched_c = fetch_output(output_c, nothing)

println("=== Regular CenterField ===")
println("  interior size: ", size(interior(c)))
println("  parent size:   ", size(parent(c)))
println("  expected:      ", expected_parent)
println("  output parent: ", size(fetched_c))
println("  has halos:     ", size(fetched_c) == expected_parent)

# --- Test 2: FieldTimeSeries (TSI) ---
# Create a simple FTS and use it as prescribed velocity
times = [0.0, 1.0, 2.0]
fts = FieldTimeSeries{Face, Center, Center}(grid, times)
for n in eachindex(times)
    set!(fts[n], (x, y, z) -> x + y + z + n)
end

model = HydrostaticFreeSurfaceModel(
    grid;
    velocities = PrescribedVelocityFields(u = fts),
    tracers = (),
    buoyancy = nothing,
    free_surface = nothing,
)

u_tsi = model.velocities.u  # This is a TimeSeriesInterpolation field

output_u = construct_output(u_tsi, (:, :, :), true)  # with_halos=true
fetched_u = fetch_output(output_u, model)

expected_u_parent = (Nx + 2H, Ny + 2H, Nz + 2H)

println("\n=== TSI field (prescribed u) ===")
println("  typeof(u):     ", typeof(u_tsi))
println("  interior size: ", size(interior(u_tsi)))
println("  parent size:   ", size(parent(u_tsi)))
println("  output parent: ", size(fetched_u))
println("  expected:      ", expected_u_parent)
println("  has halos:     ", size(fetched_u) == expected_u_parent)

# --- Test 3: Diagnostic w ---
w = model.velocities.w

output_w = construct_output(w, (:, :, :), true)
fetched_w = fetch_output(output_w, model)

expected_w_parent = (Nx + 2H, Ny + 2H, Nz + 1 + 2H)

println("\n=== Diagnostic w ===")
println("  typeof(w):     ", typeof(w))
println("  interior size: ", size(interior(w)))
println("  parent size:   ", size(parent(w)))
println("  output parent: ", size(fetched_w))
println("  expected:      ", expected_w_parent)
println("  has halos:     ", size(fetched_w) == expected_w_parent)

# --- Summary ---
println("\n=== Summary ===")
all_pass = true
for (name, fetched, expected) in [
        ("CenterField", fetched_c, expected_parent),
        ("TSI u", fetched_u, expected_u_parent),
        ("Diagnostic w", fetched_w, expected_w_parent),
    ]
    ok = size(fetched) == expected
    global all_pass &= ok
    println("  $name: $(ok ? "PASS" : "FAIL") (got $(size(fetched)), expected $expected)")
end
println(all_pass ? "\nAll tests passed." : "\nSome tests FAILED — with_halos=true does not preserve halos for all field types.")
