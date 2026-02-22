using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: zspacings
using JLD2

include("tripolargrid_reader.jl")

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENTMODEL")
    ENV["PARENTMODEL"]
else
    "ACCESS-OM2-1"
end

outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")

grid = load_tripolar_grid(grid_file, CPU())

Δz_ib = Field(zspacings(grid, Center(), Center(), Center()))
compute!(Δz_ib)

Δz_underlying = Field(zspacings(grid.underlying_grid, Center(), Center(), Center()))
compute!(Δz_underlying)

a = Array(interior(Δz_ib))
b = Array(interior(Δz_underlying))

diff = abs.(a .- b)

println("parentmodel = ", parentmodel)
println("grid_file = ", grid_file)
println("min(Δz_ib) = ", minimum(a))
println("min(Δz_underlying) = ", minimum(b))
println("maxabs(Δz_ib - Δz_underlying) = ", maximum(diff))
println("any(Δz_ib != Δz_underlying) = ", any(diff .> 0))
