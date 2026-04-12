"""
MWE: `show(CenterField)` on a 2x2 distributed tripolar grid crashes on main.

`show` calls `data_summary` → `maximum` → creates a reduced Field(Nothing,Nothing,Nothing)
→ tripolar Field constructor injects DistributedZipper north BC → `has_fold_line(TY, Nothing)`
→ MethodError.

    mpiexec -n 4 julia --project test/test_reduced_field_tripolar.jl
"""

using Oceananigans
using Oceananigans.DistributedComputations: Distributed, Partition
using MPI

MPI.Init()

arch = Distributed(CPU(), partition = Partition(2, 2))
grid = TripolarGrid(arch; size = (20, 20, 5), z = (-500, 0))

field = CenterField(grid)
@show field
