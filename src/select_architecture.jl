# Architecture selection: GPU vs CPU × serial vs distributed.
# Sets: arch, arch_str
#
# Two independent axes:
#   1. Device: PBS_NGPUS ≥ 1 → GPU(), else CPU()
#   2. Distribution: PARTITION_X × PARTITION_Y > 1 → Distributed(device, partition)
#
# include()'d by setup_model.jl, create_velocities.jl, create_closures.jl.

# 1. Device selection
ngpus = parse(Int, get(ENV, "PBS_NGPUS", "0"))
if ngpus >= 1
    using CUDA
    # CUDA runtime version is pinned in LocalPreferences.toml (version = "12.9", local = true).
    # Do NOT call CUDA.set_runtime_version!() here — it rewrites LocalPreferences.toml
    # and can clobber the [MPIPreferences] section.
    @show CUDA.versioninfo()
    device = GPU()
    device_str = "GPU"
else
    device = CPU()
    device_str = "CPU"
end

# 2. Distribution selection
px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
nranks = px * py

# Pull in just what we need here. shared_functions.jl is normally
# included AFTER this file (in setup_model.jl), so we eagerly include the
# two small helpers; downstream re-includes are no-ops.
include("shared_utils/config.jl")
include("shared_utils/load_balance.jl")

(LB_ACTIVE, LB_METHOD, LB_TAG) = parse_load_balance_env()

if nranks > 1
    using MPI
    using Oceananigans.DistributedComputations: Sizes
    MPI.Init()

    if LB_ACTIVE
        # Wet-load-balanced y-partition. Reads `bottom` (and `z_faces` for
        # cell-based) from the shared grid file on every rank —
        # deterministic, so all ranks agree on the per-rank Ny without any
        # MPI communication.
        px == 1 || error("LOAD_BALANCE=$LB_METHOD only supports PARTITION_X=1 (got px=$px). The greedy y-slab algorithm partitions along y only.")
        _cfg_for_lb = load_project_config()
        _grid_file_for_lb = joinpath(_cfg_for_lb.experiment_dir, "grid.jld2")
        _Hy_for_lb = load(_grid_file_for_lb, "Hy")
        local_Ny_lb = compute_lb_y_sizes(
            _grid_file_for_lb, py;
            method = LB_METHOD, min_size = _Hy_for_lb + 2,
        )
        arch = Distributed(device, partition = Partition(y = Sizes(local_Ny_lb...)))
        arch_str = "Distributed$(device_str)($(px)x$(py)$(LB_TAG), local_Ny=$local_Ny_lb)"
    else
        arch = Distributed(device, partition = Partition(px, py))
        arch_str = "Distributed$(device_str)($(px)x$(py))"
    end

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @info "MPI rank $rank/$nranks, $arch_str"
    if device isa GPU
        @info "CUDA device: $(CUDA.device())"
    end
else
    arch = device
    arch_str = device_str
end

@info "Using $arch_str architecture"
flush(stdout); flush(stderr)
