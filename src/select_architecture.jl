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

if nranks > 1
    using MPI
    MPI.Init()
    arch = Distributed(device, partition = Partition(px, py))
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    arch_str = "Distributed$(device_str)($(px)x$(py))"
    @info "MPI rank $rank/$nranks, partition=$(px)x$(py) ($device_str)"
    if device isa GPU
        @info "CUDA device: $(CUDA.device())"
    end
else
    arch = device
    arch_str = device_str
end

@info "Using $arch_str architecture"
flush(stdout); flush(stderr)
