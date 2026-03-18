# Architecture selection based on PBS_NGPUS (set by PBS scheduler).
# Sets: arch, arch_str, ngpus
#
# PBS_NGPUS > 1  → Distributed(GPU(), partition=Partition(px, py))  (multi-GPU via MPI)
# PBS_NGPUS == 1 → GPU()                                             (single GPU)
# PBS_NGPUS == 0 → CPU()                                             (no GPU)
#
# For multi-GPU runs, GPU_PARTITION_X and GPU_PARTITION_Y must be set in ENV
# (passed via qsub -v from driver.sh). These determine the domain decomposition.
#
# include()'d by setup_model.jl, create_velocities.jl, create_closures.jl.

ngpus = parse(Int, get(ENV, "PBS_NGPUS", "0"))
if ngpus > 1
    using CUDA
    using MPI
    MPI.Init()
    # CUDA runtime version is pinned in LocalPreferences.toml (version = "12.9", local = true).
    # Do NOT call CUDA.set_runtime_version!() here — it rewrites LocalPreferences.toml
    # and can clobber the [MPIPreferences] section.
    @show CUDA.versioninfo()
    px = parse(Int, ENV["GPU_PARTITION_X"])
    py = parse(Int, ENV["GPU_PARTITION_Y"])
    arch = Distributed(GPU(), partition = Partition(px, py))
    arch_str = "DistributedGPU($(px)x$(py))"
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    @info "MPI rank $rank/$nranks, partition=$(px)x$(py), CUDA device: $(CUDA.device())"
elseif ngpus == 1
    using CUDA
    @show CUDA.versioninfo()
    arch = GPU()
    arch_str = "GPU"
else
    arch = CPU()
    arch_str = "CPU"
end
@info "Using $arch_str architecture"
flush(stdout); flush(stderr)
