# Architecture selection based on PBS_NGPUS (set by PBS scheduler).
# Sets: arch, arch_str, ngpus
#
# PBS_NGPUS > 1  → Distributed(GPU())  (multi-GPU via MPI)
# PBS_NGPUS == 1 → GPU()               (single GPU)
# PBS_NGPUS == 0 → CPU()               (no GPU)
#
# include()'d by setup_model.jl, create_velocities.jl, create_closures.jl.

ngpus = parse(Int, ENV["PBS_NGPUS"])
if ngpus > 1
    using CUDA
    using MPI
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = Distributed(GPU())
    arch_str = "DistributedGPU"
elseif ngpus == 1
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = GPU()
    arch_str = "GPU"
else
    arch = CPU()
    arch_str = "CPU"
end
@info "Using $arch_str architecture"
flush(stdout); flush(stderr)
