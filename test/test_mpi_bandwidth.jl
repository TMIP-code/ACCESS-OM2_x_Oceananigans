"""
GPU↔GPU MPI ping-pong bandwidth probe (intra-node vs inter-node).

The point of this test: diagnose whether CUDA-aware MPI is actually using
GPUDirect RDMA across the InfiniBand fabric, or whether MPI is silently
host-staging GPU buffers (the textbook fingerprint of which is "intra-node
fast, inter-node much slower than the IB peak"). See:
  - JuliaGPU/CUDA.jl#1053 (stream-ordered allocator → cuIpc fallback)
  - JuliaParallel/MPI.jl docs on `MPI.has_cuda()`
  - NVIDIA UCX multi-node tuning guide (UCX_TLS must include cuda_copy)

What this script does, on `nranks` ranks across `nnodes` nodes:
  1. Print rank, hostname, device, MPI/CUDA-aware flags.
  2. Pick two pairs:
       - intra-node pair: rank 0 + the next rank on the same host as rank 0
       - inter-node pair: rank 0 + the first rank on a *different* host
  3. For each pair and for buffers in {CPU, GPU}, ping-pong messages of
     several sizes (1 KB → 256 MB), report one-way latency and bandwidth.

Decision rule (V100 + Gadi IB, nominal 25 GB/s):
  - Intra-node GPU peak should be ≥ ~30 GB/s (NVLink).
  - Inter-node GPU peak with GPUDirect should be ≥ ~12 GB/s.
  - Inter-node GPU peak around ~5-8 GB/s ⇒ host-staging fallback ⇒ this is
    the bug, configure UCX/PML appropriately.

Run (8 ranks across 2 V100 nodes):
    mpiexec --bind-to socket --map-by socket -n 8 julia --project test/test_mpi_bandwidth.jl

CPU-only run is also valid (for sanity / comparison):
    mpiexec -n 8 julia --project test/test_mpi_bandwidth.jl
"""

using MPI
using Printf
using Statistics

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

# --- Architecture detection ---
const NGPUS_PBS = parse(Int, get(ENV, "PBS_NGPUS", "0"))
const USE_GPU = NGPUS_PBS > 0
if USE_GPU
    using CUDA
    # Local-rank → device assignment. OpenMPI sets OMPI_COMM_WORLD_LOCAL_RANK.
    # Falls back to RANK % ndevices() if the env var isn't set.
    local_rank = parse(
        Int, get(
            ENV, "OMPI_COMM_WORLD_LOCAL_RANK",
            get(ENV, "MPI_LOCALRANKID", string(RANK))
        )
    )
    ndev = CUDA.ndevices()
    CUDA.device!(local_rank % ndev)
end

# --- Banner ---
host = gethostname()
if RANK == 0
    println("="^72)
    println("MPI bandwidth probe — $NRANKS ranks")
    println("  MPI.has_cuda() = $(MPI.has_cuda())")
    println("  USE_GPU        = $USE_GPU")
    if USE_GPU
        println("  CUDA runtime   = $(CUDA.runtime_version())")
        println("  CUDA driver    = $(CUDA.driver_version())")
    end
    println("="^72)
end
flush(stdout)
MPI.Barrier(COMM)

# --- Per-rank report ---
dev_str = USE_GPU ? "GPU $(CUDA.device())" : "CPU"
@info "rank=$RANK host=$host device=$dev_str"
flush(stdout); flush(stderr)
MPI.Barrier(COMM)

# --- Gather hostnames so rank 0 can pick intra/inter-node peers ---
all_hosts = MPI.gather(host, COMM; root = 0)

intra_peer = -1     # peer for rank 0, same host
inter_peer = -1     # peer for rank 0, different host

if RANK == 0
    for r in 1:(NRANKS - 1)
        if all_hosts[r + 1] == all_hosts[1] && intra_peer < 0
            intra_peer = r
        end
        if all_hosts[r + 1] != all_hosts[1] && inter_peer < 0
            inter_peer = r
        end
    end
    println()
    println("Hostnames per rank:")
    for r in 0:(NRANKS - 1)
        println("  rank $r → $(all_hosts[r + 1])")
    end
    println()
    println("Selected pairs (rank 0 ↔ peer):")
    println("  intra-node peer = $intra_peer")
    println("  inter-node peer = $inter_peer")
    println()
end

intra_peer = MPI.bcast(intra_peer, COMM; root = 0)
inter_peer = MPI.bcast(inter_peer, COMM; root = 0)
flush(stdout)
MPI.Barrier(COMM)

# --- Ping-pong primitive ---
"""
    pingpong!(buf, peer; nrep, nwarm) → (rtt_seconds, bw_GBps_oneway)

Standard MPI ping-pong: rank A sends to B, B sends back, repeated.
Only the two participating ranks (RANK == 0 or RANK == peer) do work;
all other ranks just barrier.

Returns one-way bandwidth (msg_bytes / (rtt/2)) in GB/s.
`buf` is reused for both directions; sizeof(buf) defines the message size.
"""
function pingpong!(buf, peer::Integer; nrep::Int = 50, nwarm::Int = 5)
    msg_bytes = sizeof(buf)
    is_a = (RANK == 0)
    is_b = (RANK == peer)
    if !(is_a || is_b)
        MPI.Barrier(COMM)
        return (NaN, NaN)
    end

    # Warmup
    for _ in 1:nwarm
        if is_a
            MPI.Send(buf, COMM; dest = peer, tag = 0)
            MPI.Recv!(buf, COMM; source = peer, tag = 1)
        else
            MPI.Recv!(buf, COMM; source = 0, tag = 0)
            MPI.Send(buf, COMM; dest = 0, tag = 1)
        end
    end
    if buf isa AbstractArray && (USE_GPU ? buf isa CuArray : false)
        CUDA.synchronize()
    end
    MPI.Barrier(COMM)

    t0 = MPI.Wtime()
    for _ in 1:nrep
        if is_a
            MPI.Send(buf, COMM; dest = peer, tag = 0)
            MPI.Recv!(buf, COMM; source = peer, tag = 1)
        else
            MPI.Recv!(buf, COMM; source = 0, tag = 0)
            MPI.Send(buf, COMM; dest = 0, tag = 1)
        end
    end
    if USE_GPU && buf isa CuArray
        CUDA.synchronize()
    end
    t1 = MPI.Wtime()

    rtt = (t1 - t0) / nrep
    bw = msg_bytes / (rtt / 2) / 1.0e9   # GB/s, one-way
    return (rtt, bw)
end

# --- Sweep sizes ---
# nbytes per message; use Float64 elements so bytes = 8 * n.
const SIZES_BYTES = [
    1 << 10,   #   1 KiB
    1 << 13,   #   8 KiB
    1 << 16,   #  64 KiB
    1 << 19,   # 512 KiB
    1 << 22,   #   4 MiB
    1 << 24,   #  16 MiB
    1 << 26,   #  64 MiB
    1 << 28,   # 256 MiB
]

function pretty_bytes(b::Integer)
    return if b >= 1 << 20
        @sprintf("%.0f MiB", b / (1 << 20))
    elseif b >= 1 << 10
        @sprintf("%.0f KiB", b / (1 << 10))
    else
        @sprintf("%d B", b)
    end
end

function run_sweep(label::String, peer::Integer, on_gpu::Bool)
    if peer < 0
        if RANK == 0
            println("→ $label: SKIP (no peer of this kind in this job layout)")
            println()
        end
        return
    end
    if RANK == 0
        println("→ $label  (rank 0 ↔ rank $peer, $(on_gpu ? "GPU" : "CPU") buffers)")
        @printf("    %-10s %-12s %-14s\n", "size", "lat (μs)", "bw (GB/s, 1-way)")
    end
    flush(stdout)

    for nbytes in SIZES_BYTES
        n = nbytes ÷ 8                 # Float64 elements
        # Lower nrep for big messages to keep wallclock bounded.
        nrep = nbytes < (1 << 22) ? 200 :
            nbytes < (1 << 26) ? 50 : 20
        nwarm = 5

        if on_gpu
            buf = CUDA.zeros(Float64, n)
        else
            buf = zeros(Float64, n)
        end

        rtt, bw = pingpong!(buf, peer; nrep = nrep, nwarm = nwarm)

        if RANK == 0
            @printf(
                "    %-10s %-12.2f %-14.3f\n",
                pretty_bytes(nbytes), rtt * 5.0e5, bw
            )
        end
        flush(stdout)
        MPI.Barrier(COMM)
    end
    return if RANK == 0
        println()
    end
end

if RANK == 0
    println("Note: 'lat' is one-way (rtt/2) in microseconds; 'bw' is one-way GB/s.")
    println()
end

# CPU sweeps first (always available)
run_sweep("CPU buffers, INTRA-node", intra_peer, false)
run_sweep("CPU buffers, INTER-node", inter_peer, false)

# GPU sweeps if applicable
if USE_GPU
    run_sweep("GPU buffers, INTRA-node", intra_peer, true)
    run_sweep("GPU buffers, INTER-node", inter_peer, true)
end

if RANK == 0
    println("="^72)
    println("Diagnostic guide:")
    println("  - V100 + Gadi IB nominal: intra-node GPU ≥ ~30 GB/s (NVLink),")
    println("    inter-node GPU ≥ ~12 GB/s with GPUDirect, ~5-8 GB/s host-staged.")
    println("  - If MPI.has_cuda() is false ⇒ MPI was built without CUDA support.")
    println("  - If GPU INTER ≈ CPU INTER ⇒ likely host-staging fallback;")
    println("    try UCX_TLS=cuda_copy,cuda_ipc,rc,sm,self and OMPI_MCA_pml=ucx.")
    println("="^72)
end

MPI.Barrier(COMM)
MPI.Finalize()
