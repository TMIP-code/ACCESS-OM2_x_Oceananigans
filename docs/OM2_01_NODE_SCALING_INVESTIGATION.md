# Investigation: OM2-01 1×8 doesn't scale beyond 1×4 (H200, gpuhopper)

**Status:** drafted at end of OM2-01 bring-up session, intended to be
picked up in a fresh session. Author note: every fact below is taken
from the May 2026 1958-1987 cycle4 runs documented in
[BENCHMARKS.md § ACCESS-OM2-01](../BENCHMARKS.md). Verify before acting.

## The observation

| Partition | GPU layout              | 1-yr integration | Setup | Total wall | Job ID    |
|-----------|-------------------------|------------------|-------|------------|-----------|
| 1×4       | 1 node, 4× H200         | **1.670 h**      | ~14 m | 1:54:33    | 167345793 |
| 1×8       | 2 nodes, 8× H200        | **1.670 h**      | ~20 m | 2:00:37    | 167345796 |

Going from 4 → 8 H200s (across a node boundary) **does not** reduce
the model integration time at all. Setup (FTS load) cost grows by
~6 min — also not free. SU pressure roughly doubles for no walltime
benefit.

Profile jobs **167527948** (1×4) and **167527949** (1×8) were
submitted at the very end of the session with `BENCHMARK_STEPS=240`
and `PROFILE=yes`; their `.qdrep`/`.nsys-rep` outputs land under
`outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1958-1987/standardrun/.../profile_*/`
once they finish. **First action of the next session: open both
profiles in Nsight Systems and compare per-step phase timings
(compute / halo / GC / output).**

## Existing references already saved in the repo

These are the curated MPI/GPU/scaling references already in the docs.
Read these first — they were collected during earlier OM2-1/-025
strong-scaling work and most are directly applicable.

### Binding & launcher

- [docs/MPI_LAUNCHER_AND_BINDING.md](MPI_LAUNCHER_AND_BINDING.md) — already
  documents `--bind-to socket --map-by socket` (the OM2-1 V100 2×2 fix
  that gave a 7× speedup), `mpiexec`/`mpirun` equivalence, PBS-cpuset
  binding behaviour.
  - [open-mpi/ompi#9647](https://github.com/open-mpi/ompi/issues/9647)
    — pthread/cpuset interactions.
  - [open-mpi/ompi#11541](https://github.com/open-mpi/ompi/issues/11541)
    — binding regressions across versions.
  - NCI: [Hybrid MPI and OpenMP](https://opus.nci.org.au/spaces/Help/pages/122552392/Hybrid+MPI+and+OpenMP)
  - NCI: [gpuhopper queue topology](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi)

### Distributed GC

- [docs/DISTRIBUTED_GC.md](DISTRIBUTED_GC.md), [docs/SYNC_GC.md](SYNC_GC.md)
  — collective GC cost dominated H200 OM2-025 1×2 runs by hundreds of
  seconds. Already wired up in
  [src/run_1year_benchmark.jl](../src/run_1year_benchmark.jl) via
  `SYNC_GC_NSTEPS`.
  - ClimaAtmos callback path:
    [ClimaAtmos/src/callbacks/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/callbacks.jl)
  - Solver path:
    [ClimaAtmos/src/solver/solve.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/solver/solve.jl)

### Combined strong-scaling tricks (sync-GC + temporal blocking + load balancing)

- [docs/COMBINED_SCALING_BENCHMARKS.md](COMBINED_SCALING_BENCHMARKS.md)
  — covers OM2-025 1×2 H200 with all three tricks stacked.
  - Land-aware load balancing technique:
    [CliMA/ClimaOcean.jl#665 discussion](https://github.com/CliMA/ClimaOcean.jl/discussions/665#discussioncomment-14737556)
  - [docs/TBLOCKING_BENCHMARKS.md](TBLOCKING_BENCHMARKS.md) cites the
    canonical temporal-blocking literature (Malas 2015, Wellein 2009,
    Wonnacott 2000).

### Profiling tooling

- [docs/PROFILING.md](PROFILING.md), [docs/PROFILING_RESULTS.md](PROFILING_RESULTS.md)
  — Nsight Systems pattern for Julia + CUDA + MPI.
  - [Nsight Systems](https://developer.nvidia.com/nsight-systems)
  - [ETH Julia + CUDA + MPI course (course-101-0250-00)](https://github.com/eth-vaw-glaciology/course-101-0250-00)

### Bugs already filed

- [CliMA/Oceananigans.jl#5410](https://github.com/CliMA/Oceananigans.jl/issues/5410)
  — JLD2OutputWriter deadlock on distributed GPU grids (worked around
  by `including=[]`). Probably unrelated to the current question, but
  worth noting since it shows distributed-grid quirks have been hit
  before.

**Nothing new from this OM2-01 session needs adding to the references
above.** The intake-catalog bug filed at
[ACCESS-NRI/access-nri-intake-catalog#603](https://github.com/ACCESS-NRI/access-nri-intake-catalog/issues/603)
is preprocessing-only, not MPI/GPU.

## Hypotheses worth testing (in priority order)

### H1 — Workload is memory-bandwidth bound on a single H200 node, so a second node doesn't help

A single hopper node has 4× H200 with NVLink between them, plus aggregate
~12 TB/s GPU-memory bandwidth. If OM2-01 at 0.1° is already
bandwidth-saturated on 1 node, adding a second node gives more compute
but the per-step kernel time stays ~the same. The integration time
matching exactly between 1×4 and 1×8 (1.670 h both) is consistent with
this — no speedup, no slowdown.

**Test:**
- Look at `nsys` profile for kernel occupancy / DRAM throughput. If
  >80 % HBM3 utilization on 1×4, H1 is supported.
- Cross-check by running 2×2 on 1 node (4 H200, but partitioned 2×2
  rather than 1×4) — if 2×2 has different per-step time vs 1×4 on the
  same 4 GPUs, the bottleneck is partition shape, not memory bandwidth.

### H2 — Inter-node halo exchange is dominant once we cross the IB boundary

NVLink intra-node ≈ 900 GB/s; InfiniBand on Gadi ≈ 25 GB/s. A 4× drop
in halo-exchange bandwidth could absorb the per-step compute saving
from going 4 → 8 GPUs.

**Test:**
- In the nsys profile, isolate `MPI_Sendrecv` / `MPI_Isend` / halo-fill
  kernels and compare wall-time per step on 1×4 vs 1×8.
- If halo time is 4-8× larger on 1×8, H2 is supported.
- If GPUDirect RDMA isn't enabled, this gets dramatically worse — see
  [docs/MPI_LAUNCHER_AND_BINDING.md](MPI_LAUNCHER_AND_BINDING.md) for
  how the launcher is currently configured. Cross-check via
  `OMPI_MCA_pml_base_verbose=10` and `UCX_LOG_LEVEL=info` in a small
  test run.

### H3 — One of the 8 ranks is a slow straggler (load imbalance amplified at 1×8)

The Y-axis tripolar grid is heavily land-biased near the poles (the
"zipper" row). With 8 partitions instead of 4, more ranks contain
fold/land slabs, and the slowest rank dictates collective cost.

**Test:**
- Apply the land-mask-aware load balancing already documented in
  [COMBINED_SCALING_BENCHMARKS.md](COMBINED_SCALING_BENCHMARKS.md)
  (`LOAD_BALANCE=cell` or `surface`) and re-run 1×8.
- If `LOAD_BALANCE=cell` brings 1×8 closer to a real 2× speedup over
  1×4, H3 is the dominant cost.

### H4 — OpenMPI 5.0.8 / CUDA-aware MPI sync issue at 2 nodes

OpenMPI 5.0.x had a series of inter-node CUDA-aware regressions
(see ompi#11541 and friends). UCX vs OB1 PML, Mellanox CX-6
vs CX-7 fabric quirks, etc.

**Test:**
- Run a small NCCL or MPI ping-pong test across the 2 nodes the job
  landed on (gadi-gpu-h200-0024 + 0025 — see comment lines in PBS
  resource info). Any anomaly there explains the inter-node bottleneck
  immediately.
- Try `mpitrampoline` vs system OpenMPI directly — see
  [scripts/env_defaults.sh](../scripts/env_defaults.sh) for the
  current `MPITRAMPOLINE_LIB` wrapping.

### H5 — Workload is genuinely too small at 1×8

3600×2700/8 = ~1.2 M cells/rank. Below ~1 M cells/rank, distributed
overhead can outweigh per-rank compute savings. Check what
[CliMA/ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) reports
as the smallest useful per-rank slab on H200 — they have benchmarks at
similar resolutions.

## Suggested concrete next-session plan

1. **Read those existing docs first** (15 min). Single-pass skim of
   `MPI_LAUNCHER_AND_BINDING.md`, `DISTRIBUTED_GC.md`, `SYNC_GC.md`,
   `COMBINED_SCALING_BENCHMARKS.md`, `PROFILING.md`. These already
   capture the playbook for distributed Oceananigans on Gadi.
2. **Open the profile jobs** (`167527948`/`167527949`) in Nsight
   Systems. Compare per-step timeline. Bucket the time into:
   `compute kernels`, `MPI halo fill`, `GC pauses`, `output writers`,
   `other`. Numbers go into `BENCHMARKS.md` under OM2-01.
3. **Test H1 vs H2 first** since they're cheap to disambiguate from
   the existing profile data. No new submissions needed.
4. If H3 looks plausible, submit a `LOAD_BALANCE=cell` 1×8 run via the
   driver (it's already wired). Cheap test, high upside.
5. If H4 looks plausible, build a 2-node MPI ping-pong test outside
   Oceananigans (e.g. via `mpibench` or a tiny `MPI.Sendrecv`
   benchmark on H200 GPU buffers). This is the most labour-intensive
   path — defer unless H1/H2/H3 ruled out.
6. **Search externally only after exhausting in-repo refs.** Likely
   hits: GitHub issues on `CliMA/Oceananigans.jl`, `JuliaParallel/MPI.jl`,
   `JuliaGPU/CUDA.jl` for terms like `H200 multi-node`, `gpuhopper
   strong scaling`, `NVLink + IB strong scaling`.

## What "saved" means in this doc

This file is a planning artifact, not a results doc. When the
investigation finishes, fold the verified findings into:

- `BENCHMARKS.md` (numbers, scaling tables, ✓/✗ verdict)
- `MPI_LAUNCHER_AND_BINDING.md` (any binding/launcher recipe changes)
- `DISTRIBUTED_GC.md` / `SYNC_GC.md` (if GC turns out to be the issue)
- A new `docs/OM2_01_SCALING_RESULTS.md` if neither bucket fits

Then this file can be deleted.
