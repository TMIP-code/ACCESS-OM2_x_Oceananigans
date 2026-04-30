# MPI launcher and rank binding on Gadi gpuhopper

Notes on what `mpiexec`/`mpirun` actually do here, why
`--bind-to socket --map-by socket` is non-deterministic on partial-node
PBS allocations, and what we do to diagnose it.

## `mpiexec` vs `mpirun`

In **Open MPI 5.x** (the only MPI we use, via the `openmpi/5.0.8` module
and MPItrampoline), `mpiexec`, `mpirun`, and `oprun` are all symlinks to
the same launcher (`prterun` in 5.x). The man page is shared. Same
flags, same mapping/binding behaviour.

The differences are conventional, not behavioural:

- **`mpiexec`** is the MPI-standard-defined name (MPI-2 onwards) → most
  portable across implementations. We use this in our scripts.
- **`mpirun`** is Open MPI's traditional name, also accepted by
  MPICH/Intel MPI, but each vendor's `mpirun` may take *different* flags.
- Some sites alias one or the other to a PBS/Slurm wrapper. NCI Gadi
  does **not** alias these for `openmpi/5.0.8` — both invoke `prterun`.

Edge cases where the choice matters:

- Cluster wrappers/aliases (not the case on Gadi for this module).
- Mixing MPI implementations on `LD_LIBRARY_PATH` and accidentally
  hitting one vendor's `mpirun` while linked against another vendor's
  `libmpi`. We pin via MPItrampoline + `MPITRAMPOLINE_LIB`, so this is
  contained.

**Conclusion.** `mpiexec` and `mpirun` are interchangeable in this repo;
we use `mpiexec` for portability.

## Binding on gpuhopper: PBS cpuset shape drives the result

Gadi gpuhopper hardware:

- Per node: 2× Intel Xeon Gold 6542Y (24 cores each) → 48 cores, **2
  sockets**, **4 NUMA nodes** (2 NUMA × 2 sockets, 12 cores each).
- 4× NVIDIA H200-SXM5-141GB GPUs per node, 1 TiB RAM per node.

Our 1×2 / 1×4 partial-node PBS request shape:

- `select=1:ngpus=N:ncpus=12N:mem=256GB×N` (= ¼ to ½ a node).

What we observed across multiple jobs with
`mpiexec --bind-to socket --map-by socket -n 2 --report-bindings`:

| `exec_host` slot pattern | Bindings printed |
|--------------------------|------------------|
| `gadi-gpu-h200-NNNN/1*24` | rank 0 *and* rank 1 → `package[0][core:0-23]` (both on socket 0, 24-core range) |
| `gadi-gpu-h200-NNNN/2*24` | rank 0 → `package[0][core:12-23]`, rank 1 → `package[1][core:36-47]` (cross-socket, 12-core / NUMA-tight) |

The deciding factor is the **PBS cgroup cpuset**: when PBS allocates us
a cpuset confined to a single socket (slot pattern `1*24`),
`--map-by socket` has only one socket to map to and round-robins both
ranks onto it. When PBS gives us a cpuset spanning both sockets (slot
pattern `2*24`), the mapping fans out as intended.

This is *not* an Open MPI bug — see
[open-mpi/ompi#9647](https://github.com/open-mpi/ompi/issues/9647)
(reported from a Gadi node). 5.x treats the cgroup as a hard
constraint and discards cores not in the cpuset.

## Reading `--report-bindings` correctly

A pitfall: the output prints the rank's **locale**, not always its
binding mask. With OpenMPI 5.x defaults for `np ≤ 2` the implicit
binding is `--bind-to core`, but if `--bind-to socket` is forced and
the cpuset is one socket, the rank ends up effectively unbound.

- `package[0][core:0-23]` (24-core range = full socket) → rank free to
  roam the whole socket. **Not a tight pin.**
- `package[0][core:12-23]` (12-core range = one NUMA node) → tight pin
  to one NUMA node. ✓
- `Rank N is not bound` (or all-cores mask) → no binding applied.

Add `--display map-devel,bind` to disambiguate — that prints the
binding mask alongside the locale.

In our scripts, both flags are on the mpiexec calls inside
`run_1year_benchmark.sh` (profile and non-profile paths). See commit
`60f00c8` (Add --display map-devel,bind to benchmark mpiexec calls).

## What to do about the non-determinism

Option list, weakest → strongest:

1. **Status quo (`--bind-to socket --map-by socket`):** sometimes good,
   sometimes both ranks share one socket. We don't choose; PBS does.
2. **NUMA-aware mapping**
   (`--bind-to numa --map-by numa` or `--map-by ppr:1:numa`): with 4
   NUMA nodes per node and our 24-core slice straddling 2 NUMA nodes
   in either layout, two ranks land on two different NUMA nodes
   deterministically. NCI's Hybrid MPI / OpenMP page recommends
   `ppr:N:numa` for NUMA-aware placement on Gadi.
3. **Whole-node request (`ngpus=4 ncpus=48`) for sub-4-rank jobs:**
   guarantees both sockets are in the cpuset, but burns SUs on idle
   GPUs. Rules out same-socket effects entirely.
4. **Explicit rankfile / `--cpu-list`:** maximum control, fragile.

Open issue (not yet decided): which of (2) / (3) we adopt for the
strong-scaling benchmarks. Currently we keep (1) and use
`--report-bindings --display map-devel,bind` to record what we got, so
results stay interpretable post-hoc.

PBS placement directives (`place=scatter` etc.) only affect multi-chunk
jobs *across* hosts — they do not force PBS to give us cores spanning
both sockets within one chunk. There is no documented Gadi-side
directive to fix this for partial-node `ngpus=2` requests.

## References

- Open MPI 5.0.x mpirun(1) man page:
  <https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html>
- [open-mpi/ompi#9647](https://github.com/open-mpi/ompi/issues/9647) —
  the same-socket-binding behaviour, reported from a Gadi node.
- [open-mpi/ompi#11541](https://github.com/open-mpi/ompi/issues/11541) —
  `--bind-to socket` interaction with default mappings when
  `np ≤ #sockets`.
- NCI: [Hybrid MPI and OpenMP](https://opus.nci.org.au/spaces/Help/pages/122552392/Hybrid+MPI+and+OpenMP)
- NCI: [Queue Structure on Gadi (gpuhopper)](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi)
