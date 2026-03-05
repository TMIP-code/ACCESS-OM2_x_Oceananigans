# Transport Matrix Solver Benchmarks — weno3

model_config: `cgridtransports_wdiagnosed_weno3_AB2`

Regenerate this file with:
```bash
julia src/summarize_TM_age_solves.jl
```

Latest run per (Size, Solver, MatrixProcessing) configuration.

## full / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378305 |
| UMFPACK | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378317 |
| ParU | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378315 |
| CUDSS | FAIL |  |  |  |  |  |  | 12 | 1 | 1.80 | 162378320 |

## coarse / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| CUDSS | OK | 11s | 124 MiB | 5.4 MiB | 428ms | 34 MiB | 5.4 MiB | 12 | 1 | 1.98 | 162378324 |
| Pardiso | OK | 15s | 184 MiB |  | 414ms | 46 MiB |  | 48 |  | 4.72 | 162378309 |
| UMFPACK | OK | 4.2min | 24 GiB |  | 1.9s |  |  | 48 |  | 11.25 | 162378318 |
| ParU | FAIL |  |  |  |  |  |  | 48 |  | 3.28 | 162378316 |

## full / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378306 |

## coarse / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 15s | 184 MiB |  | 413ms | 46 MiB |  | 48 |  | 4.91 | 162378311 |

## full / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 48 |  | 19.92 | 162378307 |

## coarse / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 14s | 180 MiB |  | 371ms | 45 MiB |  | 48 |  | 5.04 | 162378313 |

## full / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 48 |  | 10.03 | 162378308 |

## coarse / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 14s | 175 MiB |  | 2.1s | 42 MiB |  | 48 |  | 5.28 | 162378314 |

**Total: 14 cases (6 successful), 68.21 SU**
