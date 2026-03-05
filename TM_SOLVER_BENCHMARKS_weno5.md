# Transport Matrix Solver Benchmarks — weno5

model_config: `cgridtransports_wdiagnosed_weno5_AB2`

Regenerate this file with:
```bash
julia src/summarize_TM_age_solves.jl
```

Latest run per (Size, Solver, MatrixProcessing) configuration.

## full / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| ParU | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378457 |
| UMFPACK | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378472 |
| CUDSS | FAIL |  |  |  |  |  |  | 12 | 1 | 2.10 | 162378483 |
| Pardiso | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378384 |

## coarse / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| CUDSS | OK | 18s | 124 MiB | 5.4 MiB | 470ms | 34 MiB | 5.4 MiB | 12 | 1 | 2.30 | 162378484 |
| ParU | OK | 5.5min | 101 GiB |  | 1.7s | 17 MiB |  | 48 |  | 13.44 | 162378465 |
| UMFPACK | OK | 12min | 48 GiB |  | 4.1s |  |  | 48 |  | 24.69 | 162378481 |
| Pardiso | FAIL |  |  |  |  |  |  | 48 |  | 3.49 | 162378414 |

## full / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378390 |

## coarse / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 1.0min | 212 MiB |  | 810ms | 60 MiB |  | 48 |  | 6.35 | 162378421 |

## full / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 0 |  | 0.00 | 162378399 |

## coarse / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 1.1min | 200 MiB |  | 894ms | 54 MiB |  | 48 |  | 6.27 | 162378432 |

## full / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 8.4min | 468 MiB |  | 5.2s | 217 MiB |  | 48 |  | 17.97 | 162378406 |

## coarse / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 15s | 186 MiB |  | 1.5s | 47 MiB |  | 48 |  | 5.28 | 162378449 |

**Total: 14 cases (7 successful), 81.89 SU**
