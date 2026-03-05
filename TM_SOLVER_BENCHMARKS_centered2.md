# Transport Matrix Solver Benchmarks — centered2

model_config: `cgridtransports_wdiagnosed_centered2_AB2`

Regenerate this file with:
```bash
julia src/summarize_TM_age_solves.jl
```

Latest run per (Size, Solver, MatrixProcessing) configuration.

## full / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| CUDSS | OK | 46s | 123 MiB | 21 MiB | 588ms | 34 MiB | 21 MiB | 12 | 1 | 2.72 | 162378165 |
| Pardiso | OK | 1.5min | 353 MiB |  | 2.3s | 160 MiB |  | 48 |  | 6.96 | 162378140 |
| ParU | OK | 11min | 116 GiB |  | 3.5s | 62 MiB |  | 48 |  | 22.19 | 162378161 |
| UMFPACK | OK | 23min | 94 GiB |  | 8.5s |  |  | 48 |  | 40.99 | 162378163 |

## coarse / raw

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| CUDSS | OK | 11s | 124 MiB | 5.4 MiB | 444ms | 34 MiB | 5.4 MiB | 12 | 1 | 2.16 | 162378166 |
| Pardiso | OK | 13s | 174 MiB |  | 304ms | 41 MiB |  | 48 |  | 5.01 | 162378156 |
| ParU | OK | 1.4min | 23 GiB |  | 637ms | 16 MiB |  | 48 |  | 7.12 | 162378162 |
| UMFPACK | FAIL |  |  |  |  |  |  | 48 |  | 3.39 | 162378164 |

## full / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 1.4min | 353 MiB |  | 1.7s | 160 MiB |  | 48 |  | 6.85 | 162378144 |

## coarse / symfill

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 12s | 174 MiB |  | 311ms | 41 MiB |  | 48 |  | 5.39 | 162378158 |

## full / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 48 |  | 29.39 | 162378149 |

## coarse / dropzeros

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | FAIL |  |  |  |  |  |  | 48 |  | 3.39 | 162378159 |

## full / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 1.7min | 352 MiB |  | 1.5s | 160 MiB |  | 48 |  | 7.15 | 162378152 |

## coarse / symdrop

| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |
|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|
| Pardiso | OK | 12s | 174 MiB |  | 297ms | 41 MiB |  | 48 |  | 5.01 | 162378160 |

**Total: 14 cases (11 successful), 147.72 SU**
