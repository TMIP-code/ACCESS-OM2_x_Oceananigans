# Test results

All tests run on Gadi (NCI) for `PARENT_MODEL=ACCESS-OM2-1`.

## Last known good

| Test | Commit | Job ID | Date | Notes |
|------|--------|--------|------|-------|
| Grid identity | `8f59402` | `163528343` | 2026-03-19 | All 20 metrics, 4 ranks: max\|diff\| = 0 |
| CPU diag serial | `8f59402` | `163528341` | 2026-03-19 | |
| CPU diag 2x2 | `8f59402` | `163528340` | 2026-03-19 | After distributed grid fix |
| Compare (serial vs 2x2) | `8f59402` | `163528797` | 2026-03-19 | w max\|diff\|=2.34e-03 mean=3.5e-07 (metrics fixed, PartialCellBottom halos remain) |
| GPU run1yr serial | `b8fbe81` | `163519470` | 2026-03-19 | Age-only output on GPU |
| GPU run1yr 2x2 | `b8fbe81` | `163519472` | 2026-03-19 | Age-only output on GPU |
| GPU benchmark serial | `e097fcc` | `163517861` | 2026-03-19 | 41.4s |
| GPU benchmark 2x2 | `e097fcc` | `163517863` | 2026-03-19 | 41.2s |
| Preprocessing | `4cee505` | `163527903`/`04` | 2026-03-19 | |

---

## Grid identity test

Verifies that serial and distributed grids have identical coordinate/metric arrays.

```bash
GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=gridtest bash scripts/test_driver.sh
```

| Commit | Job ID | Status | Notes |
|--------|--------|--------|-------|
| `8f59402` | `163528343` | PASSED | All 20 metrics, 4 ranks: max\|diff\| = 0 |
| `4cee505` | `163527912` | FAILED | OffsetArrays not in Project.toml + Array() on OffsetArray + scope bug |

## CPU diagnostic (10-step serial vs 2x2 distributed)

10-step age simulation saving every step. Used to compare serial vs distributed outputs.

```bash
# Serial
GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diagcpuserial bash scripts/test_driver.sh
# Distributed
GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diagcpu bash scripts/test_driver.sh
```

| Commit | Job ID | Type | Status | Notes |
|--------|--------|------|--------|-------|
| `8f59402` | `163528341` | serial | FINISHED | |
| `8f59402` | `163528340` | 2x2 | FINISHED | |
| `4cee505` | `163527910` | serial | FINISHED | |
| `4cee505` | `163527908` | 2x2 | ERRORED | @__DIR__ path depth wrong after refactoring |
| `b8fbe81` | `163517865` | serial | FINISHED | Pre-refactoring, MVD fix only |
| `b8fbe81` | `163517864` | 2x2 | FINISHED | Pre-refactoring, MVD fix only |

## Compare serial vs distributed

Compares age/velocity fields between serial and distributed diagnostic outputs.

```bash
GPU_RESOURCES=gpuvolta-2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=compare DURATION_TAG=diag bash scripts/test_driver.sh
```

| Commit | Job ID | Status | w max\|diff\| | Notes |
|--------|--------|--------|--------------|-------|
| `8f59402` | `163528342` | FINISHED | 2.08e-03 | Used OLD 2x2 outputs — stale |
| `8f59402` | `163528797` | FINISHED | 2.34e-03 | Fresh outputs, grid metrics fixed; remaining diff from PartialCellBottom halos |
| `4cee505` | `163527911` | ERRORED | | @__DIR__ path depth wrong |
| `b8fbe81` | `163519499` | FINISHED | 2.08e-03 | MVD fix only — grid metrics still differ |
| `47a7da1` | `163506624` | FINISHED | 2.08e-03 | Pre-MVD fix (same result) |

## GPU standard runs (1-year age simulation)

```bash
# Serial
PARENT_MODEL=ACCESS-OM2-1 GPU_RESOURCES=gpuvolta JOB_CHAIN=run1yr bash scripts/driver.sh
# Distributed
PARENT_MODEL=ACCESS-OM2-1 GPU_RESOURCES=gpuvolta-2x2 JOB_CHAIN=run1yr bash scripts/driver.sh
```

| Commit | Job ID | Type | Status | Notes |
|--------|--------|------|--------|-------|
| `b8fbe81` | `163519470` | serial | FINISHED | Age-only output on GPU |
| `b8fbe81` | `163519472` | 2x2 | FINISHED | Age-only output on GPU |
| `e097fcc` | `163517857` | serial | ERRORED | Array(interior(TSI)) scalar indexing on GPU |
| `e097fcc` | `163517859` | 2x2 | ERRORED | Array(interior(TSI)) scalar indexing on GPU |

## GPU benchmark runs (1-year, no output writers)

```bash
PARENT_MODEL=ACCESS-OM2-1 GPU_RESOURCES=gpuvolta JOB_CHAIN=run1yrfast bash scripts/driver.sh
```

| Commit | Job ID | Type | Status | Walltime | Notes |
|--------|--------|------|--------|----------|-------|
| `e097fcc` | `163517861` | serial | FINISHED | 41.4s | |
| `e097fcc` | `163517863` | 2x2 | FINISHED | 41.2s | |

## Preprocessing (grid + velocities)

```bash
PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=preprocessing bash scripts/driver.sh
```

| Commit | Grid Job | Grid Status | Vel Job | Vel Status | Notes |
|--------|----------|-------------|---------|------------|-------|
| `4cee505` | `163527903` | FINISHED | `163527904` | FINISHED | After refactoring + distributed grid fix |
| `4cee505` | `163527576` | ERRORED | — | CANCELLED | OffsetArrays not in Project.toml |
| `e097fcc` | `163514040` | FINISHED | `163514041` | FINISHED | Save z_faces, enforce MVD at load time |
