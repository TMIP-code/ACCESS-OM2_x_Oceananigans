# Partition Load Balance Diagnostic

Per-rank loads for every y-partition we have used (1×2, 1×4, 1×8) under
each load-balance scheme:

- **equal** — `Partition(1, py)` default. Equal `Ny / py` y-rows per rank.
- **surface** — greedy wet-column-count balance (LB `_LBS`; env
  `LOAD_BALANCE=surface`).
- **cell** — greedy wet-3D-cell-count balance, using Oceananigans'
  `immersed_cell(i, j, k, grid)` as the ground truth for "is this cell
  in the prognostic state" (LB `_LB`; env `LOAD_BALANCE=cell`). Counts
  partial bottom cells as wet, matching the simulation.
- **mix** — greedy balance of an equal-weighted normalised mix of
  cells & columns: `wet_cells[j]/Σ wet_cells + wet_cols[j]/Σ wet_cols`
  (LB `_LBmix`; env `LOAD_BALANCE=mix`). Aimed at workloads where
  some kernels are cell-bound (3D tendencies) and some are
  column-bound (implicit vertical diffusion) — balancing neither
  perfectly but both well.
"True wet cells" / "true wet columns" in every table use `immersed_cell`
on the actual `ImmersedBoundaryGrid + PartialCellBottom`.

**Imbalance metric.** `imb%(cells) = (max - mean) / mean × 100` —
how much more 3D wet-cell work the heaviest rank has compared to the
ideal equal share. `imb%(surface)` is the same for wet columns
(per-column work, e.g. implicit vertical diffusion). `×ratio` is
`max / min` for cells (a different metric, more sensitive near
balance). Lower is better; 0 = perfectly balanced.

Numbers and plots are reproduced by
[`src/test_partition_balance.jl`](../src/test_partition_balance.jl), run
via [`scripts/test_driver.sh`](../scripts/test_driver.sh) (express
queue, 47GB, 1 CPU):

```bash
PARENT_MODEL=ACCESS-OM2-1   JOB_CHAIN=partbalance bash scripts/test_driver.sh
PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=partbalance bash scripts/test_driver.sh
PARENT_MODEL=ACCESS-OM2-01  JOB_CHAIN=partbalance bash scripts/test_driver.sh
```

Outputs:

- per-model log under `logs/julia/{PM}/{EXP}/test/partition_balance_{JOB_ID}.log`
- plots under `outputs/{PM}/{EXP}/partition_balance/{PM}_1x{py}.png`
  (gitignored; embedded in this doc via relative paths — visible in a
  local Markdown viewer but not on GitHub).

Rank ordering everywhere: **rank 0 = south** (j=1), rank py-1 = north
(j=Ny, tripolar fold).

---

## Headline observation

`:cell` balances **3D cells** very well (≤2.5% imbalance at all
partitions tested). It does so by giving the south fewer y-rows and the
north many more — because southern slabs are much denser in 3D wet cells
(deep Southern Ocean). But the north has *more wet columns* even in
those wider slabs, so:

- **Per-column work is unbalanced the *other* way.** Under `:cell` at
  1×2 on OM2-025, rank 1 (north) has ~19% more wet columns than rank 0
  (south). The simulation runs **implicit vertical diffusion** (per
  column), so this column surplus translates directly into more work
  for rank 1 — matching the wall-clock observation that the `_LB`
  partitions ran no faster (or slower) than baseline.
- **3D tendency kernels** (per-cell) ARE well balanced under `:cell`.
  The remaining wall-clock imbalance therefore is dominated by the
  per-column work (implicit vertical diffusion), plus tripolar-fold
  halo overhead on the north rank, and any per-cell work that doesn't
  scale with wet-cell density.
- **`:mix` trades a bit of cell-balance for a lot of column-balance.**
  Under `:mix` at 1×2, OM2-025 cell-imbalance climbs from 0.2% → 4.7%
  but column-imbalance drops from 8.8% → 4.7%. At 1×8 OM2-025 the
  trade is 1.2% → 6.6% (cells) vs 53% → 19% (cols). Whether this is a
  net win depends on the share of column-bound work in the bench — to
  be measured.

(A pre-2026-05-11 `:cell_obsolete` formula based on `z_center > bottom`
was checked separately; it differed from the new `immersed_cell`-based
`:cell` by at most one y-row at every (model, py) tested, so the
old-`_LB` partitions built with it are essentially equivalent.)

---

## ACCESS-OM2-1 (Nx=360, Ny=300, Nz=50)

Total wet cells = **2,707,869**, total wet columns = **69,809**.

### 1×2

![OM2-1 1×2 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x2.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|--------------:|--------------------:|
| equal         |         24  |    1.6 |           15  |            150, 150 |
| surface       |        8.4  |    1.2 |          0.57 |            132, 168 |
| **cell**      |        0.8  |    1.0 |           6.3 |            123, 177 |
| **mix**       |        4.1  |    1.1 |           3.2 |            127, 173 |

### 1×4

![OM2-1 1×4 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x4.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny           |
|---------------|------------:|-------:|--------------:|-------------------|
| equal         |         31  |    2.9 |           20  | 75  75  75  75    |
| surface       |        9.4  |    1.5 |          0.86 | 70  62  65 103    |
| **cell**      |        0.8  |    1.0 |           20  | 67  56  58 119    |
| **mix**       |        5.3  |    1.2 |           10  | 68  59  61 112    |

### 1×8

![OM2-1 1×8 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x8.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny                          |
|---------------|------------:|-------:|--------------:|----------------------------------|
| equal         |         69  |    5.8 |           51  | 38  38  38  38  37  37  37  37   |
| surface       |         12  |    1.7 |           2.2 | 45  25  30  32  31  34  60  43   |
| **cell**      |        2.5  |    1.1 |           51  | 45  22  27  29  28  30  38  81   |
| **mix**       |        5.9  |    1.4 |           22  | 45  23  29  30  30  31  48  64   |

---

## ACCESS-OM2-025 (Nx=1440, Ny=1080, Nz=50)

Total wet cells = **36,952,668**, total wet columns = **970,921**.

### 1×2

![OM2-025 1×2 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x2.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|--------------:|--------------------:|
| equal         |         32  |    1.9 |           20  |            540, 540 |
| surface       |        9.9  |    1.2 |          0.01 |            453, 627 |
| **cell**      |        0.2  |    1.0 |           8.8 |            415, 665 |
| **mix**       |        4.7  |    1.1 |           4.7 |            433, 647 |

**Note the 2D vs 3D split at 1×2 cell:** `:cell` drives 3D wet-cell imbalance down to 0.2%, but the corresponding wet-column (surface) imbalance is 8.8% — same direction as `equal` (rank 1 over-loaded), just smaller magnitude. Per-column work in this simulation (notably **implicit vertical diffusion**) scales with the wet-column count, so this rank-1 surplus is exactly the imbalance the wall-clock data shows.

### 1×4

![OM2-025 1×4 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x4.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny           |
|---------------|------------:|-------:|--------------:|-------------------|
| equal         |         49  |    3.1 |           34  | 270 270 270 270   |
| surface       |         12  |    1.6 |          0.53 | 260 193 251 376   |
| **cell**      |        0.4  |    1.0 |           27  | 247 168 206 459   |
| **mix**       |        5.5  |    1.2 |           14  | 253 180 226 421   |

### 1×8

![OM2-025 1×8 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x8.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny                              |
|---------------|------------:|-------:|--------------:|--------------------------------------|
| equal         |         79  |    6.4 |           58  | 135 135 135 135 135 135 135 135      |
| surface       |         16  |    1.7 |          0.69 | 175  85  87 106 109 142 232 144      |
| **cell**      |        1.2  |    1.0 |           53  | 175  72  78  90  98 108 173 286      |
| **mix**       |        6.6  |    1.3 |           19  | 175  78  82  98 104 122 241 180      |

---

## ACCESS-OM2-01 (Nx=3600, Ny=2700, Nz=75)

Total wet cells = **351,532,308**, total wet columns = **6,075,239**.

### 1×2

![OM2-01 1×2 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x2.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|--------------:|--------------------:|
| equal         |         30  |    1.9 |           19  |          1350, 1350 |
| surface       |        9.5  |    1.2 |          0.03 |          1138, 1562 |
| **cell**      |        0.1  |    1.0 |           8.5 |          1045, 1655 |
| **mix**       |        4.5  |    1.1 |           4.5 |          1090, 1610 |

### 1×4

![OM2-01 1×4 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x4.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny                |
|---------------|------------:|-------:|--------------:|------------------------|
| equal         |         47  |    2.9 |           33  | 675 675 675 675        |
| surface       |         11  |    1.5 |          0.22 | 651 487 641 921        |
| **cell**      |        0.2  |    1.0 |           26  | 620 425 530 1125       |
| **mix**       |        5.3  |    1.2 |           14  | 634 456 575 1035       |

### 1×8

![OM2-01 1×8 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x8.png)

| scheme        | imb%(cells) | ×ratio | imb%(surface) | slab Ny                            |
|---------------|------------:|-------:|--------------:|------------------------------------|
| equal         |         77  |    5.8 |           58  | 338 338 338 338 337 337 337 337    |
| surface       |         14  |    1.6 |          0.33 | 438 213 219 268 276 365 577 344    |
| **cell**      |        0.5  |    1.0 |           47  | 435 185 195 230 250 280 475 650    |
| **mix**       |        6.5  |    1.3 |           17  | 436 198 207 249 262 313 621 414    |

---

## What this suggests for the LB regressions

1. **At 1×2, `:cell` already balances 3D cells nearly perfectly.** The
   wall-clock regression we see (e.g. OM2-025 H200 LB: 5m 40s vs
   baseline 5m 35s) is not because the 3D cell count is bad. The
   suspect is rank 1 owning more wet columns AND the entire tripolar
   fold halo.

2. **`:mix` already implements a combined cell+column proxy** (equal
   weights, each metric normalised by its own total). Static numbers
   above show it cuts the column-imbalance ~3× at the cost of a few %
   cell-imbalance. Next step: a benchmark sweep over `:cell` vs `:mix`
   vs `:surface` at 1×2/1×4/1×8 on OM2-025 (cheap on H200) to see
   which actually wins on wall time. Future tuning could explore
   non-equal weights (α · wet_cells + β · wet_columns) once we have a
   measured share of column-bound vs cell-bound work from a per-y-row
   profiling pass.

3. **Tripolar-fold halo skew.** Rank py-1 always owns the
   tripolar-fold zipper boundary; its halo-exchange cost is higher per
   row than ordinary rows. The greedy splitter doesn't see this, so it
   gives the north rank more rows just because its cell density is
   lower. Worth adding a small fold-rank penalty term.
