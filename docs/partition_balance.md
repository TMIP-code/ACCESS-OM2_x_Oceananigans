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
- **cell_obsolete** — previous `:cell` formula (`z_center > bottom`).
  Under-counts wet cells because it excludes partial bottom cells. This
  is what built every `_LB` partition before 2026-05-11.

"True wet cells" / "true wet columns" in every table use `immersed_cell`
on the actual `ImmersedBoundaryGrid + PartialCellBottom`.

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

**`:cell` and `:cell_obsolete` give nearly identical partitions on
OM2-1 and OM2-025** (≤1 row difference). The "include partial bottom
cells" fix moves the boundary by at most a single y-row at these
resolutions — so the load-balance regressions we saw in profiling
cannot be blamed on the obsolete formula. (The plots below omit
`cell_obsolete` since it overlaps `cell`; tables keep it for
completeness.)

---

## ACCESS-OM2-1 (Nx=360, Ny=300, Nz=50)

Total wet cells = **2,707,869**, total wet columns = **69,809**.

### 1×2

![OM2-1 1×2 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x2.png)

| scheme        | imb%(cells) | ×ratio | rank 0 (south) cells % | rank 1 (north) cells % | rank 0 cols % | rank 1 cols % | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|-----------------------:|-----------------------:|--------------:|--------------:|--------------------:|
| equal         |     +24.2%  | ×1.638 |                    62% |                    38% |           58% |           42% |             150, 150 |
| surface       |      +8.4%  | ×1.182 |                    54% |                    46% |           50% |           50% |             132, 168 |
| **cell**      |      +0.8%  | ×1.015 |                    50% |                    50% |           47% |           53% |             123, 177 |
| cell_obsolete |      +0.8%  | ×1.015 |                    50% |                    50% |           47% |           53% |             123, 177 |

### 1×4

![OM2-1 1×4 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x4.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%)   | per-rank wet cols (%)    | slab Ny           |
|---------------|------------:|-------:|--------------------------|--------------------------|-------------------|
| equal         |     +31.0%  | ×2.889 | 29  33  27  11           | 28  30  25  18           | 75  75  75  75    |
| surface       |      +9.4%  | ×1.457 | 27  27  27  19           | 25  25  25  25           | 70  62  65 103    |
| **cell**      |      +0.8%  | ×1.020 | 25  25  25  25           | 24  23  23  30           | 67  56  58 119    |
| cell_obsolete |      +0.8%  | ×1.020 | 25  25  25  25           | 24  23  23  30           | 67  56  58 119    |

### 1×8

![OM2-1 1×8 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x8.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%)         | per-rank wet cols (%)          | slab Ny                          |
|---------------|------------:|-------:|--------------------------------|--------------------------------|----------------------------------|
| equal         |     +69.2%  | ×5.774 |  9  21  17  17  16  10   4   7 |  9  19  15  15  14  10   6  11 | 38  38  38  38  37  37  37  37   |
| surface       |     +12.2%  | ×1.735 | 13  14  14  14  13  14  11   8 | 13  13  12  13  12  12  12  12 | 45  25  30  32  31  34  60  43   |
| **cell**      |      +2.5%  | ×1.055 | 13  12  13  13  12  13  12  13 | 13  11  11  12  11  12  11  19 | 45  22  27  29  28  30  38  81   |
| cell_obsolete |      +2.5%  | ×1.055 | 13  12  13  13  12  13  12  13 | 13  11  11  12  11  12  11  19 | 45  22  27  29  28  30  38  81   |

---

## ACCESS-OM2-025 (Nx=1440, Ny=1080, Nz=50)

Total wet cells = **36,952,668**, total wet columns = **970,921**.

### 1×2

![OM2-025 1×2 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x2.png)

| scheme        | imb%(cells) | ×ratio | rank 0 (south) cells % | rank 1 (north) cells % | rank 0 cols % | rank 1 cols % | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|-----------------------:|-----------------------:|--------------:|--------------:|--------------------:|
| equal         |     +31.7%  | ×1.930 |                    66% |                    34% |           60% |           40% |            540, 540 |
| surface       |      +9.9%  | ×1.220 |                    55% |                    45% |           50% |           50% |            453, 627 |
| **cell**      |      +0.2%  | ×1.004 |                    50% |                    50% |           46% |           54% |            415, 665 |
| cell_obsolete |      +0.0%  | ×1.001 |                    50% |                    50% |           46% |           54% |            414, 666 |

**Note the 2D vs 3D split at 1×2 cell:** 3D wet cells are balanced (50%/50%) but rank 1 ends up with 54% of the wet columns (vs rank 0's 46%) — ~19% more. Per-column work in this simulation (notably **implicit vertical diffusion**) scales with the wet-column count, so this rank-1 surplus is exactly the imbalance the wall-clock data shows.

### 1×4

![OM2-025 1×4 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x4.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%) | per-rank wet cols (%) | slab Ny           |
|---------------|------------:|-------:|------------------------|-----------------------|-------------------|
| equal         |     +48.8%  | ×3.126 | 29  37  22  12         | 27  33  22  18        | 270 270 270 270   |
| surface       |     +11.5%  | ×1.591 | 27  28  28  18         | 25  25  25  25        | 260 193 251 376   |
| **cell**      |      +0.4%  | ×1.007 | 25  25  25  25         | 23  22  23  32        | 247 168 206 459   |
| cell_obsolete |      +0.2%  | ×1.003 | 25  25  25  25         | 23  22  23  32        | 247 167 207 459   |

### 1×8

![OM2-025 1×8 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x8.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%)         | per-rank wet cols (%)          | slab Ny                              |
|---------------|------------:|-------:|--------------------------------|--------------------------------|--------------------------------------|
| equal         |     +78.8%  | ×6.368 |  6  22  20  17  14   8   4   8 |  7  20  18  16  13   9   6  12 | 135 135 135 135 135 135 135 135      |
| surface       |     +15.5%  | ×1.657 | 13  14  14  14  14  14   9   9 | 13  13  12  12  13  13  13  12 | 175  85  87 106 109 142 232 144      |
| **cell**      |      +1.2%  | ×1.023 | 13  12  13  13  12  12  13  12 | 13  11  11  11  11  11  13  19 | 175  72  78  90  98 108 173 286      |
| cell_obsolete |      +1.2%  | ×1.023 | 13  12  13  12  13  12  12  13 | 13  11  11  11  11  11  12  19 | 175  72  78  89  99 108 171 288      |

---

## ACCESS-OM2-01 (Nx=3600, Ny=2700, Nz=75)

Total wet cells = **351,532,308**, total wet columns = **6,075,239**.

### 1×2

![OM2-01 1×2 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x2.png)

| scheme        | imb%(cells) | ×ratio | rank 0 (south) cells % | rank 1 (north) cells % | rank 0 cols % | rank 1 cols % | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|-----------------------:|-----------------------:|--------------:|--------------:|--------------------:|
| equal         |     +30.4%  | ×1.873 |                    65% |                    35% |           60% |           40% |          1350, 1350 |
| surface       |      +9.5%  | ×1.209 |                    55% |                    45% |           50% |           50% |          1138, 1562 |
| **cell**      |      +0.1%  | ×1.001 |                    50% |                    50% |           46% |           54% |          1045, 1655 |
| cell_obsolete |      +0.0%  | ×1.000 |                    50% |                    50% |           46% |           54% |          1044, 1656 |

### 1×4

![OM2-01 1×4 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x4.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%) | per-rank wet cols (%) | slab Ny                |
|---------------|------------:|-------:|------------------------|-----------------------|------------------------|
| equal         |     +46.7%  | ×2.867 | 29  37  22  13         | 26  33  21  19        | 675 675 675 675        |
| surface       |     +10.9%  | ×1.544 | 27  28  27  18         | 25  25  25  25        | 651 487 641 921        |
| **cell**      |      +0.2%  | ×1.003 | 25  25  25  25         | 23  22  23  31        | 620 425 530 1125       |
| cell_obsolete |      +0.2%  | ×1.005 | 25  25  25  25         | 23  22  23  32        | 620 424 530 1126       |

### 1×8

![OM2-01 1×8 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x8.png)

| scheme        | imb%(cells) | ×ratio | per-rank wet cells (%)         | per-rank wet cols (%)          | slab Ny                            |
|---------------|------------:|-------:|--------------------------------|--------------------------------|------------------------------------|
| equal         |     +76.8%  | ×5.770 |  6  22  20  17  14   8   4   9 |  7  20  18  15  13   9   7  12 | 338 338 338 338 337 337 337 337    |
| surface       |     +14.0%  | ×1.609 | 13  14  14  14  14  14   9   9 | 13  13  12  13  13  12  13  12 | 438 213 219 268 276 365 577 344    |
| **cell**      |      +0.5%  | ×1.009 | 13  13  12  13  13  12  12  12 | 12  11  11  11  11  11  13  18 | 435 185 195 230 250 280 475 650    |
| cell_obsolete |      +0.5%  | ×1.009 | 13  13  12  12  13  12  12  13 | 12  11  11  11  11  11  13  19 | 435 185 195 229 250 280 472 654    |

---

## What this suggests for the LB regressions

1. **At 1×2, `:cell` already balances 3D cells nearly perfectly.** The
   wall-clock regression we see (e.g. OM2-025 H200 LB: 5m 40s vs
   baseline 5m 35s) is not because the 3D cell count is bad. The
   suspect is rank 1 owning more wet columns AND the entire tripolar
   fold halo.

2. **`:cell` vs `:cell_obsolete` is a non-issue here.** They differ by
   ≤1 row at every (model, py) tested. The obsolete formula is not
   the cause of the LB regressions.

3. **A combined cell+column proxy might do better.** Examples to
   prototype:
   - Per-rank weighted load `α · wet_cells + β · wet_columns` with
     β/α tuned to the share of column-major work (implicit vertical
     diffusion + halo + immersed-mask) in the bench (probably 10–30%).
   - Or directly the metric the model already exposes — total cost per
     y-row from a short profiling run — instead of static cell counts.

4. **Tripolar-fold halo skew.** Rank py-1 always owns the
   tripolar-fold zipper boundary; its halo-exchange cost is higher per
   row than ordinary rows. The greedy splitter doesn't see this, so it
   gives the north rank more rows just because its cell density is
   lower. Worth adding a small fold-rank penalty term.
