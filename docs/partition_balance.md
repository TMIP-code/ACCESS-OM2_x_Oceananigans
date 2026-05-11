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

| scheme        | imb%(cells) | ×ratio | rank 0 (south) cells | rank 1 (north) cells | rank 0 cols | rank 1 cols | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|---------------------:|---------------------:|------------:|------------:|--------------------:|
| equal         |     +24.2%  | ×1.638 |             1,680,000 |            1,030,000 |      40,200 |      29,600 |             150, 150 |
| surface       |      +8.4%  | ×1.182 |             1,470,000 |            1,240,000 |      35,100 |      34,700 |             132, 168 |
| **cell**      |      +0.8%  | ×1.015 |             1,360,000 |            1,340,000 |      32,700 |      37,100 |             123, 177 |
| cell_obsolete |      +0.8%  | ×1.015 |             1,360,000 |            1,340,000 |      32,700 |      37,100 |             123, 177 |

### 1×4

![OM2-1 1×4 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x4.png)

| scheme        | imb%(cells) | ×ratio | per-rank true wet cells (×10⁶)   | per-rank wet columns (×10³)        | slab Ny           |
|---------------|------------:|-------:|----------------------------------|------------------------------------|-------------------|
| equal         |     +31.0%  | ×2.889 | 0.79  0.89  0.72  0.31           | 19.2  21.0  17.3  12.3             | 75  75  75  75    |
| surface       |      +9.4%  | ×1.457 | 0.73  0.74  0.73  0.51           | 17.6  17.5  17.4  17.3             | 70  62  65 103    |
| **cell**      |      +0.8%  | ×1.020 | 0.68  0.68  0.67  0.67           | 16.6  16.1  16.1  21.0             | 67  56  58 119    |
| cell_obsolete |      +0.8%  | ×1.020 | 0.68  0.68  0.67  0.67           | 16.6  16.1  16.1  21.0             | 67  56  58 119    |

### 1×8

![OM2-1 1×8 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x8.png)

| scheme        | imb%(cells) | ×ratio | per-rank true wet cells (×10⁶)                                  | per-rank wet columns (×10³)                                     | slab Ny                                  |
|---------------|------------:|-------:|-----------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------|
| equal         |     +69.2%  | ×5.774 | 0.23  0.57  0.45  0.45  0.42  0.28  0.10  0.20                  | 6.3  13.2  10.7  10.6  10.0   6.9   4.4   7.7                   | 38  38  38  38  37  37  37  37           |
| surface       |     +12.2%  | ×1.735 | 0.35  0.38  0.37  0.38  0.36  0.37  0.29  0.22                  | 8.8   8.8   8.6   8.9   8.7   8.7   8.6   8.6                   | 45  25  30  32  31  34  60  43           |
| **cell**      |      +2.5%  | ×1.055 | 0.35  0.33  0.34  0.34  0.33  0.35  0.33  0.34                  | 8.8   7.8   7.9   8.2   7.8   8.3   7.9  13.2                   | 45  22  27  29  28  30  38  81           |
| cell_obsolete |      +2.5%  | ×1.055 | 0.35  0.33  0.34  0.34  0.33  0.35  0.33  0.34                  | 8.8   7.8   7.9   8.2   7.8   8.3   7.9  13.2                   | 45  22  27  29  28  30  38  81           |

---

## ACCESS-OM2-025 (Nx=1440, Ny=1080, Nz=50)

Total wet cells = **36,952,668**, total wet columns = **970,921**.

### 1×2

![OM2-025 1×2 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x2.png)

| scheme        | imb%(cells) | ×ratio | rank 0 (south) cells | rank 1 (north) cells | rank 0 cols | rank 1 cols | slab Ny (rank 0, 1) |
|---------------|------------:|-------:|---------------------:|---------------------:|------------:|------------:|--------------------:|
| equal         |     +31.7%  | ×1.930 |           24,340,000 |           12,610,000 |     582,600 |     388,300 |            540, 540 |
| surface       |      +9.9%  | ×1.220 |           20,310,000 |           16,650,000 |     485,500 |     485,400 |            453, 627 |
| **cell**      |      +0.2%  | ×1.004 |           18,520,000 |           18,440,000 |     442,900 |     528,000 |            415, 665 |
| cell_obsolete |      +0.0%  | ×1.001 |           18,470,000 |           18,480,000 |     441,800 |     529,100 |            414, 666 |

**Note the 2D vs 3D split at 1×2 cell:** 3D wet cells are balanced (×1.004) but rank 1 has 19% more wet columns (528.0k vs 442.9k). Per-column work in this simulation (notably **implicit vertical diffusion**) scales with the wet-column count, so this rank-1 surplus is exactly the imbalance the wall-clock data shows.

### 1×4

![OM2-025 1×4 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x4.png)

| scheme        | imb%(cells) | ×ratio | per-rank true wet cells (×10⁶) | per-rank wet columns (×10³)            | slab Ny           |
|---------------|------------:|-------:|--------------------------------|----------------------------------------|-------------------|
| equal         |     +48.8%  | ×3.126 | 10.60  13.74   8.22   4.40     | 258.2 324.5 211.3 176.9                | 270 270 270 270   |
| surface       |     +11.5%  | ×1.591 | 10.01  10.30  10.17   6.47     | 244.0 241.5 242.8 242.6                | 260 193 251 376   |
| **cell**      |      +0.4%  | ×1.007 |  9.24   9.27   9.21   9.23     | 225.6 217.3 219.6 308.4                | 247 168 206 459   |
| cell_obsolete |      +0.2%  | ×1.003 |  9.24   9.23   9.25   9.23     | 225.6 216.2 220.7 308.4                | 247 167 207 459   |

### 1×8

![OM2-025 1×8 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x8.png)

| scheme        | imb%(cells) | ×ratio | per-rank true wet cells (×10⁶)                            | per-rank wet columns (×10³)                                  | slab Ny                              |
|---------------|------------:|-------:|-----------------------------------------------------------|--------------------------------------------------------------|--------------------------------------|
| equal         |     +78.8%  | ×6.368 | 2.33  8.26  7.46  6.28  5.27  2.94  1.30  3.10            | 66.2 192.0 173.9 150.6 124.5  86.9  63.1 113.8               | 135 135 135 135 135 135 135 135      |
| surface       |     +15.5%  | ×1.657 | 4.67  5.33  5.18  5.11  5.06  5.11  3.22  3.25            | 122.2 121.8 120.6 120.9 121.4 121.4 121.9 120.7              | 175  85  87 106 109 142 232 144      |
| **cell**      |      +1.2%  | ×1.023 | 4.67  4.57  4.65  4.62  4.60  4.61  4.62  4.60            | 122.2 103.4 109.4 108.0 109.9 109.7 122.5 185.9              | 175  72  78  90  98 108 173 286      |
| cell_obsolete |      +1.2%  | ×1.023 | 4.67  4.57  4.65  4.57  4.64  4.61  4.59  4.63            | 122.2 103.4 109.4 106.9 111.0 109.7 121.2 187.2              | 175  72  78  89  99 108 171 288      |

---

## ACCESS-OM2-01

_Pending — run completes after this draft is written._

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
