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
- **minmax** — α-weighted mix where α is chosen by bisection to
  minimise `max(imb%(cells), imb%(surface))` (LB `_LBminmax`; env
  `LOAD_BALANCE=minmax`). Same load form as `:mix` but α is *picked
  adaptively* rather than fixed at ½ — lands on (or near) the Pareto
  crossover where the cell- and surface-imbalances are roughly equal,
  bounding the overload any single rank sees regardless of which
  kernel class dominates.
"True wet cells" / "true wet columns" in every table use `immersed_cell`
on the actual `ImmersedBoundaryGrid + PartialCellBottom`.

**Imbalance metric.** `imb%(cells) = (max - mean) / mean × 100` —
how much more 3D wet-cell work the heaviest rank has compared to the
ideal equal share. `imb%(surface)` is the same for wet columns
(per-column work, e.g. implicit vertical diffusion). `max%` is the
worst of the two — the bound on what any single rank is overloaded
by, whichever kernel class dominates. `×ratio` is `max / min` for
cells (a different metric, more sensitive near balance). Lower is
better; 0 = perfectly balanced. Bold rows highlight the best `max%`
for each (model, py).

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
- **`:minmax` wins or ties at every (model, py) tested.** The
  bisection method (described in the "what this suggests" section,
  now implemented) finds an α that sits at the cell/surface Pareto
  crossover. Cuts `max%` significantly at 1×4/1×8 vs both `:mix` and
  the next-best static method:

  | (model, 1×py) | best static | minmax | drop |
  |---|---|---|---|
  | OM2-1 1×8   | surface 12 | **8.6** | −28% |
  | OM2-025 1×4 | mix 14     | **8.2** | −41% |
  | OM2-025 1×8 | surface 16 | **10**  | −38% |
  | OM2-01 1×4  | mix 14     | **7.8** | −44% |
  | OM2-01 1×8  | surface 14 | **9.4** | −33% |

  At 1×2 on OM2-025/01, `:minmax` lands on the same partition as
  `:mix` (the bisection converges to α≈0.5 because the cell/surface
  curves cross right at the equal-weight point).

(A pre-2026-05-11 `:cell_obsolete` formula based on `z_center > bottom`
was checked separately; it differed from the new `immersed_cell`-based
`:cell` by at most one y-row at every (model, py) tested, so the
old-`_LB` partitions built with it are essentially equivalent.)

---

## ACCESS-OM2-1 (Nx=360, Ny=300, Nz=50)

Total wet cells = **2,707,869**, total wet columns = **69,809**.

### 1×2

![OM2-1 1×2 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x2.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny (rank 0, 1) |
|-----------|------------:|--------------:|-----:|-------:|--------------------:|
| equal      |         24  |           15  |  24  |    1.6 |            150, 150 |
| surface    |        8.4  |          0.57 | 8.4  |    1.2 |            132, 168 |
| cell       |        0.8  |           6.3 | 6.3  |    1.0 |            123, 177 |
| mix        |        4.1  |           3.2 | 4.1  |    1.1 |            127, 173 |
| **minmax** |        3.3  |           4.0 | 4.0  |    1.1 |            126, 174 |

### 1×4

![OM2-1 1×4 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x4.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny           |
|-----------|------------:|--------------:|-----:|-------:|-------------------|
| equal      |         31  |           20  |  31  |    2.9 | 75  75  75  75    |
| surface    |        9.4  |          0.86 | 9.4  |    1.5 | 70  62  65 103    |
| cell       |        0.8  |           20  |  20  |    1.0 | 67  56  58 119    |
| mix        |        5.3  |           10  |  10  |    1.2 | 68  59  61 112    |
| **minmax** |        6.5  |           6.6 | 6.6  |    1.3 | 69  60  62 109    |

### 1×8

![OM2-1 1×8 partition balance](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-1_1x8.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny                          |
|-----------|------------:|--------------:|-----:|-------:|----------------------------------|
| equal      |         69  |           51  |  69  |    5.8 | 38  38  38  38  37  37  37  37   |
| surface    |         12  |           2.2 |  12  |    1.7 | 45  25  30  32  31  34  60  43   |
| cell       |        2.5  |           51  |  51  |    1.1 | 45  22  27  29  28  30  38  81   |
| mix        |        5.9  |           22  |  22  |    1.4 | 45  23  29  30  30  31  48  64   |
| **minmax** |        8.6  |           6.6 | 8.6  |    1.6 | 45  24  30  31  31  32  56  51   |

---

## ACCESS-OM2-025 (Nx=1440, Ny=1080, Nz=50)

Total wet cells = **36,952,668**, total wet columns = **970,921**.

### 1×2

![OM2-025 1×2 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x2.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny (rank 0, 1) |
|-----------|------------:|--------------:|-----:|-------:|--------------------:|
| equal      |         32  |           20  |  32  |    1.9 |            540, 540 |
| surface    |        9.9  |          0.01 | 9.9  |    1.2 |            453, 627 |
| cell       |        0.2  |           8.8 | 8.8  |    1.0 |            415, 665 |
| **mix**    |        4.7  |           4.7 | 4.7  |    1.1 |            433, 647 |
| **minmax** |        4.7  |           4.7 | 4.7  |    1.1 |            433, 647 |

**Note the 2D vs 3D split at 1×2 cell:** `:cell` drives 3D wet-cell imbalance down to 0.2%, but the corresponding wet-column (surface) imbalance is 8.8% — same direction as `equal` (rank 1 over-loaded), just smaller magnitude. Per-column work in this simulation (notably **implicit vertical diffusion**) scales with the wet-column count, so this rank-1 surplus is exactly the imbalance the wall-clock data shows.

### 1×4

![OM2-025 1×4 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x4.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny           |
|-----------|------------:|--------------:|-----:|-------:|-------------------|
| equal      |         49  |           34  |  49  |    3.1 | 270 270 270 270   |
| surface    |         12  |          0.53 |  12  |    1.6 | 260 193 251 376   |
| cell       |        0.4  |           27  |  27  |    1.0 | 247 168 206 459   |
| mix        |        5.5  |           14  |  14  |    1.2 | 253 180 226 421   |
| **minmax** |        8.2  |           8.1 | 8.2  |    1.4 | 256 186 235 403   |

### 1×8

![OM2-025 1×8 partition balance](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/partition_balance/ACCESS-OM2-025_1x8.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny                              |
|-----------|------------:|--------------:|-----:|-------:|--------------------------------------|
| equal      |         79  |           58  |  79  |    6.4 | 135 135 135 135 135 135 135 135      |
| surface    |         16  |          0.69 |  16  |    1.7 | 175  85  87 106 109 142 232 144      |
| cell       |        1.2  |           53  |  53  |    1.0 | 175  72  78  90  98 108 173 286      |
| mix        |        6.6  |           19  |  19  |    1.3 | 175  78  82  98 104 122 241 180      |
| **minmax** |        9.1  |           10  |  10  |    1.4 | 175  80  84 102 105 128 244 162      |

---

## ACCESS-OM2-01 (Nx=3600, Ny=2700, Nz=75)

Total wet cells = **351,532,308**, total wet columns = **6,075,239**.

### 1×2

![OM2-01 1×2 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x2.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny (rank 0, 1) |
|-----------|------------:|--------------:|-----:|-------:|--------------------:|
| equal      |         30  |           19  |  30  |    1.9 |          1350, 1350 |
| surface    |        9.5  |          0.03 | 9.5  |    1.2 |          1138, 1562 |
| cell       |        0.1  |           8.5 | 8.5  |    1.0 |          1045, 1655 |
| **mix**    |        4.5  |           4.5 | 4.5  |    1.1 |          1090, 1610 |
| **minmax** |        4.5  |           4.5 | 4.5  |    1.1 |          1090, 1610 |

### 1×4

![OM2-01 1×4 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x4.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny                |
|-----------|------------:|--------------:|-----:|-------:|------------------------|
| equal      |         47  |           33  |  47  |    2.9 | 675 675 675 675        |
| surface    |         11  |          0.22 |  11  |    1.5 | 651 487 641 921        |
| cell       |        0.2  |           26  |  26  |    1.0 | 620 425 530 1125       |
| mix        |        5.3  |           14  |  14  |    1.2 | 634 456 575 1035       |
| **minmax** |        7.8  |           7.7 | 7.8  |    1.3 | 641 470 600  989       |

### 1×8

![OM2-01 1×8 partition balance](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/partition_balance/ACCESS-OM2-01_1x8.png)

| scheme    | imb%(cells) | imb%(surface) | max% | ×ratio | slab Ny                            |
|-----------|------------:|--------------:|-----:|-------:|------------------------------------|
| equal      |         77  |           58  |  77  |    5.8 | 338 338 338 338 337 337 337 337    |
| surface    |         14  |          0.33 |  14  |    1.6 | 438 213 219 268 276 365 577 344    |
| cell       |        0.5  |           47  |  47  |    1.0 | 435 185 195 230 250 280 475 650    |
| mix        |        6.5  |           17  |  17  |    1.3 | 436 198 207 249 262 313 621 414    |
| **minmax** |        9.0  |           9.4 | 9.4  |    1.4 | 437 203 212 256 267 330 614 381    |

---

## What this suggests for the LB regressions

1. **At 1×2, `:cell` already balances 3D cells nearly perfectly.** The
   wall-clock regression we see (e.g. OM2-025 H200 LB: 5m 40s vs
   baseline 5m 35s) is not because the 3D cell count is bad. The
   suspect is rank 1 owning more wet columns AND the entire tripolar
   fold halo.

2. **`:minmax` is now implemented** (`load_balance.jl:_compute_minmax_y_sizes`).
   The α-weighted load `wet_α[j] = α·cells[j]/Σcells +
   (1-α)·cols[j]/Σcols` (same form as `:mix`) but α is chosen by
   bisection on the sign of `imb%(cells) - imb%(surface)`. Tracks the
   best `max%` seen across all bisection probes, so the step-function
   integer-greedy artefact (`ic - is` is piecewise constant in α) is
   handled. 30 iterations × O(Ny) greedy ≈ ~1s on OM2-01. The
   numbers in the tables above confirm it wins (or ties `:mix`) on
   every (model, py) tested. The right next step is a benchmark sweep
   (`:cell` vs `:mix` vs `:minmax` vs `:surface`) at 1×2/1×4/1×8 on
   OM2-025 H200 to see if the static-imbalance improvement translates
   into wall-clock improvement on the actual GPU kernel mix.

3. **Tripolar-fold halo skew.** Rank py-1 always owns the
   tripolar-fold zipper boundary; its halo-exchange cost is higher per
   row than ordinary rows. None of the static-load methods see this,
   so they all give the north rank more rows just because its cell
   density is lower. Worth adding a small fold-rank penalty term as a
   follow-up to `:minmax` once we have wall-clock evidence on how
   much it matters.
