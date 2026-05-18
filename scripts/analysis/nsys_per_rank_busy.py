#!/usr/bin/env python3
"""
Per-rank GPU-busy time from a set of `.nsys-rep` profile files.

Run on a compute node (nsys export is memory-hungry; login nodes get
killed). Wrapped by `scripts/tests/run_nsys_per_rank_busy.sh`.

For each input `.nsys-rep`:
  1. `nsys export --type=sqlite` next to it (cached if already present).
  2. Sum kernel duration → "GPU-busy time" for that rank.
  3. Take min(start), max(end) of all kernels → "capture wall".

Then print a per-rank table + min/max/ratio across ranks. The
GPU-busy column is the ground truth for "how much actual compute
this rank had to do" — independent of how long it spent waiting in
MPI_Waitall (which is just `wall - busy`, and is anti-correlated
with busy across ranks).

Usage (direct, from a compute node):
    module load cuda/12.9.0
    python3 scripts/analysis/nsys_per_rank_busy.py \\
        logs/julia/.../*_168522455.gadi-pbs_profile_*_rank*.nsys-rep \\
        --csv /tmp/per_rank.csv

Usage (PBS):
    NSYS_GLOB='logs/julia/.../*_168522455.gadi-pbs_profile_*_rank*.nsys-rep' \\
        qsub -v NSYS_GLOB scripts/tests/run_nsys_per_rank_busy.sh
"""

import argparse
import csv
import re
import sqlite3
import subprocess
import sys
from pathlib import Path


# Table names have varied across nsys releases. Try in order.
KERNEL_TABLE_CANDIDATES = [
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_KERNEL",
    "KERNEL",
    "GPU_KERNEL_EVENT",
]


def find_kernel_table(con: sqlite3.Connection) -> str:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    for candidate in KERNEL_TABLE_CANDIDATES:
        if candidate in tables:
            return candidate
    raise RuntimeError(
        f"None of {KERNEL_TABLE_CANDIDATES} found in SQLite. "
        f"Available tables: {sorted(tables)[:20]}..."
    )


def export_sqlite(repfile: Path) -> Path:
    """Export <foo>.nsys-rep to <foo>.sqlite (cached)."""
    sqlite = repfile.with_suffix(".sqlite")
    if sqlite.exists() and sqlite.stat().st_mtime >= repfile.stat().st_mtime:
        return sqlite
    print(f"  exporting {repfile.name} → SQLite ...", file=sys.stderr, flush=True)
    subprocess.run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         "--output", str(sqlite), str(repfile)],
        check=True,
    )
    return sqlite


def per_file_stats(sqlite: Path):
    """Return (n_kernels, total_kernel_ns, capture_start_ns, capture_end_ns)."""
    con = sqlite3.connect(str(sqlite))
    try:
        tbl = find_kernel_table(con)
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*), SUM(end - start), MIN(start), MAX(end) FROM {tbl}")
        n, total, s, e = cur.fetchone()
        return int(n or 0), int(total or 0), s, e
    finally:
        con.close()


def rank_from_name(p: Path) -> int:
    m = re.search(r"_rank(\d+)\.nsys-rep$", p.name)
    return int(m.group(1)) if m else -1


def fmt_seconds(ns: int) -> str:
    s = ns / 1e9
    if s >= 60:
        return f"{int(s // 60)}m{s % 60:5.2f}s"
    return f"{s:7.3f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="+", help="*.nsys-rep files (one per rank)")
    ap.add_argument("--csv", help="Optional CSV output path")
    args = ap.parse_args()

    rows = []
    for r in args.reps:
        rep = Path(r)
        if not rep.exists():
            print(f"WARN: missing {rep}", file=sys.stderr)
            continue
        sqlite = export_sqlite(rep)
        n_kernels, kernel_ns, s, e = per_file_stats(sqlite)
        wall_ns = (e - s) if (s is not None and e is not None) else 0
        rows.append({
            "rank": rank_from_name(rep),
            "n_kernels": n_kernels,
            "kernel_ns": kernel_ns,
            "wall_ns": wall_ns,
            "name": rep.name,
        })

    rows.sort(key=lambda r: r["rank"])

    # Per-rank table
    hdr = f"{'rank':>4}  {'n_kernels':>9}  {'kernel_busy':>12}  {'capture_wall':>13}  {'util%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        util = 100 * r["kernel_ns"] / r["wall_ns"] if r["wall_ns"] else 0.0
        print(
            f"{r['rank']:>4}  {r['n_kernels']:>9d}  "
            f"{fmt_seconds(r['kernel_ns']):>12}  {fmt_seconds(r['wall_ns']):>13}  "
            f"{util:>5.1f}%"
        )

    # Summary
    if rows:
        ks = [r["kernel_ns"] for r in rows]
        mean = sum(ks) / len(ks)
        mx = max(ks)
        mn = min(ks)
        imb_pct = 100 * (mx - mean) / mean if mean > 0 else 0.0
        ratio = mx / mn if mn > 0 else float("inf")
        print("-" * len(hdr))
        print(f"mean kernel_busy:     {fmt_seconds(int(mean))}")
        print(f"max  kernel_busy:     {fmt_seconds(int(mx))}    (rank {ks.index(mx)})")
        print(f"min  kernel_busy:     {fmt_seconds(int(mn))}    (rank {ks.index(mn)})")
        print(f"imbalance% (max-mean)/mean: {imb_pct:5.1f}%")
        print(f"ratio max/min:        {ratio:.3f}×")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["rank", "n_kernels", "kernel_ns", "wall_ns", "util_pct", "name"],
            )
            w.writeheader()
            for r in rows:
                util = 100 * r["kernel_ns"] / r["wall_ns"] if r["wall_ns"] else 0.0
                w.writerow({**r, "util_pct": round(util, 2)})
        print(f"\nCSV written: {args.csv}")


if __name__ == "__main__":
    main()
