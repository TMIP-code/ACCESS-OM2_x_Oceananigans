#!/usr/bin/env python3
"""
Plot simulated time (yr) vs wall time (s) for each rank, and report the
Julia-internal simulation wall time (excludes startup, package loading,
model setup, and any benchmark warmup).

Supports two log types:
  - `run_1year_benchmark.jl`: counts only lines after the "Benchmark: 1-year
    simulation" marker (skips warmup).
  - `run_1year.jl` / `run_10years.jl` / `run_100years.jl` / `run_long.jl`:
    no benchmark marker; the whole file is scanned.

Parses progress callback lines of the form:
    [ Info:   iter: 487, time: 0.083 yr, wall: 4.8 seconds        (benchmark)
    [ Info:   sim iter: 0487, time: 0.083 yr, ..., wall: 56.9 seconds   (run_1year)
And end-of-run markers:
    Simulation is stopping after running for 1.801 minutes.

Wall-time values in seconds / minutes / hours are all normalised to seconds.

Assumes ranks consistently print in the same order, so line i in the parsed
section is grouped to rank (i % N) where N is auto-detected from the number
of `Simulation is stopping…` lines (one per rank, non-TB) or `iter: 0` lines.

Usage:
    python scripts/plotting/plot_simtime_vs_walltime.py LOG_FILE [LOG_FILE ...]
    python scripts/plotting/plot_simtime_vs_walltime.py LOG_FILE --output plot.png
    python scripts/plotting/plot_simtime_vs_walltime.py LOG_FILE --no-plot   # max wall only

Notes:
- +TB runs have the per-month progress callback disabled (TBLOCKING > 0), so
  the plot will be empty. The script reports this and falls back to extracting
  the wall time from "Simulation is stopping after running for X seconds"
  lines, which are still printed at end-of-run for every config.
"""

import argparse
import re
import sys
from pathlib import Path

# Progress callback lines. Three flavours:
#   benchmark non-TB: "iter: 487, time: 0.083 yr, wall: 4.8 seconds"
#   benchmark TB:     "batch: 40/487, time: 0.082 yr, wall: 84.2 seconds"
#   run_1year.jl:     "sim iter: 0487, time: 0.083 yr, ..., wall: 56.928 seconds"
# Units in the run_1year log promote to minutes/hours when wall ≥ 60s, so accept all three.
PROGRESS_RE = re.compile(
    r"(?:iter:|batch:)\s*(?P<iter>\d+)(?:/\d+)?,\s*time:\s*(?P<time>[\d.]+)\s*yr,"
    r"(?:[^,]*,)*?\s*wall:\s*(?P<wall>[\d.]+)\s*(?P<unit>seconds|minutes|hours)"
)
# "Simulation is stopping after running for 29.593 seconds." (benchmark) or
# "... for 1.801 minutes." (run_1year). One per rank at end-of-run (non-TB).
STOP_RE = re.compile(
    r"Simulation is stopping after running for\s+(?P<wall>[\d.]+)\s*(?P<unit>seconds|minutes|hours)"
)
# "elapsed_seconds = 105.0" — printed by every rank in the "Benchmark complete" block.
ELAPSED_RE = re.compile(r"elapsed_seconds\s*=\s*(?P<wall>[\d.]+)")
BENCHMARK_START_MARKER = "Benchmark: 1-year simulation"
UNIT_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


def parse_log(path):
    """Return (progress_records, stop_wall_seconds, elapsed_walls) for the run.

    For `run_1year_benchmark.jl` logs the parser only counts lines after the
    "Benchmark: 1-year simulation" marker (skips the warmup phase). For regular
    `run_1year.jl` logs the marker is absent, in which case the parser scans
    the whole file.

    progress_records is a list of (iter, sim_time_yr, wall_seconds), normalised
    to seconds regardless of the unit printed in the log.
    """
    # Decide upfront whether this is a benchmark log; if not, parse the whole file.
    has_benchmark_marker = False
    with path.open() as f:
        for line in f:
            if BENCHMARK_START_MARKER in line:
                has_benchmark_marker = True
                break

    records = []
    stop_walls = []
    elapsed_walls = []
    in_section = not has_benchmark_marker  # if no marker, count from start
    with path.open() as f:
        for line in f:
            if has_benchmark_marker and BENCHMARK_START_MARKER in line:
                in_section = True
                continue
            if not in_section:
                continue
            m = PROGRESS_RE.search(line)
            if m:
                wall_s = float(m["wall"]) * UNIT_TO_SECONDS[m["unit"]]
                records.append((int(m["iter"]), float(m["time"]), wall_s))
                continue
            m = STOP_RE.search(line)
            if m:
                stop_walls.append(float(m["wall"]) * UNIT_TO_SECONDS[m["unit"]])
                continue
            m = ELAPSED_RE.search(line)
            if m:
                elapsed_walls.append(float(m["wall"]))
    return records, stop_walls, elapsed_walls


def detect_nranks(records, stop_walls, elapsed_walls):
    """Auto-detect rank count.

    Preferred: count of "Simulation is stopping..." lines (one per rank, non-TB only).
    Fallback 1: count of "elapsed_seconds = ..." lines (one per rank, all configs).
    Fallback 2: count consecutive iter=0 records at the start of the benchmark.
    """
    if stop_walls:
        return len(stop_walls)
    if elapsed_walls:
        return len(elapsed_walls)
    n = 0
    for r in records:
        if r[0] == 0:
            n += 1
        else:
            break
    return max(n, 1)


def split_by_rank(records, nranks):
    """Group records by index modulo nranks, assuming consistent print order."""
    per_rank = [[] for _ in range(nranks)]
    for i, r in enumerate(records):
        per_rank[i % nranks].append(r)
    return per_rank


def max_end_of_run_wall(records, stop_walls, elapsed_walls):
    """Max wall time across ranks at the end of the 1-year run.

    Prefer "Simulation is stopping" lines (non-TB).
    Fallback 1: "elapsed_seconds" from Benchmark complete (TB and non-TB).
    Fallback 2: last progress record per rank (iter at the final stop_time).
    """
    if stop_walls:
        return max(stop_walls)
    if elapsed_walls:
        return max(elapsed_walls)
    if not records:
        return None
    final_iter = max(r[0] for r in records)
    finals = [r[2] for r in records if r[0] == final_iter]
    return max(finals) if finals else None


def plot(per_rank, output, title=None):
    import matplotlib.pyplot as plt  # imported lazily so --no-plot doesn't need matplotlib

    fig, ax = plt.subplots(figsize=(8, 5))
    for rank_idx, recs in enumerate(per_rank):
        if not recs:
            continue
        walls = [r[2] for r in recs]
        times = [r[1] for r in recs]
        ax.plot(walls, times, marker=".", label=f"rank {rank_idx}")
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Simulated time (yr)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def process_one(path, plot_path):
    records, stop_walls, elapsed_walls = parse_log(path)
    nranks = detect_nranks(records, stop_walls, elapsed_walls)
    max_wall = max_end_of_run_wall(records, stop_walls, elapsed_walls)

    info = {
        "log": str(path),
        "nranks": nranks,
        "max_wall_s": max_wall,
        "n_progress_records": len(records),
        "stop_walls": stop_walls,
    }

    if plot_path is not None and records:
        per_rank = split_by_rank(records, nranks)
        plot(per_rank, plot_path, title=path.name)
        info["plot"] = str(plot_path)
    elif plot_path is not None:
        info["plot"] = "skipped (no progress callbacks in benchmark phase)"

    return info


def fmt_seconds(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Path(s) to Julia log file(s)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path (when one log given). Default: <logname>.png alongside the log.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print max wall times only; do not produce plots.",
    )
    args = parser.parse_args(argv)

    if args.output is not None and len(args.logs) > 1:
        print("--output only valid for a single log file", file=sys.stderr)
        return 2

    for log_path in args.logs:
        if not log_path.is_file():
            print(f"ERROR: file not found: {log_path}", file=sys.stderr)
            continue
        plot_path = None
        if not args.no_plot:
            plot_path = args.output if args.output else log_path.with_suffix(".simtime_vs_walltime.png")
        info = process_one(log_path, plot_path)
        max_wall = info["max_wall_s"]
        max_wall_str = fmt_seconds(max_wall) if max_wall is not None else "n/a"
        print(
            f"{log_path.name}: nranks={info['nranks']} max_wall={max_wall_str} "
            f"({info['n_progress_records']} progress records)"
        )
        if "plot" in info:
            print(f"  -> {info['plot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
