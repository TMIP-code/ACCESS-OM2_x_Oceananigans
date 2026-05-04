#!/usr/bin/env python3
"""
Fill empty PBS-side columns in scripts/runs/submissions.tsv by querying
`qstat -fx <jobid>` for each row missing them.

Canonical 20-column schema:
  timestamp, jobid, step, deps, manifest_path, case_file, git_commit,
  JOB_CHAIN, PARENT_MODEL, TIME_WINDOW, MLD_TIME_WINDOW, script,
  exit_code, queue, walltime_req, walltime_used,
  mem_req, mem_used, ncpus, ngpus

Sentinels:
  ""    pending (queued/held/running) — next reconcile picks it up
  "DRY" DRY_RUN row (not a real job)
  "?"   finished but Exit_status missing, or aged out of qstat
  "-"   field unavailable from qstat

Memory fields are normalised to GB (e.g. "47.000GB"); other units in the
qstat output ("b", "kb", "mb", "tb") are converted on read. Pre-existing
"X.XXXGB" values pass through unchanged.

Tolerates older row formats (12 or 14 columns from earlier schema versions)
by left-padding the row to 20 fields and using the schema position to
decide which field is which. Always emits canonical 20-column output.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans")
DEFAULT_TSV = REPO_ROOT / "scripts/runs/submissions.tsv"

HEADER = [
    "timestamp", "jobid", "step", "deps", "manifest_path", "case_file",
    "git_commit", "JOB_CHAIN", "PARENT_MODEL", "TIME_WINDOW",
    "MLD_TIME_WINDOW", "script",
    "exit_code", "queue", "walltime_req", "walltime_used",
    "mem_req", "mem_used", "ncpus", "ngpus",
]
N_COLS = len(HEADER)

# Indices for the 8 PBS-side columns.
IDX_EXIT, IDX_QUEUE, IDX_WREQ, IDX_WUSE, IDX_MREQ, IDX_MUSE, IDX_NCPUS, IDX_NGPUS = range(12, 20)


def to_gb(v):
    """Normalise PBS memory strings to '%.3fGB'. Sentinels pass through."""
    if v in ("", "-", "?"):
        return v
    m = re.match(r"^([\d.]+)\s*([a-zA-Z]*)$", v)
    if not m:
        return v
    n = float(m.group(1))
    suf = m.group(2).lower()
    factor = {
        "":   1 / (1024 ** 3),
        "b":  1 / (1024 ** 3),
        "kb": 1 / (1024 ** 2),
        "mb": 1 / 1024,
        "gb": 1,
        "tb": 1024,
    }.get(suf)
    if factor is None:
        return v
    return f"{n * factor:.3f}GB"


def qstat_fx(jobid):
    """Run `qstat -fx <jobid>`; return parsed key→value or None if not found."""
    try:
        out = subprocess.run(
            ["qstat", "-fx", jobid],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    fields = {}
    cur_key = None
    for line in out.stdout.splitlines():
        # qstat -f wraps long values onto continuation lines starting with whitespace.
        if line.startswith("\t") or line.startswith("        "):
            if cur_key:
                fields[cur_key] += line.strip()
            continue
        if " = " in line:
            k, v = line.split(" = ", 1)
            cur_key = k.strip()
            fields[cur_key] = v.strip()
    return fields


def parse_qstat(info):
    """Extract the 8 PBS-side fields from a parsed qstat -fx dict."""
    state = info.get("job_state", "")
    if state != "F":
        # Still pending — caller will emit empties for next pass.
        return ("", "", "", "", "", "", "", "")
    exit_code = info.get("Exit_status", "?") or "?"
    queue = info.get("queue", "-") or "-"
    wreq = info.get("Resource_List.walltime", "-") or "-"
    wuse = info.get("resources_used.walltime", "-") or "-"
    mreq = to_gb(info.get("Resource_List.mem", "-") or "-")
    muse = to_gb(info.get("resources_used.mem", "-") or "-")
    ncpus = info.get("Resource_List.ncpus", "-") or "-"
    ngpus = info.get("Resource_List.ngpus", "0") or "0"
    return (exit_code, queue, wreq, wuse, mreq, muse, ncpus, ngpus)


def normalise_row(row):
    """Pad/truncate to N_COLS, normalising older 14-col rows.

    Older 14-col rows had only (exit_code, walltime_used) at positions 12, 13.
    If we detect a walltime-shaped value at position 13 in a row of length
    ≤ 14, shift it into walltime_used (position 15) and clear queue (13).
    """
    if len(row) < N_COLS:
        if len(row) <= 14 and len(row) >= 14:
            # Old 14-col format: exit_code at 12, walltime_used at 13.
            old_exit = row[12]
            old_wuse = row[13] if len(row) > 13 else ""
            row = row[:12] + [old_exit, "", "", old_wuse] + [""] * 4
        else:
            row = row + [""] * (N_COLS - len(row))
    elif len(row) > N_COLS:
        row = row[:N_COLS]
    return row


def main():
    tsv_path = Path(os.environ.get("SUBMISSIONS_TSV", str(DEFAULT_TSV)))
    if not tsv_path.is_file():
        print(f"TSV not found: {tsv_path}", file=sys.stderr)
        return 1

    with tsv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip blank trailing lines.
    while lines and not lines[-1].strip():
        lines.pop()

    counts = {"filled": 0, "pending": 0, "dry": 0, "kept": 0, "unknown": 0}

    out_rows = [list(HEADER)]

    for line in lines:
        # Strip only the trailing newline; preserve intentional empty fields.
        row = line.rstrip("\n").split("\t")
        if not row or row == [""]:
            continue
        if row[0] == "timestamp":
            continue  # canonical header re-emitted at top
        row = normalise_row(row)

        jobid = row[1]

        # DRY runs.
        if jobid.startswith("DRY_RUN_"):
            counts["dry"] += 1
            row[IDX_EXIT:IDX_NGPUS + 1] = ["DRY"] + ["-"] * 7
            out_rows.append(row)
            continue

        exit_code = row[IDX_EXIT]
        ncpus = row[IDX_NCPUS]
        already_filled = (
            exit_code not in ("", "?")
            and ncpus not in ("", "?")
        )

        if already_filled:
            counts["kept"] += 1
            # Renormalise mem fields (idempotent for already-GB values).
            row[IDX_MREQ] = to_gb(row[IDX_MREQ])
            row[IDX_MUSE] = to_gb(row[IDX_MUSE])
            out_rows.append(row)
            continue

        info = qstat_fx(jobid)
        if info is None:
            # Aged out / not found. Keep any prior values, mark missing as '?'.
            counts["unknown"] += 1
            for i in range(IDX_EXIT, IDX_NGPUS + 1):
                if not row[i]:
                    row[i] = "?"
            out_rows.append(row)
            continue

        if info.get("job_state") != "F":
            counts["pending"] += 1
            row[IDX_EXIT:IDX_NGPUS + 1] = [""] * 8
            out_rows.append(row)
            continue

        counts["filled"] += 1
        row[IDX_EXIT:IDX_NGPUS + 1] = list(parse_qstat(info))
        out_rows.append(row)

    # Atomic rewrite: tmpfile + rename.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=tsv_path.parent, prefix=tsv_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write("\t".join(r) + "\n")
        shutil.move(tmp_path, tsv_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(
        f"reconcile: filled={counts['filled']}  pending={counts['pending']}  "
        f"dry={counts['dry']}  already={counts['kept']}  unknown={counts['unknown']}"
    )
    print(f"TSV: {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
