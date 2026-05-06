#!/usr/bin/env python3
"""
Fill empty PBS-side columns in scripts/runs/submissions.tsv by parsing
PBS resource logs from logs/PBS/<jobid>.gadi-pbs.OU/ER, with qstat fallback.

Canonical 21-column schema:
  timestamp, jobid, step, deps, manifest_path, case_file, git_commit,
  JOB_CHAIN, PARENT_MODEL, TIME_WINDOW, MLD_TIME_WINDOW, script,
  exit_code, queue, walltime_req, walltime_used,
  mem_req_GB, mem_used_GB, ncpus, ngpus, service_units

Data sources (priority order):
  1. PBS log (logs/PBS/<jobid>.gadi-pbs.OU/ER) — preferred, persistent
  2. qstat -fx (fallback) — when PBS log unavailable, but jobs still in PBS cache

Sentinels:
  ""    pending (queued/held/running) — neither source has final data yet
  "DRY" DRY_RUN row (not a real job)
  "?"   no source found (job completed, log not recorded, aged out of qstat)
  "-"   field unavailable from either source (e.g., service_units from qstat)

Memory columns ("mem_req_GB", "mem_used_GB") hold integer GB as bare numbers
(no unit suffix). PBS log memory values in "b"/"kb"/"mb"/"gb"/"tb" are
converted on read and rounded to the nearest GB. Pre-existing "X.XXXGB" /
"NNGB" values from older schema versions are also accepted and re-rounded.

Tolerates older row formats (12, 14, or 20 columns from earlier schema versions)
by left-padding the row to 21 fields and using the schema position to
decide which field is which. Always emits canonical 21-column output.
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
    "mem_req_GB", "mem_used_GB", "ncpus", "ngpus", "service_units",
]
N_COLS = len(HEADER)

# Indices for the 9 PBS-side columns (exit_code through service_units).
IDX_EXIT, IDX_QUEUE, IDX_WREQ, IDX_WUSE, IDX_MREQ, IDX_MUSE, IDX_NCPUS, IDX_NGPUS, IDX_SVCUNITS = range(12, 21)


def to_gb(v):
    """Normalise PBS memory strings to bare integer GB (no unit suffix). Sentinels pass through."""
    if v in ("", "-", "?"):
        return v
    m = re.match(r"^([\d.]+)\s*([a-zA-Z]*)$", v)
    if not m:
        return v
    n = float(m.group(1))
    suf = m.group(2).lower()
    factor = {
        "":   1,                  # bare number = already GB (idempotent re-rounding)
        "b":  1 / (1024 ** 3),
        "kb": 1 / (1024 ** 2),
        "mb": 1 / 1024,
        "gb": 1,
        "tb": 1024,
    }.get(suf)
    if factor is None:
        return v
    return f"{round(n * factor)}"


def pbs_log_path(jobid):
    """Construct path to PBS output log file for a jobid."""
    pbs_dir = REPO_ROOT / "logs/PBS"
    # Try .OU (output) first, then .ER (error)
    ou_path = pbs_dir / f"{jobid}.OU"
    if ou_path.is_file():
        return ou_path
    er_path = pbs_dir / f"{jobid}.ER"
    if er_path.is_file():
        return er_path
    return None


def extract_gpu_resources(lines):
    """Extract GPU_RESOURCES from the beginning of the log file."""
    for line in lines[:50]:  # Check first 50 lines for GPU_RESOURCES
        if "GPU_RESOURCES=" in line:
            # "GPU_RESOURCES=gpuhopper" or similar
            parts = line.split("GPU_RESOURCES=")
            if len(parts) > 1:
                val = parts[1].strip()
                return val
    return None


def parse_pbs_log(log_path):
    """Parse PBS resource usage and metadata from stdout/stderr log file.

    Returns tuple of (exit_code, queue, walltime_req, walltime_used, mem_req_GB, mem_used_GB, ncpus, ngpus, service_units)
    or None if log doesn't contain resource section.

    Queue is inferred from GPU_RESOURCES when ngpus > 0.
    """
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return None

    # Find the resource usage section (last occurrence of "Resource Usage")
    resource_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Resource Usage" in lines[i]:
            resource_start = i
            break

    if resource_start < 0:
        return None

    # Parse the resource section (next ~15 lines after "Resource Usage")
    exit_code = "?"
    ncpus = "-"
    ngpus = "-"
    mem_req = "-"
    mem_use = "-"
    walltime_req = "-"
    walltime_use = "-"
    service_units = "-"

    for line in lines[resource_start:resource_start + 15]:
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # Parse lines with "Key: Value" or "Key: Value1    Key2: Value2" format

        if "Exit Status:" in line:
            # "Exit Status:        0" or "Exit Status:        -29 (Job failed due to exceeding walltime)"
            parts = line.split("Exit Status:")
            if len(parts) > 1:
                val = parts[1].strip().split()[0]
                exit_code = val

        elif "Service Units:" in line:
            # "Service Units:      21.93"
            parts = line.split("Service Units:")
            if len(parts) > 1:
                val = parts[1].strip().split()[0]
                service_units = val

        elif "NCPUs Requested:" in line:
            # "NCPUs Requested:    12                  CPU Time Used: 00:12:37"
            parts = line.split("NCPUs Requested:")
            if len(parts) > 1:
                val = parts[1].strip().split()[0]
                ncpus = val

        elif "NGPUs Requested:" in line:
            # "NGPUs Requested:    1                 GPU Utilisation: 100%"
            parts = line.split("NGPUs Requested:")
            if len(parts) > 1:
                val = parts[1].strip().split()[0]
                ngpus = val

        elif "Memory Requested:" in line:
            # "Memory Requested:   256.0GB               Memory Used: 24.16GB"
            parts = line.split("Memory Requested:")
            if len(parts) > 1:
                val = parts[1].split("Memory Used:")[0].strip()
                mem_req = to_gb(val)
                # Also grab Memory Used from same line if present
                if "Memory Used:" in line:
                    val_use = line.split("Memory Used:")[1].strip()
                    mem_use = to_gb(val_use)

        elif "Walltime Requested:" in line:
            # "Walltime Requested: 00:30:00            Walltime Used: 00:14:37"
            parts = line.split("Walltime Requested:")
            if len(parts) > 1:
                val = parts[1].split("Walltime Used:")[0].strip()
                walltime_req = val
                # Also grab Walltime Used from same line if present
                if "Walltime Used:" in line:
                    val_use = line.split("Walltime Used:")[1].strip()
                    walltime_use = val_use

    # Infer queue from GPU_RESOURCES if GPUs are used
    queue = "-"
    if ngpus not in ("-", "0"):
        # Extract GPU_RESOURCES from the top of the log
        gpu_resources = extract_gpu_resources(lines)
        if gpu_resources:
            queue = gpu_resources

    return (exit_code, queue, walltime_req, walltime_use, mem_req, mem_use, ncpus, ngpus, service_units)


def qstat_fx(jobid):
    """Run `qstat -fx <jobid>`; return parsed key→value or None if not found.

    Fallback for when PBS logs are unavailable.
    """
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


def parse_qstat_fallback(info):
    """Extract PBS-side fields from a parsed qstat -fx dict (fallback only).

    Used when PBS logs are unavailable. Returns 9-tuple to match parse_pbs_log.
    """
    state = info.get("job_state", "")
    if state != "F":
        # Still pending — return empties
        return ("", "-", "", "", "-", "-", "-", "-", "-")
    exit_code = info.get("Exit_status", "?") or "?"
    queue = info.get("queue", "-") or "-"
    wreq = info.get("Resource_List.walltime", "-") or "-"
    wuse = info.get("resources_used.walltime", "-") or "-"
    mreq = to_gb(info.get("Resource_List.mem", "-") or "-")
    muse = to_gb(info.get("resources_used.mem", "-") or "-")
    ncpus = info.get("Resource_List.ncpus", "-") or "-"
    ngpus = info.get("Resource_List.ngpus", "0") or "0"
    # qstat doesn't provide service_units
    return (exit_code, queue, wreq, wuse, mreq, muse, ncpus, ngpus, "-")


def get_pbs_fields(jobid):
    """Get PBS-side fields from PBS log file or qstat (fallback).

    Returns tuple (exit_code, queue, walltime_req, walltime_used, mem_req, mem_used, ncpus, ngpus, service_units)
    or None if not found in either source.

    Priority: PBS log > qstat fallback
    """
    # Try PBS log first
    log_path = pbs_log_path(jobid)
    if log_path is not None:
        result = parse_pbs_log(log_path)
        if result is not None:
            return result

    # Fallback to qstat if PBS log unavailable
    info = qstat_fx(jobid)
    if info is None:
        return None
    return parse_qstat_fallback(info)


def normalise_row(row):
    """Pad/truncate to N_COLS, normalising older row formats.

    Older 14-col rows had only (exit_code, walltime_used) at positions 12, 13.
    Older 20-col rows didn't have service_units at the end.
    If we detect a walltime-shaped value at position 13 in a row of length
    ≤ 14, shift it into walltime_used (position 15) and clear queue (13).
    """
    if len(row) < N_COLS:
        if len(row) <= 14 and len(row) >= 14:
            # Old 14-col format: exit_code at 12, walltime_used at 13.
            old_exit = row[12]
            old_wuse = row[13] if len(row) > 13 else ""
            row = row[:12] + [old_exit, "", "", old_wuse] + [""] * 5
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
            row[IDX_EXIT:IDX_SVCUNITS + 1] = ["DRY"] + ["-"] * 8
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

        pbs_fields = get_pbs_fields(jobid)
        if pbs_fields is None:
            # No PBS log found — job still pending or log not yet written.
            counts["unknown"] += 1
            for i in range(IDX_EXIT, IDX_SVCUNITS + 1):
                if not row[i]:
                    row[i] = "?"
            out_rows.append(row)
            continue

        counts["filled"] += 1
        # Fill all PBS fields except queue (which was populated at submission time)
        pbs_exit, pbs_queue, pbs_wreq, pbs_wuse, pbs_mreq, pbs_muse, pbs_ncpus, pbs_ngpus, pbs_svc = pbs_fields
        row[IDX_EXIT] = pbs_exit
        # Keep existing queue value (populated at submission time)
        row[IDX_WREQ] = pbs_wreq
        row[IDX_WUSE] = pbs_wuse
        row[IDX_MREQ] = pbs_mreq
        row[IDX_MUSE] = pbs_muse
        row[IDX_NCPUS] = pbs_ncpus
        row[IDX_NGPUS] = pbs_ngpus
        row[IDX_SVCUNITS] = pbs_svc
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
