#!/usr/bin/env python3
"""
Rebuild scripts/runs/submissions.tsv from per-run manifests.

Walks outputs/**/manifests/*.toml, extracts one row per [[jobs]] entry,
emits the 12 stable submission fields (PBS-side fields left empty for
reconcile_submissions.py to fill).

Use this when the TSV index is lost or corrupted; the manifests are the
source of truth.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path("/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans")
TSV = REPO_ROOT / "scripts/runs/submissions.tsv"
OUTPUTS = REPO_ROOT / "outputs"

HEADER = [
    "timestamp", "jobid", "step", "deps", "manifest_path", "case_file",
    "git_commit", "JOB_CHAIN", "PARENT_MODEL", "TIME_WINDOW",
    "MLD_TIME_WINDOW", "script",
    "exit_code", "queue", "walltime_req", "walltime_used",
    "mem_req", "mem_used", "ncpus", "ngpus",
]


_RE_KV = re.compile(r'^([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$')


def parse_manifest(path):
    """Parse our constrained manifest format: [section] / [[array]] / key = "value".
    Returns dict with sections as keys, where [[array]] sections become lists.
    """
    data = {}
    cur_section = None
    cur_dict = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[[") and line.endswith("]]"):
                cur_section = line[2:-2].strip()
                data.setdefault(cur_section, [])
                cur_dict = {}
                data[cur_section].append(cur_dict)
                continue
            if line.startswith("[") and line.endswith("]"):
                cur_section = line[1:-1].strip()
                cur_dict = {}
                data[cur_section] = cur_dict
                continue
            m = _RE_KV.match(line)
            if not m or cur_dict is None:
                continue
            k, v = m.group(1), m.group(2)
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            elif v.startswith('[') and v.endswith(']'):
                inner = v[1:-1].strip()
                v = [x.strip().strip('"') for x in inner.split(",") if x.strip()]
            cur_dict[k] = v
    return data


def main() -> int:
    rows = []
    n_manifests = 0
    for mpath in sorted(OUTPUTS.glob("**/manifests/*.toml")):
        data = parse_manifest(mpath)
        n_manifests += 1
        meta = data.get("meta", {})
        git = data.get("git", {})
        env = data.get("env", {})
        jobs = data.get("jobs", [])

        ts = meta.get("timestamp", "")
        case_file = meta.get("case_file", "") or ""
        gc = git.get("commit", "")
        jc = env.get("JOB_CHAIN", "")
        pm = env.get("PARENT_MODEL", "")
        tw = env.get("TIME_WINDOW", "")
        mtw = env.get("MLD_TIME_WINDOW", "") if "MLD_TIME_WINDOW" in env else ""

        for job in jobs:
            step = job.get("step", "")
            jobid = job.get("jobid", "")
            script = job.get("script", "")
            deps_list = job.get("deps", [])
            deps = ":".join(deps_list) if isinstance(deps_list, list) else str(deps_list)
            row = [
                ts, jobid, step, deps, str(mpath), case_file,
                gc, jc, pm, tw, mtw, script,
                "", "", "", "", "", "", "", "",
            ]
            rows.append(row)

    rows.sort(key=lambda r: (r[0], r[1]))

    with TSV.open("w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"rebuilt {len(rows)} rows from {n_manifests} manifests")
    print(f"TSV: {TSV}")
    print("Run scripts/runs/reconcile_submissions.sh to fill PBS-side fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
