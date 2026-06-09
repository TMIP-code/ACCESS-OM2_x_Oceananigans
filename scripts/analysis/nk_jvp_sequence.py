#!/usr/bin/env python3
"""Track the interleaving of Φ! and G! calls across Newton–Krylov (NK) solve logs.

Background
----------
In our periodic NK solves (`src/solve_periodic_NK.jl`) the forward map Φ! is the
GPU integrator that advances the age tracer one year. It is invoked in two roles:

  * once per **Newton iteration** to evaluate the residual G(x) = Φ(x) − x
    (logged as `Φ! call #k starting (source_rate=1.0)` and followed by a
    `G! residual` block holding the drift norms); and
  * once per **GMRES inner iteration** as a Jacobian–vector product (JVP), using
    the linearised tracer linΦ! with no source
    (logged as `Φ! call #k starting (source_rate=0.0)`).

So every `G! residual` block opens a new Newton iteration, and the JVPs charged
to that iteration are the `source_rate=0.0` Φ! calls that follow it, up to the
next `G! residual`. The absolute Φ! counter printed in the log is *not* relied
upon (it has changed format over time and resets across restarts) — iterations
are numbered sequentially by the `G! residual` blocks actually seen.

This script answers: for each parent model (PM), how does the JVP count per
Newton iteration evolve as the residual drops — does it decrease as the solve
converges?

Job selection
-------------
NK jobs are taken from `scripts/runs/submissions.tsv` (steps matching NK), most
recent N per PM (default 10), then each jobid is matched to its log under
`logs/julia/<PM>/<EXP>/<TW>/periodic/NK/*<jobid>*.log`. Pass --logs to instead
parse explicit log files directly.

Usage
-----
  scripts/analysis/nk_jvp_sequence.py                 # default: OM2-1 & OM2-025, last 10 each
  scripts/analysis/nk_jvp_sequence.py --pm ACCESS-OM2-1 --last 5
  scripts/analysis/nk_jvp_sequence.py --tw 1968-1977 1999-2008
  scripts/analysis/nk_jvp_sequence.py --csv out.csv
  scripts/analysis/nk_jvp_sequence.py --logs path/to/a.log path/to/b.log
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMISSIONS = os.path.join(REPO, "scripts", "runs", "submissions.tsv")

# Φ! call start, capturing the source_rate (1.0 => residual/G! eval, 0.0 => JVP).
RE_PHI = re.compile(r"Φ! call #(\d+) starting \(source_rate=([0-9.eE+-]+)\)")
# Opening line of a residual block; numbered sequentially as Newton iterations.
RE_GRES = re.compile(r"G! residual")
RE_DRIFT = {
    "vol_rms_drift_years": re.compile(r"vol_rms_drift_years\s*=\s*([0-9.eE+-]+)"),
    "max_drift_years": re.compile(r"max_drift_years\s*=\s*([0-9.eE+-]+)"),
    "mean_drift_years": re.compile(r"mean_drift_years\s*=\s*([0-9.eE+-]+)"),
}
JOBID_RE = re.compile(r"(\d+)\.gadi-pbs")


def parse_log(path):
    """Return list of Newton-iteration dicts for one log.

    Each dict: {newton, jvps, residual_phi, vol_rms_drift_years, max/mean,
                first_residual} where `jvps` is the count of source_rate=0.0 Φ!
    calls charged to that Newton iteration. A leading group of JVPs before any
    `G! residual` (shouldn't normally happen) is dropped with a note.
    """
    iters = []
    cur = None
    pre_jvps = 0  # JVPs seen before the first G! residual (anomaly guard)
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_PHI.search(line)
            if m:
                sr = float(m.group(2))
                if abs(sr) < 1e-12:  # source_rate == 0 -> JVP
                    if cur is None:
                        pre_jvps += 1
                    else:
                        cur["jvps"] += 1
                # source_rate != 0 (== 1.0): residual eval; the following
                # `G! residual` block opens the iteration, so nothing to do here.
                continue
            if RE_GRES.search(line):
                cur = {
                    "newton": len(iters) + 1,
                    "jvps": 0,
                    "vol_rms_drift_years": None,
                    "max_drift_years": None,
                    "mean_drift_years": None,
                }
                iters.append(cur)
                continue
            if cur is not None:
                for key, rx in RE_DRIFT.items():
                    if cur[key] is None:
                        dm = rx.search(line)
                        if dm:
                            cur[key] = float(dm.group(1))
    return iters, pre_jvps


def load_jobs_from_tsv(pms, tws):
    """Map each PM -> chronological list of (jobid, tw, script) for *all* its NK
    jobs in submissions.tsv, deduped by jobid (last occurrence wins, order kept)."""
    if not os.path.exists(SUBMISSIONS):
        sys.exit(f"submissions.tsv not found at {SUBMISSIONS}")
    per_pm = {pm: OrderedDict() for pm in pms}
    with open(SUBMISSIONS, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pm = row.get("PARENT_MODEL", "")
            step = row.get("step", "")
            jobid = row.get("jobid", "")
            tw = row.get("TIME_WINDOW", "")
            if pm not in per_pm:
                continue
            if "NK" not in step:
                continue
            if not JOBID_RE.search(jobid):  # skip DRY_RUN_* etc.
                continue
            if tws and tw not in tws:
                continue
            short = JOBID_RE.search(jobid).group(1)
            per_pm[pm][short] = (jobid, tw, row.get("script", ""))
    return {pm: list(d.values()) for pm, d in per_pm.items()}


def find_log_for_jobid(pm, jobid):
    short = JOBID_RE.search(jobid).group(1)
    pat = os.path.join(REPO, "logs", "julia", pm, "**", "periodic", "NK",
                       f"*{short}*.log")
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


# Diffusivity / timestep / adjoint markers embedded in the log filename, e.g.
# ..._kH75_kVML5e-2_kVBG15e-6_mkappaV_LBS_DTx2_traf_Pardiso_prec_169826921...
RE_KH = re.compile(r"_kH([0-9eE.+-]+)")
RE_KVML = re.compile(r"_kVML([0-9eE.+-]+)")
RE_KVBG = re.compile(r"_kVBG([0-9eE.+-]+)")
RE_DT = re.compile(r"_DTx(\d+)")


# Older OM2-025 (and OM2-1) runs did not encode diffusivities in the filename;
# those inherited the OM2-1 defaults below. We surface that explicitly rather
# than leaving "?" so the high- vs reduced-diffusivity comparison is readable.
DEFAULT_DIFF = {"kH": "300", "kVML": "1e-1", "kVBG": "3e-5"}


def config_tag(path):
    """Diffusivity/timestep signature from the log filename (for grouping).

    When kH/kVML/kVBG are absent from the filename the run used the inherited
    OM2-1 defaults (kH300 kVML1e-1 kVBG3e-5); `diff_default` flags that case."""
    b = os.path.basename(path)
    kh = RE_KH.search(b)
    kvml = RE_KVML.search(b)
    kvbg = RE_KVBG.search(b)
    dt = RE_DT.search(b)
    diff_in_name = bool(kh or kvml or kvbg)
    return {
        "kH": kh.group(1) if kh else DEFAULT_DIFF["kH"],
        "kVML": kvml.group(1) if kvml else DEFAULT_DIFF["kVML"],
        "kVBG": kvbg.group(1) if kvbg else DEFAULT_DIFF["kVBG"],
        "DTx": dt.group(1) if dt else "?",
        "traf": "traf" in b,
        "diff_default": not diff_in_name,
    }


def cfg_str(cfg):
    s = f"kH{cfg['kH']} kVML{cfg['kVML']} kVBG{cfg['kVBG']}"
    if cfg["diff_default"]:
        s += "(default,untagged)"
    s += f" DTx{cfg['DTx']}"
    return s + (" traf" if cfg["traf"] else "")


def min_drift(iters):
    vals = [it["vol_rms_drift_years"] for it in iters
            if it["vol_rms_drift_years"] is not None]
    return min(vals) if vals else None


def is_converged(iters, threshold):
    """A solve counts as converged if its smallest vol_rms_drift dips below the
    threshold (default 1e-6 yr) — i.e. it actually reached the NK tolerance."""
    md = min_drift(iters)
    return md is not None and md < threshold


def fmt_resid(x):
    return "—" if x is None else f"{x:.2e}"


def trend(jvps):
    """Short verdict on whether JVPs/Newton-iter trend downward."""
    nz = [j for j in jvps if j > 0]
    if len(nz) < 2:
        return "n/a"
    decreasing = all(b <= a for a, b in zip(nz, nz[1:]))
    if decreasing:
        return "decreasing"
    return "down→0" if nz[-1] < nz[0] else "non-monotonic"


def report_log(path, indent=""):
    iters, pre = parse_log(path)
    if not iters:
        print(f"{indent}(no G! residual blocks found — not an NK solve log?)")
        return None
    if pre:
        print(f"{indent}note: {pre} JVP(s) before first residual — ignored")
    jvps = [it["jvps"] for it in iters]
    print(f"{indent}Newton iters: {len(iters)}   "
          f"total JVPs: {sum(jvps)}   JVP/iter: {jvps}   ({trend(jvps)})")
    print(f"{indent}{'it':>3} {'JVPs':>5} {'vol_rms_drift':>14} "
          f"{'max_drift':>12} {'mean_drift':>12}")
    for it in iters:
        print(f"{indent}{it['newton']:>3} {it['jvps']:>5} "
              f"{fmt_resid(it['vol_rms_drift_years']):>14} "
              f"{fmt_resid(it['max_drift_years']):>12} "
              f"{fmt_resid(it['mean_drift_years']):>12}")
    return iters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pm", nargs="+",
                    default=["ACCESS-OM2-1", "ACCESS-OM2-025"],
                    help="parent models to analyse")
    ap.add_argument("--last", type=int, default=10,
                    help="number of most-recent matching NK solves per PM (0 = all)")
    ap.add_argument("--tw", nargs="*", default=None,
                    help="restrict to these TIME_WINDOWs (e.g. 1968-1977 1999-2008)")
    ap.add_argument("--converged-only", action="store_true",
                    help="keep only solves that reached the NK tolerance "
                         "(min vol_rms_drift < --converged-threshold)")
    ap.add_argument("--converged-threshold", type=float, default=1e-6,
                    help="vol_rms_drift_years below which a solve counts as "
                         "converged (default 1e-6)")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="parse these log files directly, ignoring submissions.tsv")
    ap.add_argument("--csv", default=None,
                    help="also write a tidy CSV of (pm,tw,jobid,cfg,newton,jvps,drifts)")
    args = ap.parse_args()

    csv_rows = []

    if args.logs:
        for path in args.logs:
            print(f"\n=== {os.path.relpath(path, REPO)} ===")
            iters = report_log(path)
            jm = JOBID_RE.search(os.path.basename(path))
            jid = jm.group(1) if jm else os.path.basename(path)
            for it in (iters or []):
                csv_rows.append(("", "", jid, "", it["newton"], it["jvps"],
                                 it["vol_rms_drift_years"], it["max_drift_years"],
                                 it["mean_drift_years"]))
        _write_csv(args.csv, csv_rows)
        return

    per_pm = load_jobs_from_tsv(args.pm, args.tw)
    for pm in args.pm:
        candidates = per_pm.get(pm, [])
        # Parse every candidate that has a log + at least one residual block,
        # then filter/select. We scan ALL jobs so "last N converged" works even
        # when most recent submissions failed or never converged.
        parsed = []
        n_nolog = n_noiter = 0
        for jobid, tw, _script in candidates:
            log = find_log_for_jobid(pm, jobid)
            if not log:
                n_nolog += 1
                continue
            iters, _pre = parse_log(log)
            if not iters:
                n_noiter += 1
                continue
            parsed.append({
                "short": JOBID_RE.search(jobid).group(1), "tw": tw, "log": log,
                "iters": iters, "cfg": config_tag(log),
                "converged": is_converged(iters, args.converged_threshold),
            })

        sel = [p for p in parsed if p["converged"]] if args.converged_only else parsed
        n_match = len(sel)
        if args.last:
            sel = sel[-args.last:]

        print("\n" + "=" * 78)
        label = "converged " if args.converged_only else ""
        print(f"PM = {pm}   ({len(sel)} {label}NK solve(s) shown"
              + (f"; {n_match} matched, " if args.last and n_match > len(sel) else "; ")
              + f"{len(parsed)} parsed, {n_nolog} no-log, {n_noiter} no-residual"
              + (f", TW in {args.tw}" if args.tw else "") + ")")
        if args.converged_only:
            print(f"  (converged := min vol_rms_drift < {args.converged_threshold:g})")
        print("=" * 78)

        for p in sel:
            print(f"\n[{p['tw']}] job {p['short']}   [{cfg_str(p['cfg'])}]"
                  + ("" if p["converged"] else "   NOT CONVERGED"))
            print(f"  {os.path.relpath(p['log'], REPO)}")
            report_log(p["log"], indent="  ")
            for it in p["iters"]:
                csv_rows.append((pm, p["tw"], p["short"], cfg_str(p["cfg"]),
                                 it["newton"], it["jvps"],
                                 it["vol_rms_drift_years"],
                                 it["max_drift_years"], it["mean_drift_years"]))

        # per-PM roll-up, grouped by diffusivity/timestep config so the effect of
        # changing diffusivities is easy to read off.
        if sel:
            print(f"\n  --- {pm} summary (grouped by config) ---")
            groups = OrderedDict()
            for p in sel:
                groups.setdefault(cfg_str(p["cfg"]), []).append(p)
            for cfg, ps in groups.items():
                print(f"\n  [{cfg}]  ({len(ps)} solve(s))")
                for p in ps:
                    jl = [it["jvps"] for it in p["iters"]]
                    print(f"    {p['tw']} {p['short']}: {len(jl)} iters, "
                          f"JVP/iter={jl}, ΣJVP={sum(jl)}, "
                          f"min_drift={fmt_resid(min_drift(p['iters']))} "
                          f"({trend(jl)})")

    _write_csv(args.csv, csv_rows)


def _write_csv(path, rows):
    if not path:
        return
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pm", "tw", "jobid", "config", "newton", "jvps",
                    "vol_rms_drift_years", "max_drift_years", "mean_drift_years"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
