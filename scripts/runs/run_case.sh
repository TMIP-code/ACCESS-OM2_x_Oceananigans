#!/usr/bin/env bash
# Launch a "case" — sources a bash case file under scripts/runs/cases/ that
# exports env vars (PARENT_MODEL, EXPERIMENT, TIME_WINDOW, MLD_TIME_WINDOW,
# JOB_CHAIN, …) and then invokes scripts/driver.sh with that environment.
#
# Usage:
#   bash scripts/runs/run_case.sh <case_file>
#   DRY_RUN=yes bash scripts/runs/run_case.sh <case_file>
#
# Case files mirror the model_configs/{PARENT_MODEL}.sh pattern: plain bash
# files containing `export KEY=VALUE` lines.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <case_file>" >&2
    echo "Example: $0 scripts/runs/cases/OM2-1_TR1968-1977_MLD1972.sh" >&2
    exit 1
fi

CASE_FILE="$1"

if [ ! -f "$CASE_FILE" ]; then
    echo "ERROR: case file not found: $CASE_FILE" >&2
    exit 1
fi

case "$CASE_FILE" in
    *.sh) ;;
    *)
        echo "ERROR: case file must end in .sh (got: $CASE_FILE)" >&2
        exit 1
        ;;
esac

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

# Resolve to a repo-relative path so the manifest records a stable identifier
# even when the user passes an absolute path.
case_file_abs=$(readlink -f "$CASE_FILE")
case_file_rel="${case_file_abs#$repo_root/}"
export CASE_FILE="$case_file_rel"

echo "=== run_case.sh: sourcing $CASE_FILE ==="
# shellcheck disable=SC1090
source "$CASE_FILE"

echo "=== run_case.sh: invoking driver ==="
exec bash scripts/driver.sh
