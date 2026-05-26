#!/bin/bash
# Print cross-model and per-model default tables in Markdown.
#
# Tables are generated from the source files themselves
# (scripts/env_defaults.sh and model_configs/*.sh) so the README is never
# out of date with what production submissions actually use. Pipe into
# the README section between the markers
#   <!-- defaults-tables:begin -->  ...  <!-- defaults-tables:end -->
# either by hand (`scripts/print_defaults_tables.sh > /tmp/t.md`) or via
# the regen helper in the README ("Updating the defaults tables").

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# extract_defaults FILE
#   For lines of the form
#     VAR=${VAR:-VALUE}   # optional comment
#   print three TAB-separated columns: VAR, VALUE, COMMENT.
extract_defaults() {
    awk '
        /^[A-Za-z_][A-Za-z0-9_]*=\$\{[A-Za-z_][A-Za-z0-9_]*:-/ {
            line = $0
            split(line, a, "=\\$\\{[^:]*:-")
            var = a[1]
            rest = a[2]
            # strip the closing brace and anything after
            close_idx = index(rest, "}")
            value = substr(rest, 1, close_idx - 1)
            tail  = substr(rest, close_idx + 1)
            # extract trailing "# comment" if present
            sub(/^[[:space:]]*/, "", tail)
            comment = ""
            if (substr(tail, 1, 1) == "#") {
                comment = substr(tail, 2)
                sub(/^[[:space:]]*/, "", comment)
            }
            printf "%s\t%s\t%s\n", var, value, comment
        }
    ' "$1"
}

# Cross-model table from env_defaults.sh
echo "<!-- defaults-tables:begin -->"
echo ""
echo "#### Cross-model defaults (\`scripts/env_defaults.sh\`)"
echo ""
echo "| Variable | Default | Notes |"
echo "|---|---|---|"
extract_defaults scripts/env_defaults.sh | \
    awk -F'\t' '{
        # surround the value with backticks for monospace; empty stays empty
        v = $2; if (v == "") v = "(empty)"
        # collapse pipes in comments so the table cell doesn'\''t break
        gsub(/\|/, "\\|", $3)
        printf "| `%s` | `%s` | %s |\n", $1, v, $3
    }'
echo ""

# Per-model table
echo "#### Per-model defaults (\`model_configs/\`)"
echo ""
echo "| Variable | OM2-1 | OM2-025 | OM2-01 |"
echo "|---|---|---|---|"

# Collect everything into associative arrays, then join on the variable name.
declare -A OM1 OM025 OM01
ALL_VARS=""

collect() {
    local model="$1" file="$2"
    while IFS=$'\t' read -r var val _; do
        [ -n "$var" ] || continue
        # Tag a sentinel for variables that appear in some configs but not others
        case "$model" in
            OM2-1)   OM1[$var]="$val" ;;
            OM2-025) OM025[$var]="$val" ;;
            OM2-01)  OM01[$var]="$val" ;;
        esac
        # accumulate union of variable names, preserving first-seen order
        case " $ALL_VARS " in
            *" $var "*) ;;
            *) ALL_VARS="$ALL_VARS $var" ;;
        esac
    done < <(extract_defaults "$file")
}

collect OM2-1   model_configs/ACCESS-OM2-1.sh
collect OM2-025 model_configs/ACCESS-OM2-025.sh
collect OM2-01  model_configs/ACCESS-OM2-01.sh

# Also pull MODEL_SHORT (set without :- pattern) — these aren't in extract_defaults
for var in $ALL_VARS; do
    v1="${OM1[$var]:-—}"
    v2="${OM025[$var]:-—}"
    v3="${OM01[$var]:-—}"
    # Skip purely-walltime/memory variables (per-job resource knobs, voluminous)
    case "$var" in
        WALLTIME_*|PREP_*|VEL_*|CLO_*|PARTITION_MEM_PER_RANK|PARTITION_NCPUS|TMBUILD_*|PARTITION_QUEUE|PLOT_NK_*|PLOT_TM_*)
            continue ;;
    esac
    echo "| \`$var\` | \`$v1\` | \`$v2\` | \`$v3\` |"
done
echo ""
echo "_Resource knobs (walltimes, memory limits, queue choices for individual"
echo "PBS jobs) also live in \`model_configs/*.sh\` but are omitted from this"
echo "table — see the files for the full list._"
echo ""
echo "<!-- defaults-tables:end -->"
