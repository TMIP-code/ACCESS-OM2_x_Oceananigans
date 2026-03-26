# Parse the mermaid DAG from scripts/pipeline.mmd into a bash associative array.
# Sourced by driver.sh. Expects `declare -A DAG` to exist before sourcing.
#
# Handles mermaid `&` grouping syntax: `A & B --> C & D` expands to
# edges A→C, A→D, B→C, B→D.
# Strips mermaid style annotations (e.g., `:::gpu`) from node names.

_dag_file="$(dirname "${BASH_SOURCE[0]}")/pipeline.mmd"
if [ ! -f "$_dag_file" ]; then
    echo "ERROR: DAG file not found: $_dag_file" >&2
    exit 1
fi

while read -r line; do
    # Trim leading whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    # Skip non-edge lines (graph directives, comments, blank lines)
    [[ "$line" == *"-->"* ]] || continue
    # Split on -->
    lhs="${line%% -->*}"
    rhs="${line##*--> }"
    # Split each side on &
    IFS='&' read -ra parents <<< "$lhs"
    IFS='&' read -ra children <<< "$rhs"
    for p in "${parents[@]}"; do
        p="${p// /}"; p="${p%%:::*}"
        [ -z "$p" ] && continue
        for c in "${children[@]}"; do
            c="${c// /}"; c="${c%%:::*}"
            [ -z "$c" ] && continue
            DAG[$p]="${DAG[$p]:+${DAG[$p]} }$c"
        done
    done
done < "$_dag_file"
unset _dag_file
