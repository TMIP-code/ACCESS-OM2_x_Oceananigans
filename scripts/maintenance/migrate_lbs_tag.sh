#!/bin/bash
# Migrate MODEL_CONFIG directories to carry the _LBS tag fragment.
#
# Context: with LOAD_BALANCE=surface as the cross-model default (commit
# 6e85648), env_defaults.sh now produces MODEL_CONFIG=..._LBS_DTx<M> for
# OM2-025 (1x2) and OM2-01 (1x4). Existing on-disk artefacts predate this
# flip and live under the no-_LBS name. This script renames each old leaf
# to its _LBS-tagged sibling and leaves a relative symlink behind so
# pre-refactor jobs still in flight (which compute the no-_LBS path
# Julia-side) keep working through the transition.
#
# Run --dry-run first (the default) and review the proposed renames.
# After applying, the symlinks should be removed via --cleanup-symlinks
# once all pre-refactor PBS jobs have drained.
#
# Usage:
#   scripts/maintenance/migrate_lbs_tag.sh                    # dry-run, default policy
#   scripts/maintenance/migrate_lbs_tag.sh --apply            # apply renames + create symlinks
#   scripts/maintenance/migrate_lbs_tag.sh --all              # widen to every no-_LBS leaf
#   scripts/maintenance/migrate_lbs_tag.sh --cleanup-symlinks # remove back-compat symlinks

set -euo pipefail

mode="dry-run"
policy="default"

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)             mode="apply" ;;
        --dry-run)           mode="dry-run" ;;
        --cleanup-symlinks)  mode="cleanup" ;;
        --all)               policy="all" ;;
        --default)           policy="default" ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed '$d'
            exit 0
            ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

parent_models=(ACCESS-OM2-025 ACCESS-OM2-01)
sub_kinds=(TM periodic standardrun)

# Compute the _LBS-tagged sibling name. Insertion order (matches
# env_defaults.sh L118-L161): right before _DTx, else right before _traf,
# else appended at end.
insert_lbs() {
    local name="$1"
    if [[ "$name" == *_DTx* ]]; then
        echo "${name/_DTx/_LBS_DTx}"
    elif [[ "$name" == *_traf ]]; then
        echo "${name/%_traf/_LBS_traf}"
    else
        echo "${name}_LBS"
    fi
}

# Default policy: only target leaves that match the current default
# config tuple (MONTHLY_KAPPAV=yes, IMPLICIT_KAPPAV=yes, TBLOCKING=no) —
# these are the names that fresh submissions under today's defaults will
# hit. --all skips this filter.
matches_policy() {
    local name="$1"
    [ "$policy" = "all" ] && return 0
    [[ "$name" == *_mkappaV* ]] || return 1
    [[ "$name" == *_noKV* ]] && return 1
    [[ "$name" =~ _TB[0-9]+ ]] && return 1
    return 0
}

if [ "$mode" = "cleanup" ]; then
    echo "=== Cleanup mode: removing _LBS back-compat symlinks ==="
    found=0
    for pm in "${parent_models[@]}"; do
        for sub in "${sub_kinds[@]}"; do
            while IFS= read -r -d '' link; do
                target="$(readlink "$link")"
                base="$(basename "$link")"
                expected="$(insert_lbs "$base")"
                if [ "$target" = "$expected" ]; then
                    found=$((found + 1))
                    echo "rm $link  ->  $target"
                    rm "$link"
                fi
            done < <(find "outputs/$pm" -mindepth 4 -maxdepth 4 -type l -path "*/$sub/*" -print0 2>/dev/null || true)
        done
    done
    echo "Removed $found symlinks."
    exit 0
fi

echo "=== Migration plan (mode=$mode, policy=$policy) ==="
echo

renames=()
skipped_already_lbs=0
skipped_policy=0
collisions=0

for pm in "${parent_models[@]}"; do
    for sub in "${sub_kinds[@]}"; do
        # outputs/{PM}/{EXP}/{TW}/{sub}/<leaf>  -> mindepth/maxdepth 4 from outputs/{PM}
        while IFS= read -r -d '' dir; do
            [ -L "$dir" ] && continue  # skip existing symlinks
            base="$(basename "$dir")"
            parent="$(dirname "$dir")"
            if [[ "$base" == *_LBS* ]]; then
                skipped_already_lbs=$((skipped_already_lbs + 1))
                continue
            fi
            if ! matches_policy "$base"; then
                skipped_policy=$((skipped_policy + 1))
                continue
            fi
            new_base="$(insert_lbs "$base")"
            new_dir="$parent/$new_base"
            if [ -e "$new_dir" ] || [ -L "$new_dir" ]; then
                echo "COLLISION  $dir"
                echo "       ->  $new_dir  (already exists; skipping)"
                collisions=$((collisions + 1))
                continue
            fi
            renames+=("$dir|$new_dir")
            rel="${dir#$repo_root/}"
            echo "RENAME  $rel"
            echo "    ->  $(basename "$new_dir")  (+ symlink: $(basename "$dir") -> $(basename "$new_dir"))"
        done < <(find "outputs/$pm" -mindepth 4 -maxdepth 4 -type d -path "*/$sub/*" -print0 2>/dev/null || true)
    done
done

echo
echo "=== Summary ==="
echo "Renames proposed:        ${#renames[@]}"
echo "Skipped (already _LBS):  $skipped_already_lbs"
echo "Skipped (policy filter): $skipped_policy"
echo "Collisions (skipped):    $collisions"

if [ "$mode" = "dry-run" ]; then
    echo
    echo "(dry-run — no changes made. Re-run with --apply to execute.)"
    exit 0
fi

if [ ${#renames[@]} -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

echo
echo "=== Applying ==="
for entry in "${renames[@]}"; do
    old="${entry%|*}"
    new="${entry#*|}"
    mv "$old" "$new"
    ln -s "$(basename "$new")" "$old"
    echo "OK  $(basename "$old") -> $(basename "$new")"
done
echo
echo "Applied ${#renames[@]} renames. Symlinks left behind for back-compat;"
echo "run --cleanup-symlinks after pre-refactor PBS jobs have drained."
