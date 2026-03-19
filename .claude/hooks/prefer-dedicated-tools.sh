#!/bin/bash
# PreToolUse hook: block Bash commands that should use dedicated tools instead
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [[ -z "$COMMAND" ]]; then
  exit 0
fi

# Extract the first command token (skip env vars like FOO=bar)
CMD_FIRST=$(echo "$COMMAND" | sed 's/^[A-Za-z_][A-Za-z_0-9]*=[^ ]* *//' | awk '{print $1}')

case "$CMD_FIRST" in
  grep|rg|egrep|fgrep)
    echo "Use the Grep tool instead of $CMD_FIRST via Bash." >&2
    exit 2
    ;;
  find)
    echo "Use the Glob tool instead of find via Bash." >&2
    exit 2
    ;;
  cat|head|tail)
    echo "Use the Read tool instead of $CMD_FIRST via Bash." >&2
    exit 2
    ;;
  sed|awk)
    echo "Use the Edit tool instead of $CMD_FIRST via Bash." >&2
    exit 2
    ;;
esac

exit 0
