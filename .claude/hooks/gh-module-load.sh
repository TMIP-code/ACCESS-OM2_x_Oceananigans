#!/bin/bash
# PreToolUse hook: block bare `gh` commands that haven't loaded the module
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [[ -z "$COMMAND" ]]; then
  exit 0
fi

# If the command invokes gh as a command (first token, or after && / | / ;)
# but doesn't load the module, block it
if echo "$COMMAND" | grep -qE '(^|&&|\|{1,2}|;)\s*gh\s'; then
  if ! echo "$COMMAND" | grep -q 'module load system-tools/gh'; then
    echo "gh requires module loading. Prepend: module use /g/data/vk83/modules && module load system-tools/gh &&" >&2
    exit 2
  fi
fi

exit 0
