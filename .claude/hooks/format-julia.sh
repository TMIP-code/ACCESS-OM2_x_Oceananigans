#!/bin/bash
# PostToolUse hook: auto-format Julia files with runic after Edit/Write
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

if [[ "$FILE_PATH" == *.jl ]]; then
  runic --inplace "$FILE_PATH"
fi

exit 0
