Read the AGENTS.md file for project context.

## Bash: disable colored output

This shell's startup config forces ANSI color codes into command output (e.g.
`alias ls='command ls --color'` always colorizes, even when piped), which
clutters tool results with escape sequences like `[0m`, `[32m`, `[36m`. Always
suppress color when running Bash commands:

- `ls`/`l`/`la` → add `--color=never` (e.g. `ls --color=never`, `ls -la --color=never`).
- `grep`/`egrep`/`fgrep` → add `--color=never` (their aliases default to `--color=auto`).
- General fallback: prefix the command with a backslash to bypass the alias
  entirely (e.g. `\ls`, `\grep`), or pass `--color=never` to any tool that
  supports it.