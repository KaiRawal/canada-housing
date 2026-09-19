---
description: Ask questions about the repo, recent changes, or past flywheel runs — read-only, never edits.
agent: ask
---

Ask the ask agent about `$ARGUMENTS` (a question, or empty for orientation).

With no arguments, orient the user: what this repo does, the latest run
state across `.flywheel` (via `.flywheel/runs/<ts>/flywheel-state.json` and
`.flywheel/shared/observe.py query`), and the most recent commits (`git log --oneline
-10`). With a question, answer it from the repo, git history, and run
provenance (`events.jsonl` / `decisions.md`). Read-only: never edit files,
never run training, tests, or installs — offer verification commands instead.
