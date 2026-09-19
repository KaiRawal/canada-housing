---
description: Show flywheel state for a problem — reads .flywheel/runs/<ts>/flywheel-state.json and decisions.md.
agent: flywheel-orchestrator
---

Show status for `$ARGUMENTS` (a problem path or empty for the latest run).

Read `.flywheel/runs/<ts>/flywheel-state.json` (phase, iteration, pids, last_scores) and query `events.jsonl` via `.venv/bin/python .flywheel/shared/observe.py query <run-dir> [--phase ..] [--failed-only] [--gate ..]` (filters AND; raw `jq` also works). Tail `decisions.md` for the human render. Report which gates pass, which BG jobs are still running (`pgrep`), and disk/mem free. For a live view, `tail -f .flywheel/runs/<ts>/events.jsonl`.
