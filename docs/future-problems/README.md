# Future problem specs (reference only — NOT for execution)

These drafts capture scoped future flywheel runs so they can be read,
compared, and promoted to a live `.flywheel/problem.yaml` later. The
harness only ever reads `.flywheel/problem.yaml` (or an explicit
`$ARGUMENTS` path) — nothing here executes.

To promote one: `git checkout -b flywheel/<slug>-<ts>`, copy the draft
over `.flywheel/problem.yaml`, re-measure `constraints` (`df -h /`,
`vm_stat`/`free`, `nproc`), confirm open questions inside, then
`/flywheel-new` or `/flywheel-run .flywheel`.

- `housing-baselines-rerun.problem.yaml` — vanilla levels rerun with
  rerun-2 follow-ups (LSTM small-CMA exclusion, linear round-2 audit,
  conditional Chronos). Baselines pinned to rerun-2 numbers.
- `housing-direction.problem.yaml` — binary up/down direction head.
  Tracked, ungated; expect ~coin flip per learnings. Open questions
  flagged inside (base rate, naive tie-break).
