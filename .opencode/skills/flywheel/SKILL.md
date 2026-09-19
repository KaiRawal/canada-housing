---
name: flywheel
description: Use ONLY when running the data-flywheel loop — the executor/reviewer/researcher/planner/relentless-executor cycle driven by problem.yaml gates. Triggered by /flywheel-new or /flywheel-run.
---

# Flywheel skill

A problem-agnostic harness for iteratively building ML systems, predictors, or software features. The flywheel builds models, scores them against declarative gates, brings in outside perspective when stuck, plans, then relentlessly executes.

## Modes

| Mode | Agent (all `primary`) | Question tool |
|------|------------------------|---------------|
| `autonomous` | `flywheel-orchestrator` | never — log to `decisions.md` |
| `interactive` | `flywheel-orchestrator-interactive` | on ambiguous gate fail, researcher tie, commit-split |
| `new` | `problem-architect` | always — Socratic YAML builder |
| `ask` | `ask` | always welcome, never required — read-only observer (`/ask`) |

Lifecycle: `new` (setup interview, once) → `autonomous`/`interactive` (long run) → repeat. `ask` sits outside the loop: it answers questions from the repo map, git history, and run provenance without mutating anything.

## Loop

1. **Sandbox loop** (`sandbox-executor` + `sandbox-reviewer`, gate-driven, up to 3 nudges or until pass). Executor copies data (global cache), trains one variant, writes `metrics.json`. Parallel light variants only; heavy variants run serially, one job → one log. Reviewer scores vs `gates:` in `problem.yaml`.
2. **Research** (`researcher`) — only on plateau. Mines `references` / docs / papers for new metric/backbone ideas. Proposes candidates, never mutates `problem.yaml`.
3. **Plan** (`planner`) — merges logs + research into `.flywheel/runs/<ts>/plan.md` with chosen variant(s), pinned thresholds, commit split.
4. **Flywheel** (`flywheel-executor`) — consumes `plan.md`, fires `nohup` BG jobs (one job → one log, never two peak-RAM at once), polls via `while pgrep -f <job> >/dev/null; do sleep 10; done`, picks winners by gate margin, mints artifacts, generates tests/notebooks via builder scripts, runs `pytest/ruff/nbconvert` under `timeout`.

All installs in `{constraints.venv}`. Global caches reused. Resources measured at setup (`/flywheel-new`), confirmed once at run start; limits live in `problem.yaml: constraints`.

## Observability

Provenance is mandatory, phase + gate-delta granularity only. The orchestrator
is the sole writer via `.flywheel/shared/observe.py` (see `.flywheel/shared/event-schema.md`):
`init` once per run, then `append` one event per phase transition / reviewer
score / nudge / abort. `observe.py` dual-writes `.flywheel/runs/<ts>/events.jsonl`
(machine-readable) + `decisions.md` (human render) + `flywheel-state.json`.
Subagents return `gates` maps; the orchestrator logs. Query with
`.flywheel/shared/observe.py query <run-dir> [--phase ..] [--failed-only] [--gate ..]`
or plain `jq` over `events.jsonl`.

## Problem file

`.flywheel/problem.yaml` — the only problem-specific file. See `.flywheel/shared/gate-contract.md` for gate syntax.
