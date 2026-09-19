# Gate contract

Gates in `problem.yaml: gates` are expressions over metric names produced by the executor.

- Syntax: `"<metric> <op> <threshold>"` where `<op>` is `>= <= > < ==` and RHS is a float/int.
- Compound: `"f1 >= 0.85 and accuracy > random + 0.10"` — `random` is the baseline metric if present.
- Budgets: `gates.budgets: {samples: 200}` controls how many samples the reviewer scores.

Reviewer computes each metric, `eval`s the expression in a restricted namespace (no builtins), and returns `{pass: bool, margin: float}`. A gate with no metric is skipped.

## Search hints

`problem.yaml: hints` is optional free text from the user that biases search
order only — never gates, thresholds, or evaluation. The executor tries
hinted directions first; the planner weighs hints alongside evidence; the
reviewer stays hints-blind (scores gates only). Contradictory evidence
overrides hints, and the override must be logged in the decision rationale.
Empty hints means open search.

## Decision log contract

Every orchestrator event MUST go through `.flywheel/shared/observe.py append` (schema:
`.flywheel/shared/event-schema.md`), which dual-writes:

- `.flywheel/runs/<ts>/events.jsonl` — one JSON object per line (`phase`, `agent`, `event`, `gates: {<gate>: {pass, margin}}`)
- `.flywheel/runs/<ts>/decisions.md` — human render, one block per event:

```
## 2026-01-13T12:00:00Z — <phase> — <agent>
- Decision: ...
- Rationale: ...
- Gate delta: ...
```

Never hand-append `decisions.md`. Granularity is phase + gate deltas only.

Autonomous mode never calls `question`; interactive mode may call it at gate-fail / researcher tie / commit-split.

## Manifest

`.flywheel/runs/<ts>/manifest.json` or `artifacts/manifest.json` records `{sha256, shapes}` per artifact plus `{python, numpy, sklearn, ...}` versions — same pattern as `timpara/opencode-academic-research`'s Material Passport.
