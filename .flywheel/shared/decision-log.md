# Decision log pattern

Provenance is dual-written via `.flywheel/shared/observe.py` (schema: `.flywheel/shared/event-schema.md`):

- `.flywheel/runs/<ts>/events.jsonl` — queryable audit trail (phase + gate deltas only). Filter with `.flywheel/shared/observe.py query <run-dir> [--phase ..] [--failed-only] [--gate ..]` or plain `jq`.
- `.flywheel/runs/<ts>/decisions.md` — human render, one `## <timestamp> — <phase> — <agent>` block per event with Decision/Rationale/Gate delta.

- Autonomous mode: every iteration appends one event. No `question` calls.
- Interactive mode: same, plus interleaved `Q: ...` / `A: ...` blocks from the `question` tool.

The orchestrator is the only writer; subagents read and return `gates` maps for the orchestrator to log.

## Flywheel state

`.flywheel/runs/<ts>/flywheel-state.json`:

```json
{"phase":"sandbox-loop","iteration":2,"nudges":["try rbf"],"last_scores":{"f1":0.82},"pids":[12345]}
```

Polled via `while pgrep -f "train.py" >/dev/null; do sleep 10; done`.
