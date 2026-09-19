# Event schema v1 — queryable provenance for the flywheel

`.flywheel/runs/<ts>/events.jsonl` is the machine-readable audit trail.
`.flywheel/runs/<ts>/decisions.md` is the human render, dual-written from the
same `.flywheel/shared/observe.py:append_event` call. Both are orchestrator-owned.

## Event fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `ts` | ISO-8601 UTC `...Z` | yes | set by logger if omitted |
| `run_id` | str | yes | `<problem>/<ts>`, derived from run dir |
| `problem` | str | yes | problem slug |
| `iteration` | int >= 0 | yes | sandbox iteration, 0 for non-loop phases |
| `phase` | enum | yes | `sandbox-loop\|research\|plan\|flywheel\|done\|aborted\|new` |
| `agent` | str | yes | e.g. `orchestrator`, `sandbox-reviewer`, `researcher` |
| `event` | enum | yes | `phase_transition\|gate_score\|decision\|nudge\|abort\|init` |
| `decision` | str | no | what was done |
| `rationale` | str | no | why |
| `gate_delta` | str | no | human summary, e.g. `f1 0.82->0.87 (+0.05)` |
| `gates` | map | no | `{<gate>: {pass: bool, margin: float}}` |
| `pids` | int list | no | BG job pids alive at emit time |

`decisions.md` block per event (backward compat with `gate-contract.md`):

```
## <ts> — <phase> — <agent>
- Decision: ...
- Rationale: ...
- Gate delta: ...
```

## Usage

```bash
# start a run (creates events.jsonl + decisions.md + flywheel-state.json)
.venv/bin/python .flywheel/shared/observe.py init --run-dir .flywheel/runs/<ts> --problem toy-tabular

# log anything (phase + gate deltas only — no prompts/tool traces)
.venv/bin/python .flywheel/shared/observe.py append --run-dir .flywheel/runs/<ts> \
  --phase sandbox-loop --agent sandbox-reviewer --event gate_score \
  --iteration 1 --decision "scored linear" --rationale "first variant" \
  --gate-delta "f1 0.82 vs 0.85 (-0.03)" --gates '{"fidelity":{"pass":false,"margin":-0.03}}'

# interrogate for your custom harness
.venv/bin/python .flywheel/shared/observe.py query .flywheel/runs/<ts> --failed-only
.venv/bin/python .flywheel/shared/observe.py query .flywheel/runs/<ts> --phase research
.venv/bin/python .flywheel/shared/observe.py query .flywheel/runs/<ts> --event decision --gate fidelity
jq -c 'select(.gates.fidelity.pass==false)' .flywheel/runs/<ts>/events.jsonl
```

`query` filters are ANDed. `--failed-only` matches any gate with `pass==false`.
`--gate <name>` keeps events mentioning that gate. Raw `jq` works because the
file is plain JSONL.
