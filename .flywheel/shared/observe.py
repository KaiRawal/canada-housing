"""Queryable provenance logger for the flywheel (stdlib only).

Dual-writes every event to:
  runs/<problem>/<ts>/events.jsonl        (machine-readable, JSONL v1)
  runs/<problem>/<ts>/decisions.md        (human render, gate-contract compat)
and refreshes runs/<problem>/<ts>/flywheel-state.json (orchestrator-owned).

Schema: see shared/event-schema.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = frozenset(
    {"sandbox-loop", "research", "plan", "flywheel", "done", "aborted", "new"}
)
EVENTS = frozenset(
    {"phase_transition", "gate_score", "decision", "nudge", "abort", "init"}
)

STATE_FILENAME = "flywheel-state.json"
EVENTS_FILENAME = "events.jsonl"
DECISIONS_FILENAME = "decisions.md"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(run_dir: Path) -> str:
    # run_dir is runs/<problem>/<ts> -> run_id "<problem>/<ts>"
    return f"{run_dir.parent.name}/{run_dir.name}"


def validate_event(d: dict) -> dict:
    missing = [k for k in ("phase", "agent", "event") if not d.get(k)]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if d["phase"] not in PHASES:
        raise ValueError(f"bad phase {d['phase']!r}, want one of {sorted(PHASES)}")
    if d["event"] not in EVENTS:
        raise ValueError(f"bad event {d['event']!r}, want one of {sorted(EVENTS)}")
    gates = d.get("gates") or {}
    if not isinstance(gates, dict):
        raise TypeError("gates must be a map")
    for name, g in gates.items():
        if not isinstance(g, dict) or "pass" not in g or "margin" not in g:
            raise ValueError(f"gate {name!r} needs {{pass, margin}}")
        if not isinstance(g["pass"], bool):
            raise TypeError(f"gate {name!r}.pass must be bool")
        try:
            g["margin"] = float(g["margin"])
        except (TypeError, ValueError):
            raise ValueError(f"gate {name!r}.margin must be numeric") from None
    if "pids" in d and d["pids"] is not None and not isinstance(d["pids"], list):
        raise ValueError("pids must be a list")
    it = d.get("iteration", 0)
    if not isinstance(it, int) or it < 0:
        raise ValueError("iteration must be int >= 0")
    return d


def _decisions_block(e: dict) -> str:
    return (
        f"## {e['ts']} — {e['phase']} — {e['agent']}\n"
        f"- Decision: {e.get('decision', '')}\n"
        f"- Rationale: {e.get('rationale', '')}\n"
        f"- Gate delta: {e.get('gate_delta', '')}\n"
    )


def append_event(run_dir: str | Path, **fields) -> dict:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    problem = fields.get("problem") or (run.parent.name if run.parent else "")
    event = {
        "ts": fields.get("ts") or utcnow(),
        "run_id": _run_id(run),
        "problem": problem,
        "iteration": int(fields.get("iteration", 0)),
        "phase": fields["phase"],
        "agent": fields["agent"],
        "event": fields["event"],
        "decision": fields.get("decision", ""),
        "rationale": fields.get("rationale", ""),
        "gate_delta": fields.get("gate_delta", ""),
        "gates": fields.get("gates") or {},
        "pids": fields.get("pids") or [],
    }
    validate_event(event)
    with open(run / EVENTS_FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    with open(run / DECISIONS_FILENAME, "a", encoding="utf-8") as f:
        f.write(_decisions_block(event) + "\n")
    # Refresh orchestrator-owned state (best-effort merge, never clobber pids
    # unless the caller passes them explicitly).
    state_path = run / STATE_FILENAME
    try:
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        state = {}
    state.update(
        {
            "phase": event["phase"],
            "iteration": event["iteration"],
            "last_scores": {
                k: v.get("margin") for k, v in event["gates"].items()
            }
            or state.get("last_scores", {}),
            "run_id": event["run_id"],
        }
    )
    if fields.get("pids") is not None:
        state["pids"] = event["pids"]
    state.setdefault("pids", [])
    state.setdefault("nudges", [])
    if event["event"] == "nudge" and event.get("decision"):
        state["nudges"] = [*state.get("nudges", []), event["decision"]]
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return event


def init_run(run_dir: str | Path, problem: str) -> dict:
    return append_event(
        run_dir,
        problem=problem,
        phase="sandbox-loop",
        agent="orchestrator",
        event="init",
        iteration=0,
        decision=f"run started for {problem}",
        rationale="flywheel init",
    )


def query_events(
    run_dir: str | Path,
    phase: str | None = None,
    event: str | None = None,
    gate: str | None = None,
    failed_only: bool = False,
) -> list[dict]:
    path = Path(run_dir) / EVENTS_FILENAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if phase and e.get("phase") != phase:
            continue
        if event and e.get("event") != event:
            continue
        gates = e.get("gates") or {}
        if gate and gate not in gates:
            continue
        if failed_only and not any(
            isinstance(g, dict) and g.get("pass") is False for g in gates.values()
        ):
            continue
        out.append(e)
    return out


def _parse_gates(s: str | None) -> dict:
    if not s:
        return {}
    try:
        d = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--gates must be JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise TypeError("--gates must be a JSON object")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="flywheel provenance logger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="start a run")
    p_init.add_argument("--run-dir", required=True)
    p_init.add_argument("--problem", required=True)

    p_ap = sub.add_parser("append", help="log one event")
    p_ap.add_argument("--run-dir", required=True)
    p_ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    p_ap.add_argument("--agent", required=True)
    p_ap.add_argument("--event", required=True, choices=sorted(EVENTS))
    p_ap.add_argument("--iteration", type=int, default=0)
    p_ap.add_argument("--problem", default="")
    p_ap.add_argument("--decision", default="")
    p_ap.add_argument("--rationale", default="")
    p_ap.add_argument("--gate-delta", default="")
    p_ap.add_argument("--gates", default="")
    p_ap.add_argument("--pids", default="")

    p_q = sub.add_parser("query", help="filter events")
    p_q.add_argument("run_dir")
    p_q.add_argument("--phase", default=None, choices=sorted(PHASES))
    p_q.add_argument("--event", default=None, choices=sorted(EVENTS))
    p_q.add_argument("--gate", default=None)
    p_q.add_argument("--failed-only", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "init":
        e = init_run(args.run_dir, args.problem)
        print(json.dumps(e))
        return 0
    if args.cmd == "append":
        pids = (
            [int(x) for x in args.pids.split(",") if x.strip()] if args.pids else []
        )
        e = append_event(
            args.run_dir,
            problem=args.problem,
            phase=args.phase,
            agent=args.agent,
            event=args.event,
            iteration=args.iteration,
            decision=args.decision,
            rationale=args.rationale,
            gate_delta=args.gate_delta,
            gates=_parse_gates(args.gates),
            pids=pids,
        )
        print(json.dumps(e))
        return 0
    rows = query_events(
        args.run_dir,
        phase=args.phase,
        event=args.event,
        gate=args.gate,
        failed_only=args.failed_only,
    )
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
