---
description: Run the autonomous data flywheel on a problem — relentless until gates pass.
agent: flywheel-orchestrator
---

Run the autonomous flywheel on the problem at `$ARGUMENTS`.

`$ARGUMENTS` is a path to `.flywheel` or its `problem.yaml`. If empty, use `.flywheel/problem.yaml`.

The orchestrator owns `.flywheel/runs/<ts>/flywheel-state.json` and `decisions.md`, confirms resources once at run start (measured at setup; abort/downscale + log on drift), dispatches sandbox-executor/reviewer loops (gate-driven, parallel light variants only, heavy serially), then researcher/planner/flywheel-executor as needed. Heavy jobs run via `nohup` (one job → one log) + `while pgrep ... sleep 10` so verification can proceed in parallel. All installs in `constraints.venv`, global caches, `timeout` where configured.
