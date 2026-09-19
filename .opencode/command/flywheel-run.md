---
description: Run the autonomous data flywheel on a problem — relentless until gates pass.
agent: flywheel-orchestrator
---

Run the autonomous flywheel on the problem at `$ARGUMENTS`.

`$ARGUMENTS` is a path to `.flywheel` or its `problem.yaml`. If empty, use `.flywheel/problem.yaml`.

The orchestrator owns `.flywheel/runs/<ts>/flywheel-state.json`, `.flywheel/runs/<ts>/learnings.md` (append-only learnings loop: per-iteration reviewer learnings, consolidated every 3 iterations or on plateau, injected into each next executor and into `plan.md`), and `decisions.md`, confirms resources once at run start (measured at setup; abort/downscale + log on drift), dispatches sandbox-executor/reviewer loops (gate-driven, parallel light variants only, heavy serially), then researcher/planner/flywheel-executor as needed. `problem.yaml` goals/gates are immutable — learnings nudge future steps only. Heavy jobs run via `nohup` (one job → one log) + `while pgrep ... sleep 10` so verification can proceed in parallel. All installs in `constraints.venv`, global caches, `timeout` where configured.
