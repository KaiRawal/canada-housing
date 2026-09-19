# AGENTS.md

<!-- flywheel:start -->
## Flywheel (scaffolded by auto-flywheel)
- Flywheel home is `.flywheel/`: spec `.flywheel/problem.yaml`, scratch `.flywheel/runs/`, logger `.flywheel/shared/observe.py`.
- Work on scratch branch `flywheel/<problem>-<ts>`; deliver the winner there. Never push unless asked.
- Edit only within `target.scope` from `.flywheel/problem.yaml`. Never hand-edit `.flywheel/` mid-run.
- `.venv`-only installs; disk/mem guards + `timeout` per `constraints`.
- Commands: `/flywheel-new`, `/flywheel-run .flywheel`, `/flywheel-status`, `/flywheel-abort`.
<!-- flywheel:end -->
