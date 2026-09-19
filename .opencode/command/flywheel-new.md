---
description: Scaffold a new problem via Socratic interview — launches problem-architect.
agent: problem-architect
---

Launch the problem-architect to interview the user and scaffold `.flywheel/problem.yaml`.

User input: $ARGUMENTS (optional problem name or "help")

If no name is given, ask for one. Fill in `.flywheel/problem.yaml` (scaffolded by `flywheel-init`), walk through type / data / metric / gates / deliverables, measure the host once (`df -h /`, `vm_stat`/`free`, `nproc`) and confirm `constraints` (measured free minus headroom — never invent), validate against `.flywheel/shared/gate-contract.md`, and write the file.

In autonomous context (no `question` tool), use defaults and log to `decisions.md`.
