---
description: Run the interactive flywheel variant — asks via question tool on ambiguous gate fail, researcher tie, commit-split, resource conflict.
agent: flywheel-orchestrator-interactive
---

Run the interactive flywheel variant on `$ARGUMENTS` (same as `/flywheel-run` but with `question` enabled).

Use when you want human nudges at: gate-fail ambiguity, researcher tie, planner commit-split, unclear `problem.yaml` field, or resource conflict. Audit still lands in `.flywheel/runs/<ts>/decisions.md` interleaved with `Q:` blocks.
