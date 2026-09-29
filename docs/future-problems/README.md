# Future problem specs (reference only — NOT for execution)

These drafts capture scoped future flywheel runs so they can be read,
compared, and promoted to a live `.flywheel/problem.yaml` later. The
harness only ever reads `.flywheel/problem.yaml` (or an explicit
`$ARGUMENTS` path) — nothing here executes.

To promote one: `git checkout -b flywheel/<slug>-<ts>`, copy the draft
over `.flywheel/problem.yaml`, re-measure `constraints` (`df -h /`,
`vm_stat`/`free`, `nproc`), confirm open questions inside, then
`/flywheel-new` or `/flywheel-run .flywheel`.

- `housing-baselines-rerun.problem.yaml` — vanilla levels rerun with
  rerun-2 follow-ups (LSTM small-CMA exclusion, linear round-2 audit,
  conditional Chronos). Baselines pinned to rerun-2 numbers.
- `housing-direction.problem.yaml` — binary up/down direction head.
  Tracked, ungated; expect ~coin flip per learnings. Open questions
  flagged inside (base rate, naive tie-break).
- `housing-linear-sarimax.problem.yaml` — per-CMA SARIMAX-style linear
  rounds (AR → MA/diff/seasonal → liberal cross-CMA exo → pooled
  reconciliation), iterative diagnosis not grids. Gates `pass_rate == 1.0`
  plus `delta_rmse < 0.8499` (pinned pooled-AR(3) from delta_search.md).
- `housing-tabpfn-refined.problem.yaml` — doc-driven TabPFN v3 refinement
  (raw+categorical vs centered control, exo sets, wrapper search). Gates
  `pass_rate == 1.0` plus `delta_rmse < 0.8059` (pinned winner).
- `housing-timesfm-multivariate.problem.yaml` — TimesFM 3.0 native
  multivariate (16 joint variates + covariates + quantiles) with 2.5+XReg
  fallback and optional LoRA. Gates `pass_rate == 1.0` plus
  `delta_rmse < 0.8757` (pinned univariate). 3.0 weights non-commercial.
