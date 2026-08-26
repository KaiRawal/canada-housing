# KNOWLEDGE.md — Accumulated Research Learnings

The researcher agent reads this file FIRST every session. One entry per sub-stage: the single most important insight for building the final model. Newest entries at the bottom.

## Learnings

- [T1.1] (2026-08-26) Canonical fold design (in harness.py, use everywhere): 5 expanding folds with 2-year validation windows 2008-09, 2010-11, 2012-13, 2014-15, 2016-17 — anchored at the last year where ≥50% of CMAs are active (2017), NOT the global max train year (2022), which yields degenerate Sherbrooke-only folds; data gotchas: X/y train CSVs align positionally, MDA needs a last-train-row anchor to score the first val month's direction, and evaluation.py's NMSE/NRMSE/NMAE normalize by row count n (MSE/n² etc.), not by scale.

## Standing decisions

- Target formulation (level vs Δlog vs YoY): to be decided empirically in T2
- Validation scheme: expanding-window CV over years, grouped by CMA (defined once in harness.py, T1)
- Metrics: MDA, NMSE, NRMSE, NMAE (reuse evaluation/evaluation.py); persistence baseline always reported alongside

## Open questions

- Does spatial pooling help at this scale? (probe scheduled in T2.3)
- Do spatial-lag features from the CMA shapefile add value over temporal features alone? (T3/T6)
