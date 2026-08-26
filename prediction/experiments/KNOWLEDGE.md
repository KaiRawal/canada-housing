# KNOWLEDGE.md — Accumulated Research Learnings

The researcher agent reads this file FIRST every session. One entry per sub-stage: the single most important insight for building the final model. Newest entries at the bottom.

## Learnings

(none yet — T1 pending)

## Standing decisions

- Target formulation (level vs Δlog vs YoY): to be decided empirically in T2
- Validation scheme: expanding-window CV over years, grouped by CMA (defined once in harness.py, T1)
- Metrics: MDA, NMSE, NRMSE, NMAE (reuse evaluation/evaluation.py); persistence baseline always reported alongside

## Open questions

- Does spatial pooling help at this scale? (probe scheduled in T2.3)
- Do spatial-lag features from the CMA shapefile add value over temporal features alone? (T3/T6)
