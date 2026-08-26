# KNOWLEDGE.md — Accumulated Research Learnings

The researcher agent reads this file FIRST every session. One entry per sub-stage: the single most important insight for building the final model. Newest entries at the bottom.

## Learnings

- [T1.1] (2026-08-26) Canonical fold design (in harness.py, use everywhere): 5 expanding folds with 2-year validation windows 2008-09 … 2016-17 — anchored at the last year where ≥50% of CMAs are active (2017), NOT the global max train year (2022), which yields degenerate Sherbrooke-only folds; data gotchas: X/y train CSVs align positionally, MDA needs a last-train-row anchor to score the first val month's direction, and evaluation.py's NMSE/NRMSE/NMAE normalize by row count n (MSE/n² etc.), not by scale.
- [T1.2] (2026-08-26) Harness VALIDATED and the bar is set: persistence reference under the 5 canonical folds = MDA 0.4587±0.0342, NMSE 1.52e-06±1.27e-06, NRMSE 0.00113±0.00049, NMAE 0.00061±0.00024 (per-fold values in artifacts/T1.2/, exact smoke-test match confirmed, prediction.py shift(1) convention verified identical on every scored row); two caveats all later work must respect: (1) upstream imputation (data_imputation.py ~L458-467) fits annual→monthly distribution + bidirectional interpolation on train+test JOINTLY before the split → CV absolutes may be optimistic; train-only re-imputation sensitivity check scheduled for T3.3/before T8; (2) fold-5 eval sets are protocol-dependent (317 rows for persistence-as-model vs 245 under a model blind to the 6 late-starter CMAs) — model-vs-baseline comparisons are only valid within a fold's common mask, never across folds.
- [T2.1] (2026-08-26) PERSISTENCE REMAINS THE BAR after the full simple-baseline suite: on identical folds/masks, drift MDA 0.4203±0.0384 / ARIMA(1,1,0) 0.4343±0.0396 / pooled ridge-on-lags 0.3710±0.0425 all LOSE to persistence on every metric (canonical reference now MDA 0.4625±0.0394 — supersedes T1.2's 0.4587±0.0342 because _anchored_mda was fixed to score unanchored within-window directions for CMAs with no train history; corrected row T1.2-baseline-validation-mdafix in runs.jsonl). Clarification of the T1.2 entry: "exact smoke-test match" certifies harness DETERMINISM/reproducibility only, not predictive strength. Substantive insights: NHPI total is near-martingale at monthly frequency (estimated drift/AR extrapolation actively hurts at 12–24-month horizons); ridge-on-lags matches persistence's error class on point metrics (NRMSE 0.00166 vs 0.00113) but has WORSE-than-persistence direction (MDA 0.371) → directional skill must come from target reformulation or different signal, motivating T2.2.

## Standing decisions

- Target formulation (level vs Δlog vs YoY): to be decided empirically in T2
- Validation scheme: expanding-window CV over years, grouped by CMA (defined once in harness.py, T1)
- Metrics: MDA, NMSE, NRMSE, NMAE (reuse evaluation/evaluation.py); persistence baseline always reported alongside

## Open questions

- Does spatial pooling help at this scale? (probe scheduled in T2.3)
- Do spatial-lag features from the CMA shapefile add value over temporal features alone? (T3/T6)
