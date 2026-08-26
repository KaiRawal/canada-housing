# TASKS.md — Research Roadmap

Progress state for the Canadian Housing ML research program. Tick sub-stage checkboxes as they complete. The `researcher` agent resumes at the first unticked item.

Conventions: each sub-stage = one approved plan -> one experimenter run -> one critic review -> one commit (`exp(T<stage>.<sub>): <learning>`).

---

## T1. Harness & CV protocol
**Mini-goal**: shared experiment infrastructure so every later task is comparable.
- [x] T1.1 — Build `prediction/experiments/harness.py`: loads imputation outputs, defines expanding-window folds (grouped by CMA) ONCE, computes MDA/NMSE/NRMSE/NMAE via evaluation.py, writes runs.jsonl rows
- [x] T1.2 — Validate harness by reproducing persistence-baseline CV metrics; confirm no leakage paths

*Learning target: the exact fold design and metric conventions used everywhere.*

## T2. Baselines & target formulation
**Mini-goal**: establish what "beating persistence" means and predict the right quantity.
- [x] T2.1 — Baseline suite under the harness: persistence, drift (random walk with drift), per-CMA ARIMA/SARIMAX, ridge-on-lags
- [x] T2.2 — Target formulation study: level vs Δlog vs YoY % change, compared head-to-head on identical folds; decide empirically and record rationale
- [x] T2.3 — Spatial sanity check: leave-one-CMA-out probe to see whether spatial pooling helps at all (verdict: NO — persistence beats pooled-LOCO on 13/18 CMAs; spatial features demoted to ablation status; canonical MDA reference corrected to 0.6771±0.0474 under the symmetric zero-drop protocol)

*Learning target: which target form + which simple baseline is the bar to beat.*

## T3. Feature engineering
**Mini-goal**: give models spatial and temporal signal beyond lag-1.
- [x] T3.1 — Spatial features from CMA shapefile: kNN spatial weights, neighbor lag-1 targets/starts/rents, region encodings, centroid coordinates (demoted cheap ablation per T2.3 — kNN neighbour-lag features tested: marginal keep for T6 screening only, +0.0006/+0.0077 mean paired ΔMDA, still ~0.06 MDA below persistence; all five T2.3 critic fixes folded in, incl. intersection-mask paired scorer showing drift's MDA edge is real but jointly irrelevant)
- [x] T3.2 — Temporal features: multi-year lags/diffs/rolling stats of key CMHC groups (SCSS starts/completions, RMS rents, Census) — verdict: NO family improves paired ΔMDA (all mean ΔMDA negative, ≤2/5 folds), but the OWN-target block cuts ridge NRMSE 22% (0.00169→0.00131) and SCSS/RMS/Census blocks add nothing; T6 candidates = own-index long-horizon memory only (+ k=5 spatial marginal keep). All five T3.1 critic fixes folded in: anchor-bug fix leaves persistence-containing pairs identical, k3 now FAILS the keep rule (−0.0009), k5 still marginal keep (+0.0100); drift-edge wording downgraded to sign-consistent-but-not-established; explicit multi-metric bar added to KNOWLEDGE.md
- [x] T3.3 — Leakage audit of all new features (fit transforms per-fold only). **MUST include the train-only re-imputation sensitivity check** (T1.2/T2.1 critic gate): upstream `imputation/data_imputation.py` (~L458–467) fits annual→monthly distribution + bidirectional interpolation on train+test JOINTLY, so CV absolutes may be optimistic — re-run imputation restricted to the train split, re-score the then-current model on the canonical folds, and report any degradation BEFORE finalizing features (hard gate before T8). **DONE:** joint-imputation optimism quantified as NEGLIGIBLE (|ΔNRMSE| ≤ 0.8% relative, |ΔMDA| ≤ 0.001, persistence exact) → joint panels stay canonical with caveat; all builders certified causal after fixing bare global shifts + cross-sectional fill; OWN-block NRMSE claim corrected (0.954 full-panel / 0.781 without late starters); feature inventory in artifacts/T3.3/

*Learning target: which engineered feature families carry signal.*

## T4. Model benchmark
**Mini-goal**: rank model families fairly on identical folds.
- [ ] T4.1 — Linear family: Ridge/ElasticNet (with standardization inside fold pipeline)
- [ ] T4.2 — Tree ensembles: RandomForest, LightGBM/XGBoost/CatBoost
- [ ] T4.3 — Recursive forecasting GBM via skforecast; SARIMAX/VAR comparison where feasible
- [ ] T4.4 — Leaderboard with per-fold variance; shortlist top 2 families

*Learning target: winning model family (with uncertainty, not just mean CV score).*

## T5. Hyperparameter tuning
**Mini-goal**: squeeze validated performance out of the shortlist.
- [ ] T5.1 — Optuna search under the same CV folds; every trial logged to runs.jsonl
- [ ] T5.2 — Overfitting guardrails: tuning curves, chosen-vs-default deltas

*Learning target: best hyperparameters + evidence they generalize across folds.*

## T6. Feature ablations
**Mini-goal**: prove which features earn their place.
- [ ] T6.1 — Leave-one-family-out ablations (spatial, SCSS, RMS, Census, core-need, temporal) on tuned model
- [ ] T6.2 — Forward selection starting from lag-only, using the 15% FS downsample set first, confirming on full train folds

*Learning target: minimal sufficient feature set + quantified contribution of each family.*

## T7. Deep-learning probe (optional — may be skipped)
**Mini-goal**: test whether complexity beats gradient boosting here.
- [ ] T7.1 — ST-GNN (PyTorch Geometric) or HousingNet Transformer revival, same folds, clearly framed as exploratory

*Learning target: is deep learning worth it at ~1k rows? (A negative result is a valid learning.)*

## T8. Final model & single test evaluation
**Mini-goal**: the deliverable.
- [ ] T8.1 — Refit winner (all learnings applied: target form, feature set, family, params) on full training data
- [ ] T8.2 — ONE evaluation on held-out test split vs persistence baseline (MDA, NMSE, NRMSE, NMAE); report val->test gap
- [ ] T8.3 — Generate MODEL_CARD.md from runs.jsonl + KNOWLEDGE.md

*Learning target: final accuracy and the documented case for why the model is good.*
