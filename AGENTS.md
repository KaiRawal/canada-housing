# AGENTS.md — Canadian Housing ML Research Project

Read this before doing ANY work in this repository.

## Mission

Build and rigorously justify a predictive ML model for the **`total`** New Housing Price Index (NHPI) across 24 CMAs, beating the persistence baseline, with full documentation of *why* it is good: feature ablations, cross-validated hyperparameter tuning, validation accuracy, and model choice.

## Pipeline context

- `data_cleaning/` -> `imputation/data_imputation.py` -> `prediction/prediction.py` -> `evaluation/evaluation.py`
- Imputation already produces, in `prediction/`: `X_train_full_total.csv`, `y_train_full_total.csv`, `X_test_full_total.csv`, `y_test_full_total.csv`, `X_train_FS_total.csv`, `dropped_constant_features_total.txt`
- The only implemented predictor is the persistence baseline (`prediction.py`). Everything ML is built under `prediction/experiments/`.
- Evaluation metrics (MDA, NMSE, NRMSE, NMAE) live in `evaluation/evaluation.py` — reuse them.

## Research rules (mandatory)

1. **Target**: experiments use the `total` target ONLY. house/land are out of scope.
2. **Test set**: `X_test_full_total.csv` / `y_test_full_total.csv` are touched exactly once — final evaluation in T8. Never earlier. No exceptions.
3. **Validation**: expanding-window CV over years, grouped by CMA. Random KFold / shuffles across time are banned. Spatial generalization may additionally be probed via leave-one-CMA-out.
4. **Baseline**: persistence baseline metrics reported alongside every result, same folds, same metrics.
5. **Target formulation** (level vs Δlog vs YoY %) is decided empirically in T2, not a priori.
6. **Registry**: every experiment run appends to `prediction/experiments/runs.jsonl`. One learning per sub-stage goes into `prediction/experiments/KNOWLEDGE.md`. Read both before any new work.
7. **Branching**: all research work on the `research` branch. Never commit to main. Never push.
8. **Commits**: one commit per sub-stage by the experimenter agent; message format `exp(T<stage>.<sub>): <learning>`.
9. **Artifacts**: metrics JSONs, plots and small CSVs ARE committed (under `prediction/experiments/artifacts/`). No binaries >10MB.

## Agent setup

- `.opencode/agent/researcher.md` — orchestrator (primary). Run it to drive the whole roadmap.
- `.opencode/agent/experimenter.md` — subagent implementing one approved sub-stage.
- `.opencode/agent/critic.md` — subagent adversarially reviewing results pre-commit.
