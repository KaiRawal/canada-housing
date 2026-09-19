"""Tests for the linear_regression baseline (variant V2)."""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_linear_regression  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "linear_regression.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"


def test_artifact_exists_and_reloads():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "linear_regression"
    assert obj.get("target") == TARGET
    # TRY-5 (H1 hierarchical partial-pooling): shared EN(0.03/1.0) on the
    # 110-col dedup-0.99 set + shrunk per-CMA intercept offsets (k=10);
    # 2 samples (partial-pooling + Sherbrooke ablation), val-gated, honest test.
    assert "search_grid" in obj and len(obj["search_grid"]) == 2
    assert "best_params" in obj and "model" in obj
    # TRY-5 features: 176 exogenous + 11 causal lag/rolling = 187 full,
    # greedy keep-first full corr-clustering (|corr| > 0.99 vs ANY kept col,
    # exo+lag order): 110 kept / 77 dropped. Keep-first keeps the exogenous
    # total_lag_1 and drops the identical lag_1 (corr=1.0).
    assert "exo_feature_cols" in obj and len(obj["exo_feature_cols"]) == 176
    assert "lag_feature_cols" in obj and len(obj["lag_feature_cols"]) == 11
    assert obj.get("try") == 5
    assert obj.get("dedup_thresh") == 0.99
    assert "dropped_dup_cols" in obj and len(obj["dropped_dup_cols"]) == 77
    assert "lag_1" in obj["dropped_dup_cols"]
    assert "total_lag_1" in obj["feature_cols"]
    assert "lag_1" not in obj["feature_cols"]
    assert len(obj["feature_cols"]) == 110
    # Hierarchical offsets: one shrunk intercept per CMA, k recorded.
    assert "cma_offsets" in obj and len(obj["cma_offsets"]) == 24
    assert all(math.isfinite(v) for v in obj["cma_offsets"].values())
    assert obj.get("offset_k") == 10
    # Winner is one of the grid entries; blend weight recorded + on grid.
    assert obj["best_params"] in obj["search_grid"]
    assert "blend_w" in obj and math.isfinite(obj["blend_w"])
    assert obj["blend_w"] in obj.get("blend_grid", [obj["blend_w"]])


def test_predict_row_count_matches_test_rows():
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    _, test_eval = predict_linear_regression(TARGET, ROOT / "artifacts")
    # Predictions align to test rows; blended output carries NaN on the
    # first row per CMA (persistence warmup), like the persistence test.
    assert len(test_eval) == len(y_test)
    n_cmas = int(y_test["cma_canonical"].nunique())
    valid = int(
        (test_eval[f"{TARGET}_true"].notna() & test_eval[f"{TARGET}_pred"].notna()).sum()
    )
    assert valid > 0
    assert valid == len(y_test) - n_cmas
    assert test_eval.loc[
        test_eval[f"{TARGET}_pred"].notna(), f"{TARGET}_pred"
    ].map(math.isfinite).all()


def test_metrics_json_has_linear_regression_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "linear_regression" in data
    entry = data["linear_regression"]
    for key in ("nrmse_train", "nrmse_test", "mda_test", "nmae_test"):
        assert key in entry, f"missing key: {key}"
        assert math.isfinite(entry[key]), f"non-finite {key}: {entry[key]}"
