"""Tests for the persistence baseline (variant V1)."""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_persistence  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "persistence.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"


def test_artifact_exists_and_reloads():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "persistence"


def test_predict_row_count_matches_test_rows():
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    train_eval, test_eval = predict_persistence(TARGET, ROOT / "artifacts")
    # predict output row count equals test rows
    assert len(test_eval) == len(y_test)
    # valid preds = test rows minus one NaN first-row per CMA
    n_cmas = int(y_test["cma_canonical"].nunique())
    valid = int(
        (test_eval[f"{TARGET}_true"].notna() & test_eval[f"{TARGET}_pred"].notna()).sum()
    )
    assert valid > 0
    assert valid == len(y_test) - n_cmas


def test_metrics_json_has_persistence_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "persistence" in data
    nrmse_test = data["persistence"]["nrmse_test"]
    assert math.isfinite(nrmse_test)
