"""Tests for the pretrained_ts baseline (variant V6, Chronos-Bolt-Small)."""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_pretrained_ts  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "pretrained_ts.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"


def test_artifact_exists_and_reloads():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "pretrained_ts"
    assert obj.get("target") == TARGET
    # Primary model ran (Chronos), weights NOT pickled into the artifact.
    assert obj.get("model_name") == "amazon/chronos-bolt-small"
    assert obj.get("model_run") == "primary-chronos"
    # Searched grid + winner recorded.
    assert "search_grid" in obj and len(obj["search_grid"]) == 2
    assert {g["context"] for g in obj["search_grid"]} == {"full", "last120"}
    assert "best_params" in obj
    assert obj["best_params"]["context"] in ("full", "last120")
    assert obj["best_params"] in [
        {"context": g["context"], "device": v["device"]}
        for g, v in zip(obj["search_grid"], obj["val_nrmse"])
    ]
    assert math.isfinite(obj["best_val_nrmse"])
    # Per-CMA forecasts stored (24 CMAs), weights file must stay small.
    assert "test_forecasts" in obj and len(obj["test_forecasts"]) == 24
    assert "train_forecasts" in obj and len(obj["train_forecasts"]) == 24
    assert ARTIFACT.stat().st_size < 10 * 1024 * 1024  # no 191MB weights


def test_predict_row_count_matches_test_rows():
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    _, test_eval = predict_pretrained_ts(TARGET, ROOT / "artifacts")
    # Predictions align to test rows; TRY-1 blended output carries NaN on
    # the first row per CMA (persistence warmup), like the other blended
    # variants (run-1 raw path had full coverage; w=0 winner follows the
    # persistence mask bit-for-bit).
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


def test_metrics_json_has_pretrained_ts_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "pretrained_ts" in data
    entry = data["pretrained_ts"]
    for key in ("nrmse_train", "nrmse_test", "mda_test", "nmae_test"):
        assert key in entry, f"missing key: {key}"
        assert math.isfinite(entry[key]), f"non-finite {key}: {entry[key]}"
