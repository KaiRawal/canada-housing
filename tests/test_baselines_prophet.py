"""Tests for the prophet baseline (variant V4), best-of-3 restore (TRY-1 winner)."""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_prophet  # noqa: E402
from evaluation.evaluation import calculate_all_metrics  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "prophet.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"

EXPECTED_TUNED = {
    "Edmonton, Alberta",
    "Guelph, Ontario",
    "London, Ontario",
    "Montréal, Quebec",
}


def test_artifact_exists_and_reloads():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "prophet"
    assert obj.get("target") == TARGET
    # Best-of-3 restore: class artifact = TRY-1 winner (tie, w=0).
    # Accept only the restored best (try==1), not try==2/3.
    assert obj.get("try") == 1
    assert obj.get("kind") == "prophet-per-cma-lag-regressor"
    # TRY-1 provenance: {lag_1} vs {lag_1,lag_12} x cps{0.001,0.01} = 4;
    # sps=0.1 fixed, yearly True globally; 16 tune + 24 refit = 40 fits.
    assert "search_grid" in obj and len(obj["search_grid"]) == 4
    assert "best_params" in obj and "models" in obj
    assert obj["best_params"] in obj["search_grid"]
    for entry in obj["search_grid"]:
        assert list(entry["regressors"]) in (["lag_1"], ["lag_1", "lag_12"])
        assert float(entry["changepoint_prior_scale"]) in (0.001, 0.01)
    # Winner is lag_1-only / cps 0.01 (pooled model-only val ~1.72e-03).
    assert list(obj["best_params"]["regressors"]) == ["lag_1"]
    assert float(obj["best_params"]["changepoint_prior_scale"]) == 0.01
    # Same 4 tuning CMAs as run-1/TRY-2.
    assert "tuned_cmas" in obj and set(obj["tuned_cmas"]) == EXPECTED_TUNED
    # Global affine cal + all-or-nothing blend, cal-before-blend, w=0 tie.
    for key in ("calib_a", "calib_b", "blend_w", "regressor_cols", "cal_order"):
        assert key in obj, f"missing key: {key}"
    assert math.isfinite(float(obj["calib_a"]))
    assert math.isfinite(float(obj["calib_b"]))
    assert float(obj["blend_w"]) == 0.0
    assert list(obj["blend_grid"]) == [0.0, 1.0]
    assert str(obj["cal_order"]) == "cal-before-blend"
    assert list(obj["regressor_cols"]) == ["lag_1"]
    assert float(obj.get("seasonality_prior_scale", 0.1)) == 0.1
    # TRY-1 performs 40 Prophet fits (16 tune + 24 refit).
    assert int(obj.get("n_fits", -1)) == 40
    # Yearly ON globally (no per-CMA switch in TRY-1).
    assert obj.get("yearly_seasonality") is True
    assert str(obj.get("yearly_rule")) == "yearly=True"
    assert "yearly_by_cma" in obj and len(obj["yearly_by_cma"]) == 24
    assert len(obj.get("yearly_false_cmas", ["x"])) == 0
    assert len(obj.get("yearly_true_cmas", [])) == 24
    assert all(v is True for v in obj["yearly_by_cma"].values())
    # One fitted model per CMA.
    assert len(obj["models"]) == 24
    assert int(obj.get("n_models", 0)) == 24
    # Nesting proof: fitted lag_1 effective slopes ~= 1 (persistence).
    slopes = obj.get("lag1_slopes", {})
    assert len(slopes) == 24
    finite = [v for v in slopes.values() if math.isfinite(v)]
    assert len(finite) == 24
    median = float(obj["median_lag1_slope"])
    assert math.isfinite(median)
    assert 0.5 < median < 1.5, f"median lag_1 slope {median} not ~= 1"


def test_predict_row_count_matches_test_rows():
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    _, test_eval = predict_prophet(TARGET, ROOT / "artifacts")
    assert len(test_eval) == len(y_test)
    valid = int(
        (test_eval[f"{TARGET}_true"].notna() & test_eval[f"{TARGET}_pred"].notna()).sum()
    )
    # Blend propagates the persistence NaN mask: exactly the first row
    # per CMA has no persistence history inside the test frame.
    n_cmas = int(y_test["cma_canonical"].nunique())
    assert valid == len(y_test) - n_cmas, f"valid={valid} len={len(y_test)}"
    assert test_eval.loc[
        test_eval[f"{TARGET}_pred"].notna(), f"{TARGET}_pred"
    ].map(math.isfinite).all()


def test_predict_reproduces_metrics_without_refit():
    # predict.py must reproduce the train-time metrics from the stored
    # models + calibration (no refitting).
    data = json.loads(METRICS.read_text())
    train_eval, test_eval = predict_prophet(TARGET, ROOT / "artifacts")
    m_train = calculate_all_metrics(train_eval, TARGET)
    m_test = calculate_all_metrics(test_eval, TARGET)
    entry = data["prophet"]
    assert math.isclose(
        float(m_test["NRMSE"]), float(entry["nrmse_test"]), rel_tol=1e-6
    )
    assert math.isclose(
        float(m_train["NRMSE"]), float(entry["nrmse_train"]), rel_tol=1e-6
    )


def test_metrics_json_has_prophet_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "prophet" in data
    entry = data["prophet"]
    for key in ("nrmse_train", "nrmse_test", "mda_test", "nmae_test"):
        assert key in entry, f"missing key: {key}"
        assert math.isfinite(entry[key]), f"non-finite {key}: {entry[key]}"
    # Best-of-3 restore: TRY-1 ties persistence bit-for-bit (w=0).
    assert entry["nrmse_test"] == 0.0003778184008276179
    assert entry["nrmse_test"] == data["persistence"]["nrmse_test"]
