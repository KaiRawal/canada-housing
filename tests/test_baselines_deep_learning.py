"""Tests for the lstm_rnn baseline (variant V5, artifact deep_learning.joblib).

Kind-aware: run-1 univariate artifacts (``lstm-global-univariate``) use
the lag-window path with full-row validity; try-1 multivariate-delta
artifacts (``lstm-global-multivariate-delta``) rebuild deduped feature
windows (GBT-style NaN semantics: first-row-per-CMA NaN via the
persistence blend, plus windowless/warmup NaN rows masked).
"""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_lstm_rnn  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "deep_learning.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"


def _load_artifact():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "lstm_rnn"
    assert obj.get("target") == TARGET
    return obj


def test_artifact_exists_and_reloads():
    obj = _load_artifact()
    # Searched grid + winner recorded (small grid, <= 6 configs).
    assert "search_grid" in obj and 1 <= len(obj["search_grid"]) <= 6
    assert "best_params" in obj and "state_dict" in obj
    assert obj["best_params"] in obj["search_grid"]
    assert set(("lookback", "hidden_size", "num_layers")) <= set(obj["best_params"])
    # Per-CMA target scalers stored (24 CMAs) with finite mean/std.
    assert "scalers" in obj and len(obj["scalers"]) == 24
    for cma, sc in obj["scalers"].items():
        assert math.isfinite(sc["mean"]) and math.isfinite(sc["std"])
        assert sc["std"] > 0
    # Device recorded.
    assert obj.get("device") in ("mps", "cpu")
    # Reloaded state dict builds a working model (input dim from artifact).
    import torch

    from baselines._lstm_rnn import build_model_from_state

    input_size = int(obj.get("input_size", 1))
    model = build_model_from_state(
        obj["state_dict"],
        hidden_size=int(obj["hidden_size"]),
        num_layers=int(obj["num_layers"]),
        device="cpu",
        input_size=input_size,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, int(obj["lookback"]), input_size))
    assert out.shape == (2, 1)
    assert all(math.isfinite(float(v)) for v in out.reshape(-1).tolist())
    if obj.get("kind") == "lstm-global-multivariate-delta":
        # Try-1: 110-col linear TRY-3 dedup set + delta nesting record.
        assert len(obj["feature_cols"]) == 110
        assert len(obj["exo_feature_cols"]) == 176
        assert len(obj["lag_feature_cols"]) == 11
        assert obj.get("formulation") == "delta"
        assert obj.get("lag_col") == "lag_1"
        assert math.isfinite(obj.get("calib_a", float("nan")))
        assert math.isfinite(obj.get("calib_b", float("nan")))
        assert "blend_w" in obj and math.isfinite(obj["blend_w"])
        assert obj["blend_w"] in obj.get("blend_grid", [obj["blend_w"]])


def test_predict_row_count_matches_test_rows():
    obj = _load_artifact()
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    _, test_eval = predict_lstm_rnn(TARGET, ROOT / "artifacts")
    # Predictions align to test rows (one-step-ahead per CMA).
    assert len(test_eval) == len(y_test)
    n_cmas = int(y_test["cma_canonical"].nunique())
    valid = int(
        (test_eval[f"{TARGET}_true"].notna() & test_eval[f"{TARGET}_pred"].notna()).sum()
    )
    assert valid > 0
    if obj.get("kind") == "lstm-global-multivariate-delta":
        # GBT-style NaN semantics: first-row-per-CMA NaN from the blend
        # plus windowless rows (first L-1 per CMA) masked by evaluation.
        L = int(obj["lookback"])
        assert valid <= len(y_test) - n_cmas
        assert valid >= len(y_test) - n_cmas * L
    else:
        assert valid == len(y_test)
    assert test_eval.loc[
        test_eval[f"{TARGET}_pred"].notna(), f"{TARGET}_pred"
    ].map(math.isfinite).all()


def test_metrics_json_has_lstm_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "lstm_rnn" in data
    entry = data["lstm_rnn"]
    for key in ("nrmse_train", "nrmse_test", "mda_test", "nmae_test"):
        assert key in entry, f"missing key: {key}"
        assert math.isfinite(entry[key]), f"non-finite {key}: {entry[key]}"
