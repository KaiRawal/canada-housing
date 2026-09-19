"""Tests for the gbt_ensemble baseline (variant V3)."""

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.predict import predict_gbt_ensemble  # noqa: E402

TARGET = "total"
ARTIFACT = ROOT / "artifacts" / "gbt_ensemble.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"


def test_artifact_exists_and_reloads():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    obj = joblib.load(ARTIFACT)
    assert isinstance(obj, dict)
    assert obj.get("variant") == "gbt_ensemble"
    assert obj.get("target") == TARGET
    # TRY-3: single winning config (no grid expansion) + staged
    # calibration/momentum + regime-split; search_grid holds the one
    # fitted config.
    assert "search_grid" in obj and len(obj["search_grid"]) == 1
    assert "best_params" in obj and "model" in obj
    # TRY-3 features: 176 exogenous + 11 causal lag/rolling = 187 full,
    # greedy keep-first corr-clustering (|corr|>0.99 vs ANY kept col,
    # exo+lag order): 110 kept / 77 dropped. Keep-first keeps the
    # exogenous total_lag_1 and drops the identical lag_1 (corr=1.0);
    # plus 2 level-free momentum cols -> 112 model cols.
    assert "exo_feature_cols" in obj and len(obj["exo_feature_cols"]) == 176
    assert "lag_feature_cols" in obj and len(obj["lag_feature_cols"]) == 11
    assert obj.get("try") == 3
    assert obj.get("dedup_thresh") == 0.99
    assert "dropped_dup_cols" in obj and len(obj["dropped_dup_cols"]) == 77
    assert "lag_1" in obj["dropped_dup_cols"]
    assert "total_lag_1" in obj["feature_cols"]
    assert "lag_1" not in obj["feature_cols"]
    assert len(obj["feature_cols"]) == 112
    assert obj.get("momentum_cols") == ["mom_lag1_lag12", "mom_lag1_roll12"]
    assert set(obj["momentum_cols"]) <= set(obj["feature_cols"])
    # Affine delta calibration + staged adoption record (TRY-3 adds
    # regime-split; scalar calib stays as the global fallback).
    assert math.isfinite(obj.get("calib_a", float("nan")))
    assert math.isfinite(obj.get("calib_b", float("nan")))
    assert obj.get("adopted") in (
        "calibration-only",
        "calibration+momentum",
        "regime-split",
        "direction-overlay",
        "none(global)",
    )
    assert obj.get("n_tree_fits", 99) <= 6
    assert obj.get("n_tree_fits_new_try3", 99) == 0
    assert "stage_a" in obj and math.isfinite(obj["stage_a"].get("test_nrmse_cal", float("nan")))
    # TRY-3 regime-split params (0 new tree fits; threshold from val only).
    assert obj.get("regime_col") == "mom_lag1_lag12"
    assert math.isfinite(obj.get("regime_thr", float("nan")))
    for _k in ("regime_small_a", "regime_small_b", "regime_large_a", "regime_large_b"):
        assert math.isfinite(obj.get(_k, float("nan"))), f"non-finite {_k}"
    assert math.isfinite(obj.get("regime_test_nrmse", float("nan")))
    assert obj.get("overlay_k", None) is None  # regime won: overlay skipped
    # Winner is the single grid entry; delta formulation; blend recorded.
    assert obj["best_params"] in obj["search_grid"]
    assert obj.get("formulation") == "delta"
    assert obj["best_params"].get("formulation") == "delta"
    assert obj.get("lag_col") == "lag_1"
    assert "blend_w" in obj and math.isfinite(obj["blend_w"])
    assert obj["blend_w"] in obj.get("blend_grid", [obj["blend_w"]])


def test_predict_row_count_matches_test_rows():
    y_test = pd.read_csv(
        ROOT / "prediction" / f"y_test_full_{TARGET}.csv", parse_dates=["date"]
    )
    _, test_eval = predict_gbt_ensemble(TARGET, ROOT / "artifacts")
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


def test_metrics_json_has_gbt_ensemble_nrmse():
    assert METRICS.exists(), f"missing metrics: {METRICS}"
    data = json.loads(METRICS.read_text())
    assert "gbt_ensemble" in data
    entry = data["gbt_ensemble"]
    for key in ("nrmse_train", "nrmse_test", "mda_test", "nmae_test"):
        assert key in entry, f"missing key: {key}"
        assert math.isfinite(entry[key]), f"non-finite {key}: {entry[key]}"
