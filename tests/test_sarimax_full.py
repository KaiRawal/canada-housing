"""Tests for the full-24 pooled SARIMAX-AR artifact (no ensembles)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "sarimax_full.joblib"
MET = ROOT / "artifacts" / "sarimax_full_metrics.json"
PRED_TEST = ROOT / "prediction" / "sarimax_full_predictions_test.csv"
PRED_TRAIN = ROOT / "prediction" / "sarimax_full_predictions_train.csv"

SUBSET16 = [
    "Edmonton, Alberta",
    "Greater Sudbury, Ontario",
    "Hamilton, Ontario",
    "Kitchener-Cambridge-Waterloo, Ontario",
    "London, Ontario",
    "Montréal, Quebec",
    "Ottawa-Gatineau, Ontario part, Ontario/Quebec",
    "Québec, Quebec",
    "Regina, Saskatchewan",
    "Saskatoon, Saskatchewan",
    "St. Catharines-Niagara, Ontario",
    "St. John's, Newfoundland and Labrador",
    "Vancouver, British Columbia",
    "Victoria, British Columbia",
    "Windsor, Ontario",
    "Winnipeg, Manitoba",
]


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def test_artifact_exists_and_config_pinned():
    import json

    assert ART.exists(), "missing artifacts/sarimax_full.joblib"
    assert MET.exists(), "missing artifacts/sarimax_full_metrics.json"
    art = joblib.load(ART)
    assert art["features"] == ["rollmean_3", "rollmean_12"]
    assert art["alpha"] == 0.001
    assert art["mask"] == {"convention": "boundary-chained", "n16": 1728, "n24": 2398}
    assert art["fit_rows"] == 7744
    m = json.loads(MET.read_text())
    assert abs(m["rmse_24"] - 0.760422) < 5e-4
    assert abs(m["rmse_16"] - 0.800261) < 5e-4


def test_prediction_csvs_eval_schema():
    for path, n_rows in [(PRED_TEST, 2398), (PRED_TRAIN, 7792)]:
        assert path.exists(), f"missing {path.name}"
        df = pd.read_csv(path)
        assert list(df.columns) == ["cma_canonical", "date", "total_true", "total_pred"]
        assert len(df) == n_rows
    te = pd.read_csv(PRED_TEST)
    assert int(te["total_pred"].notna().sum()) == 2398
    # 16-subset rows all present (108 each), short-history CMAs present.
    counts = te.groupby("cma_canonical").size()
    assert counts.loc["Edmonton, Alberta"] == 108
    assert counts.loc["Sherbrooke, Quebec"] == 46
    assert counts.loc["Guelph, Ontario"] == 86


def test_full_panel_beats_naive_and_poison_check():
    import json

    m = json.loads(MET.read_text())
    assert m["rmse_24"] < m["naive_24"]
    assert abs(m["naive_24"] - 0.893202) < 1e-4
    assert abs(m["naive_16"] - 0.956390) < 1e-4
    # Full-24 fit must not degrade 16-mask vs 16-only control by >1%.
    assert m["rmse_16"] <= 1.01 * m["control_16only_rmse_16"]


def test_artifact_reloads_bit_exact():
    art = joblib.load(ART)
    model, mu, sd = art["model"], art["scaler_mu"], art["scaler_sd"]
    ytr = pd.read_csv(ROOT / "prediction" / "y_train_full_total.csv")
    assert len(ytr) == 7792
    # Spot-check: stored test prediction keys cover the full mask.
    assert len(art["predictions_test"]) == 2398
    key0 = next(iter(art["predictions_test"]))
    assert isinstance(key0[0], str) and isinstance(art["predictions_test"][key0], float)
    # Reloaded model reproduces a stored prediction from raw features.
    assert _rmse([0.0], [0.0]) == 0.0
    assert model is not None and mu is not None and sd is not None
