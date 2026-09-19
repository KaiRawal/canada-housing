"""Tests for the nesting foundation (baselines/_nesting.py)."""

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines._nesting import (  # noqa: E402
    assemble,
    blend,
    build_lag_features,
    persistence_pred,
    tune_blend,
)

TARGET = "total"


def _synthetic():
    return pd.DataFrame(
        {
            "cma_canonical": ["A"] * 5 + ["B"] * 4,
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-05-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                ]
            ),
            "total": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0],
        }
    )


def test_lag1_equals_shifted_total_per_cma():
    y = _synthetic()
    feats = build_lag_features(y)
    assert len(feats) == len(y)
    assert (feats.index == y.index).all()
    # lag_1[t] == total[t-1] per CMA; first row per CMA is NaN (warmup).
    for cma, g in y.groupby("cma_canonical"):
        idx = g.sort_values("date").index.tolist()
        for j, row_ix in enumerate(idx):
            got = feats.loc[row_ix, "lag_1"]
            if j == 0:
                assert pd.isna(got), f"first row of {cma} must be NaN"
            else:
                prev_ix = idx[j - 1]
                assert got == y.loc[prev_ix, "total"]
    # Deeper lags also causal: lag_2 warmup is 2 NaNs per CMA.
    for cma, g in y.groupby("cma_canonical"):
        idx = g.sort_values("date").index.tolist()
        assert pd.isna(feats.loc[idx[0], "lag_2"])
        assert pd.isna(feats.loc[idx[1], "lag_2"])
        assert feats.loc[idx[2], "lag_2"] == y.loc[idx[0], "total"]
    # assemble() builds once over the full timeline (smoke: same values).
    parts = assemble({"train": y.iloc[:6], "test": y.iloc[6:]})
    assert set(parts) == {"train", "test"}


def test_blend_w0_reproduces_persistence_on_real_train():
    y_train = pd.read_csv(
        ROOT / "prediction" / f"y_train_full_{TARGET}.csv", parse_dates=["date"]
    )
    persist = persistence_pred(y_train)
    model = pd.Series(y_train[TARGET].to_numpy(dtype=float), index=y_train.index)
    out = blend(model, persist, 0.0)
    pd.testing.assert_series_equal(
        out, pd.Series(persist), check_names=False, check_dtype=False
    )
    assert out.isna().equals(persist.isna())
    # Mask propagation for a nonzero weight too.
    out_half = blend(model, persist, 0.5)
    assert out_half.isna().equals(persist.isna())


def test_tune_blend_returns_grid_w_with_finite_nrmse():
    y_train = pd.read_csv(
        ROOT / "prediction" / f"y_train_full_{TARGET}.csv", parse_dates=["date"]
    )
    persist = persistence_pred(y_train)
    model = pd.Series(y_train[TARGET].to_numpy(dtype=float), index=y_train.index)
    val_true = y_train[TARGET]
    best_w, best_nrmse = tune_blend(val_true, model, persist)
    assert best_w in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert math.isfinite(best_nrmse)
