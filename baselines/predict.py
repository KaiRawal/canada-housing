"""Reload baseline artifacts and recompute metrics on train+test.

Extensible mirror of baselines/train.py: each variant registers a
predictor in PREDICTORS (dict name -> callable). Prints one JSON line:
{"pass_rate": <float>, "models": {<variant>: {nrmse_test, ...}}}.

Usage:
    .venv/bin/python baselines/predict.py --target total
    .venv/bin/python baselines/predict.py --target total --variant persistence
"""

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluation import calculate_all_metrics  # noqa: E402
from prediction.prediction import calculate_persistence_predictions  # noqa: E402

PREDICTION_DIR = ROOT / "prediction"
DEFAULT_OUT = ROOT / "artifacts"

FUTURE_VARIANTS: list = []

# Artifact filename aliases: variant -> on-disk stem when the file is not
# <variant>.joblib (lstm_rnn persists to deep_learning.joblib per
# problem.yaml deliverables).
ARTIFACT_FILENAMES = {
    "lstm_rnn": "deep_learning",
}


def load_y(target: str):
    train_path = PREDICTION_DIR / f"y_train_full_{target}.csv"
    test_path = PREDICTION_DIR / f"y_test_full_{target}.csv"
    y_train = pd.read_csv(train_path, parse_dates=["date"])
    y_test = pd.read_csv(test_path, parse_dates=["date"])
    return y_train, y_test


def build_eval_frame(y_df: pd.DataFrame, preds: pd.Series, target: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cma_canonical": y_df["cma_canonical"].values,
            "date": pd.to_datetime(y_df["date"]).values,
            f"{target}_true": y_df[target].values,
            f"{target}_pred": preds.values,
        }
    )
    return out.sort_values(["cma_canonical", "date"]).reset_index(drop=True)


def load_artifact(variant: str, out_dir: Path):
    path = out_dir / f"{variant}.joblib"
    if not path.exists() and variant in ARTIFACT_FILENAMES:
        path = out_dir / f"{ARTIFACT_FILENAMES[variant]}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def predict_persistence(target: str, out_dir: Path):
    """Reload persistence artifact, recompute train/test eval frames."""
    load_artifact("persistence", out_dir)  # validates artifact reloads
    y_train, y_test = load_y(target)
    train_preds = calculate_persistence_predictions(y_train, target)
    test_preds = calculate_persistence_predictions(y_test, target)
    train_eval = build_eval_frame(y_train, train_preds, target)
    test_eval = build_eval_frame(y_test, test_preds, target)
    return train_eval, test_eval


def predict_linear_regression(target: str, out_dir: Path):
    """Reload linear_regression artifact (TRY-1 lags+blend / TRY-2 dedup+blend).

    Rebuilds the causal lag features via assemble() on train+test y (same
    as training, so test rows see the train-tail history), predicts the
    stored Pipeline on rows with complete features, and returns
    blend(model, persistence, blend_w). TRY-2 artifacts carry deduped
    ``feature_cols`` (subset of exo+lag); prediction selects those model
    columns (for TRY-1 artifacts feature_cols == exo+lag, identical to
    before). Warmup rows (NaN lag features) and first-row-per-CMA rows
    (NaN persistence) yield NaN predictions, which evaluation masks.
    Falls back to plain contemporaneous prediction for pre-TRY-1
    artifacts without lag/blend keys.
    """
    import numpy as np

    from baselines._nesting import assemble, blend, persistence_pred

    artifact = load_artifact("linear_regression", out_dir)  # validates reload
    model = artifact["model"]
    blend_w = float(artifact.get("blend_w", 1.0))
    lag_feature_cols = list(artifact.get("lag_feature_cols", []))
    if "exo_feature_cols" in artifact:
        exo_feature_cols = list(artifact["exo_feature_cols"])
    else:
        exo_feature_cols = list(artifact["feature_cols"])
    y_train, y_test = load_y(target)
    X_train = pd.read_csv(PREDICTION_DIR / f"X_train_full_{target}.csv")
    X_test = pd.read_csv(PREDICTION_DIR / f"X_test_full_{target}.csv")

    if not lag_feature_cols:
        train_preds = pd.Series(
            model.predict(X_train[exo_feature_cols]).astype(float),
            index=y_train.index,
        )
        test_preds = pd.Series(
            model.predict(X_test[exo_feature_cols]).astype(float),
            index=y_test.index,
        )
        return build_eval_frame(y_train, train_preds, target), build_eval_frame(
            y_test, test_preds, target
        )

    lag_parts = assemble({"train": y_train, "test": y_test})
    # TRY-2 artifacts store deduped model columns in feature_cols; TRY-1
    # artifacts store the full exo+lag list (identical fallback).
    cols = list(artifact.get("feature_cols", exo_feature_cols + lag_feature_cols))

    def _model_preds(X_full: pd.DataFrame, lag: pd.DataFrame, y_df: pd.DataFrame):
        F_full = pd.concat(
            [X_full.reset_index(drop=True), lag.reset_index(drop=True)], axis=1
        )
        F_full.index = y_df.index
        out = pd.Series(np.nan, index=y_df.index, dtype=float)
        ok = F_full[cols].notna().all(axis=1).to_numpy()
        if ok.any():
            out.loc[F_full.index[ok]] = model.predict(F_full.loc[ok, cols])
        return out

    train_model = _model_preds(X_train, lag_parts["train"], y_train)
    test_model = _model_preds(X_test, lag_parts["test"], y_test)
    train_preds = blend(train_model, persistence_pred(y_train), blend_w)
    train_preds.index = y_train.index
    test_preds = blend(test_model, persistence_pred(y_test), blend_w)
    test_preds.index = y_test.index
    train_eval = build_eval_frame(y_train, train_preds, target)
    test_eval = build_eval_frame(y_test, test_preds, target)
    return train_eval, test_eval


def predict_gbt_ensemble(target: str, out_dir: Path):
    """Reload gbt_ensemble artifact (TRY-3 regime / TRY-2 calibrated / TRY-1).

    TRY-3 artifacts carry per-regime affine delta calibration
    (``regime_thr``/``regime_small_a``/``regime_small_b``/``regime_large_a``/
    ``regime_large_b`` + ``regime_col``, rule ``large if |mom| > thr else
    small`` with the VAL-derived threshold) plus the TRY-2 scalar
    ``calib_a``/``calib_b`` fallback: rebuild causal lags via assemble()
    (test sees the train-tail history), predict the stored HistGB
    (delta -> level as lag_1 + per-regime (a*delta + b)), and return
    blend(model, persistence, blend_w). TRY-3 ``direction-overlay``
    artifacts (only if regime had failed) carry ``overlay_k`` applied as
    k*momentum where |momentum| > thr on top of the global calibration.
    TRY-2 artifacts (scalar ``calib_a``/``calib_b`` + ``momentum_defs``)
    use the global calibration. Warmup rows (NaN lag features) and
    first-row-per-CMA rows (NaN persistence) yield NaN predictions, which
    evaluation masks. Falls back to plain contemporaneous prediction for
    pre-TRY-1 artifacts without lag/blend keys.
    """
    import numpy as np

    from baselines._nesting import assemble, blend, persistence_pred

    artifact = load_artifact("gbt_ensemble", out_dir)  # validates reload
    model = artifact["model"]
    blend_w = float(artifact.get("blend_w", 1.0))
    lag_feature_cols = list(artifact.get("lag_feature_cols", []))
    formulation = str(artifact.get("formulation", "level"))
    lag_col = str(artifact.get("lag_col", "lag_1"))
    calib_a = float(artifact.get("calib_a", 1.0))
    calib_b = float(artifact.get("calib_b", 0.0))
    momentum_defs = dict(artifact.get("momentum_defs", {}))
    adopted = str(artifact.get("adopted", ""))
    regime_col = str(artifact.get("regime_col", "mom_lag1_lag12"))
    regime_thr = artifact.get("regime_thr", None)
    regime_thr = None if regime_thr is None else float(regime_thr)
    overlay_k = artifact.get("overlay_k", None)
    overlay_k = None if overlay_k is None else float(overlay_k)
    if "exo_feature_cols" in artifact:
        exo_feature_cols = list(artifact["exo_feature_cols"])
    else:
        exo_feature_cols = list(artifact["feature_cols"])
    y_train, y_test = load_y(target)
    X_train = pd.read_csv(PREDICTION_DIR / f"X_train_full_{target}.csv")
    X_test = pd.read_csv(PREDICTION_DIR / f"X_test_full_{target}.csv")

    if not lag_feature_cols or "blend_w" not in artifact:
        feature_cols = artifact["feature_cols"]
        train_preds = pd.Series(
            model.predict(X_train[feature_cols]).astype(float), index=y_train.index
        )
        test_preds = pd.Series(
            model.predict(X_test[feature_cols]).astype(float), index=y_test.index
        )
        train_eval = build_eval_frame(y_train, train_preds, target)
        test_eval = build_eval_frame(y_test, test_preds, target)
        return train_eval, test_eval

    lag_parts = assemble({"train": y_train, "test": y_test})
    cols = list(artifact.get("feature_cols", exo_feature_cols + lag_feature_cols))

    def _momentum_frame(lag: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                new: lag[pos].to_numpy(dtype=float) - lag[neg].to_numpy(dtype=float)
                for new, (pos, neg) in momentum_defs.items()
            },
            index=lag.index,
        )

    def _level_preds(X_full: pd.DataFrame, lag: pd.DataFrame, y_df: pd.DataFrame):
        F_full = pd.concat(
            [X_full.reset_index(drop=True), lag.reset_index(drop=True)], axis=1
        )
        F_full.index = y_df.index
        if momentum_defs:
            mom = _momentum_frame(lag)
            mom.index = y_df.index
            F_full = pd.concat([F_full, mom], axis=1)
        lag1_s = lag[lag_col] if lag_col in lag.columns else lag.iloc[:, 0]
        lag1_s.index = y_df.index
        out = pd.Series(np.nan, index=y_df.index, dtype=float)
        if formulation == "delta":
            ok = F_full[cols].notna().all(axis=1).to_numpy() & lag1_s.notna().to_numpy()
            if ok.any():
                d = np.asarray(model.predict(F_full.loc[ok, cols]), dtype=float)
                base = lag1_s.to_numpy()[ok] + (
                    calib_a * np.asarray(d, dtype=float) + calib_b
                )
                if adopted == "regime-split" and regime_thr is not None:
                    # Per-regime calibration: large if |mom| > thr else small.
                    a_s = float(artifact["regime_small_a"])
                    b_s = float(artifact["regime_small_b"])
                    a_l = float(artifact["regime_large_a"])
                    b_l = float(artifact["regime_large_b"])
                    legs = momentum_defs.get(
                        regime_col, momentum_defs.get("mom_lag1_lag12")
                    )
                    if legs is not None:
                        mom_s = (
                            lag[legs[0]].to_numpy(dtype=float)
                            - lag[legs[1]].to_numpy(dtype=float)
                        )
                        ok_idx = F_full.index[ok]
                        m = pd.Series(mom_s, index=y_df.index).loc[ok_idx].to_numpy(dtype=float)
                        lg = np.abs(m) > float(regime_thr)
                        cal = np.where(lg, a_l * d + b_l, a_s * d + b_s)
                        valid_mom = ~np.isnan(m)
                        cal = np.where(valid_mom, cal, calib_a * d + calib_b)
                        out.loc[F_full.index[ok]] = lag1_s.to_numpy()[ok] + cal
                    else:
                        out.loc[F_full.index[ok]] = base
                elif adopted == "direction-overlay" and overlay_k is not None:
                    legs = momentum_defs.get(
                        regime_col, momentum_defs.get("mom_lag1_lag12")
                    )
                    if legs is not None and regime_thr is not None:
                        mom_s = (
                            lag[legs[0]].to_numpy(dtype=float)
                            - lag[legs[1]].to_numpy(dtype=float)
                        )
                        ok_idx = F_full.index[ok]
                        m = pd.Series(mom_s, index=y_df.index).loc[ok_idx].to_numpy(dtype=float)
                        kick = np.where(
                            np.abs(m) > float(regime_thr),
                            float(overlay_k) * np.nan_to_num(m, nan=0.0), 0.0,
                        )
                        out.loc[F_full.index[ok]] = base + kick
                    else:
                        out.loc[F_full.index[ok]] = base
                else:
                    out.loc[F_full.index[ok]] = base
        else:
            ok = F_full[cols].notna().all(axis=1).to_numpy()
            if ok.any():
                out.loc[F_full.index[ok]] = model.predict(F_full.loc[ok, cols])
        return out

    train_model = _level_preds(X_train, lag_parts["train"], y_train)
    test_model = _level_preds(X_test, lag_parts["test"], y_test)
    train_preds = blend(train_model, persistence_pred(y_train), blend_w)
    train_preds.index = y_train.index
    test_preds = blend(test_model, persistence_pred(y_test), blend_w)
    test_preds.index = y_test.index
    train_eval = build_eval_frame(y_train, train_preds, target)
    test_eval = build_eval_frame(y_test, test_preds, target)
    return train_eval, test_eval


def predict_prophet(target: str, out_dir: Path):
    """Reload prophet artifact, forecast train+test dates (TRY-2 aware).

    TRY-2 artifacts (``regressor_cols`` + ``calib_a``/``calib_b`` +
    ``blend_w`` + ``cal_order``): rebuild causal lags via assemble()
    (test sees the train-tail history, same as training), forecast per
    CMA with the stored models (future frame carries the regressor
    columns), then apply the winning cal_order -- ``cal-before-blend``
    (calibrate raw then blend, TRY-1 path) or ``cal-after-blend``
    (blend raw with persistence first, then global affine) -- no
    refitting. TRY-1 artifacts (no ``cal_order``) use the
    cal-before-blend path. Run-1 univariate artifacts (no
    ``regressor_cols``) use the legacy date-only forecast path.
    """
    import logging

    import numpy as np

    from baselines._nesting import assemble, blend, persistence_pred

    _cs_logger = logging.getLogger("cmdstanpy")
    if not _cs_logger.hasHandlers():
        _cs_logger.addHandler(logging.NullHandler())
    _cs_logger.setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    artifact = load_artifact("prophet", out_dir)  # validates reload
    models = artifact["models"]
    y_train, y_test = load_y(target)
    regressors = list(artifact.get("regressor_cols", []))

    if not regressors:
        # Run-1 univariate path: date-only futures, no calibration/blend.

        def forecast(y_df: pd.DataFrame) -> pd.Series:
            preds = pd.Series(np.nan, index=y_df.index, dtype=float)
            for cma, g in y_df.groupby("cma_canonical"):
                m = models.get(cma)
                if m is None:
                    continue
                future = pd.DataFrame({"ds": pd.to_datetime(g["date"])})
                preds.loc[g.index] = m.predict(future)["yhat"].to_numpy(float)
            return preds

        train_preds = forecast(y_train)
        test_preds = forecast(y_test)
        train_eval = build_eval_frame(y_train, train_preds, target)
        test_eval = build_eval_frame(y_test, test_preds, target)
        return train_eval, test_eval

    lag_parts = assemble({"train": y_train, "test": y_test})
    calib_a = float(artifact.get("calib_a", 1.0))
    calib_b = float(artifact.get("calib_b", 0.0))
    blend_w = float(artifact.get("blend_w", 1.0))
    cal_order = str(artifact.get("cal_order", "cal-before-blend"))

    def forecast_raw(y_df: pd.DataFrame, lag: pd.DataFrame) -> pd.Series:
        model_preds = pd.Series(np.nan, index=y_df.index, dtype=float)
        for cma, g in y_df.groupby("cma_canonical"):
            m = models.get(cma)
            if m is None:
                continue
            gs = g.sort_values("date")
            if any(r not in lag.columns for r in regressors):
                continue
            fut = pd.DataFrame(
                {"ds": pd.to_datetime(gs["date"].values)}, index=gs.index
            )
            for r in regressors:
                fut[r] = lag.loc[gs.index, r].to_numpy(float)
            ok = fut[regressors].notna().all(axis=1).to_numpy()
            if not bool(ok.any()):
                continue
            raw = m.predict(fut.loc[ok])["yhat"].to_numpy(float)
            model_preds.loc[fut.index[ok]] = raw
        return model_preds

    def forecast_calibrated(y_df: pd.DataFrame, lag: pd.DataFrame) -> pd.Series:
        raw = forecast_raw(y_df, lag)
        out = pd.Series(np.nan, index=y_df.index, dtype=float)
        ok = raw.notna().to_numpy()
        out.loc[raw.index[ok]] = (
            calib_a * raw.loc[raw.index[ok]].to_numpy(dtype=float) + calib_b
        )
        return out

    if cal_order == "cal-after-blend":
        train_raw = forecast_raw(y_train, lag_parts["train"])
        test_raw = forecast_raw(y_test, lag_parts["test"])
        _tr_blended = blend(train_raw, persistence_pred(y_train), blend_w)
        train_preds = pd.Series(
            calib_a * _tr_blended.to_numpy(dtype=float) + calib_b,
            index=y_train.index,
        )
        _te_blended = blend(test_raw, persistence_pred(y_test), blend_w)
        test_preds = pd.Series(
            calib_a * _te_blended.to_numpy(dtype=float) + calib_b,
            index=y_test.index,
        )
    else:
        train_model = forecast_calibrated(y_train, lag_parts["train"])
        test_model = forecast_calibrated(y_test, lag_parts["test"])
        train_preds = blend(train_model, persistence_pred(y_train), blend_w)
        test_preds = blend(test_model, persistence_pred(y_test), blend_w)
    train_preds.index = y_train.index
    test_preds.index = y_test.index
    train_eval = build_eval_frame(y_train, train_preds, target)
    test_eval = build_eval_frame(y_test, test_preds, target)
    return train_eval, test_eval


def predict_lstm_rnn(target: str, out_dir: Path):
    """Reload lstm_rnn artifact (deep_learning.joblib) + one-step-ahead forecast.

    Dispatches on artifact kind: legacy run-1 univariate artifacts
    (``lstm-global-univariate``, no ``feature_cols``) use the per-CMA
    standardized lag-window path; try-1/try-2 multivariate-delta artifacts
    (``lstm-global-multivariate-delta``) rebuild the deduped feature
    frame via assemble() (test sees the train-tail history, same as
    training), run windowed delta preds per CMA (test chains the
    train-tail windows), nest to levels as lag_1 + (a*delta + b) with
    the stored val calibration (try-2 regime-split artifacts apply the
    per-regime affine with the VAL-derived threshold), and blend with
    persistence. Rows
    without a full window / with NaN features yield NaN model preds
    (evaluation masks them; blend propagates the persistence NaN on
    first-row-per-CMA, GBT-style).
    """
    import numpy as np
    import torch

    from baselines._lstm_rnn import (
        build_model_from_state,
        delta_level_predict_windows,
        make_windows_multi,
        onestep_predict_frame,
        standardize_frame,
    )
    from baselines._nesting import assemble, blend, persistence_pred

    artifact = load_artifact("lstm_rnn", out_dir)  # validates reload
    assert artifact.get("variant") == "lstm_rnn"

    if artifact.get("kind") != "lstm-global-multivariate-delta":
        assert "scalers" in artifact and "state_dict" in artifact
        torch.manual_seed(int(artifact.get("seed", 42)))
        device = "cpu"  # inference on CPU: deterministic, no MPS transfer needed
        model = build_model_from_state(
            artifact["state_dict"],
            hidden_size=int(artifact["hidden_size"]),
            num_layers=int(artifact["num_layers"]),
            device=device,
        )
        scalers = artifact["scalers"]
        lookback = int(artifact["lookback"])
        y_train, y_test = load_y(target)
        train_preds = onestep_predict_frame(y_train, target, scalers, model, lookback, device)
        test_preds = onestep_predict_frame(y_test, target, scalers, model, lookback, device)
        train_eval = build_eval_frame(y_train, train_preds, target)
        test_eval = build_eval_frame(y_test, test_preds, target)
        return train_eval, test_eval

    # ---- Try-1/try-2 multivariate-delta path (try-2 adds regime-split) ----
    torch.manual_seed(int(artifact.get("seed", 42)))
    device = "cpu"
    model_cols = list(artifact["feature_cols"])
    lag_col = str(artifact.get("lag_col", "lag_1"))
    lookback = int(artifact["lookback"])
    feat_scalers = dict(artifact["feature_scalers"])
    delta_scalers = dict(artifact["scalers"])
    calib_a = float(artifact.get("calib_a", 1.0))
    calib_b = float(artifact.get("calib_b", 0.0))
    blend_w = float(artifact.get("blend_w", 1.0))
    adopted = str(artifact.get("adopted", "global"))
    regime_thr = artifact.get("regime_thr", None)
    regime_thr = None if regime_thr is None else float(regime_thr)
    use_regime = bool(adopted == "regime-split" and regime_thr is not None)
    if use_regime:
        regime_small_a = float(artifact["regime_small_a"])
        regime_small_b = float(artifact["regime_small_b"])
        regime_large_a = float(artifact["regime_large_a"])
        regime_large_b = float(artifact["regime_large_b"])
        momentum_defs = dict(artifact.get("momentum_defs", {}))
        regime_col = str(artifact.get("regime_col", "mom_lag1_lag12"))
        legs = momentum_defs.get(regime_col)
        if legs is None:  # fallback to canonical legs
            legs = ["lag_1", "lag_12"]
    else:
        regime_small_a = regime_small_b = None
        regime_large_a = regime_large_b = None
        legs = None
    model = build_model_from_state(
        artifact["state_dict"],
        hidden_size=int(artifact["hidden_size"]),
        num_layers=int(artifact["num_layers"]),
        device=device,
        input_size=int(artifact.get("input_size", len(model_cols))),
    )
    y_train, y_test = load_y(target)
    X_train = pd.read_csv(PREDICTION_DIR / f"X_train_full_{target}.csv")
    X_test = pd.read_csv(PREDICTION_DIR / f"X_test_full_{target}.csv")
    lag_parts = assemble({"train": y_train, "test": y_test})

    def _F(X_full: pd.DataFrame, lag: pd.DataFrame, y_df: pd.DataFrame):
        F_full = pd.concat(
            [X_full.reset_index(drop=True), lag.reset_index(drop=True)], axis=1
        )
        F_full.index = y_df.index
        return F_full

    F_train = _F(X_train, lag_parts["train"], y_train)
    F_test = _F(X_test, lag_parts["test"], y_test)
    Z_train_full = standardize_frame(F_train[model_cols], model_cols,
                                     feat_scalers)

    def _level_preds(y_df: pd.DataFrame, F_full: pd.DataFrame,
                     lag: pd.DataFrame, chain_Z=None) -> pd.Series:
        out = pd.Series(np.nan, index=y_df.index, dtype=float)
        F_std = (Z_train_full if chain_Z is None else
                 standardize_frame(F_full[model_cols], model_cols,
                                   feat_scalers))
        for cma, g in y_df.groupby("cma_canonical"):
            gs = g.sort_values("date")
            idx = gs.index.to_numpy()
            Z = F_std[F_full.index.get_indexer(idx)]
            ok_feat = ~np.isnan(Z).any(axis=1)
            lag1 = lag.loc[idx, lag_col].to_numpy(dtype=float)
            if chain_Z is not None:
                Zc = np.concatenate(
                    [np.asarray(chain_Z[str(cma)][-lookback:]), Z], axis=0)
                Xw, pw = make_windows_multi(Zc, lookback)
                keep = pw >= lookback
                Xw, pw = Xw[keep], pw[keep]
                tgt = pw - lookback
            else:
                Xw, pw = make_windows_multi(Z, lookback)
                tgt = pw
            if len(pw) == 0:
                continue
            if use_regime:
                lvl_raw = delta_level_predict_windows(
                    model, Xw, lag1[tgt], [str(cma)] * len(tgt),
                    delta_scalers, device, calib_a=1.0, calib_b=0.0)
                d_raw = lvl_raw - lag1[tgt]
                try:
                    mom_s = (lag.loc[idx[tgt], legs[0]].to_numpy(dtype=float)
                             - lag.loc[idx[tgt], legs[1]].to_numpy(dtype=float))
                except Exception:  # noqa: BLE001
                    mom_s = np.full(len(tgt), np.nan)
                lg = np.abs(mom_s) > float(regime_thr)
                cal = np.where(lg, regime_large_a * d_raw + regime_large_b,
                               regime_small_a * d_raw + regime_small_b)
                cal = np.where(np.isnan(mom_s), calib_a * d_raw + calib_b, cal)
                lvl = lag1[tgt] + cal
            else:
                lvl = delta_level_predict_windows(
                    model, Xw, lag1[tgt], [str(cma)] * len(tgt),
                    delta_scalers, device, calib_a=calib_a, calib_b=calib_b)
            good = ok_feat[tgt] & ~np.isnan(lag1[tgt])
            out.loc[idx[tgt]] = np.where(good, lvl, np.nan)
        return out

    # Per-CMA train-tail standardized blocks for test chaining.
    chain: dict = {}
    for cma, g in y_train.groupby("cma_canonical"):
        gs = g.sort_values("date")
        chain[str(cma)] = Z_train_full[F_train.index.get_indexer(
            gs.index.to_numpy())]
    train_model = _level_preds(y_train, F_train, lag_parts["train"], None)
    test_model = _level_preds(y_test, F_test, lag_parts["test"], chain)
    train_preds = blend(train_model, persistence_pred(y_train), blend_w)
    train_preds.index = y_train.index
    test_preds = blend(test_model, persistence_pred(y_test), blend_w)
    test_preds.index = y_test.index
    train_eval = build_eval_frame(y_train, train_preds, target)
    test_eval = build_eval_frame(y_test, test_preds, target)
    return train_eval, test_eval


def predict_pretrained_ts(target: str, out_dir: Path):
    """Reload pretrained_ts artifact, regenerate Chronos forecasts, recompute eval.

    TRY-1 artifacts (``calib_a``/``calib_b`` + ``blend_w`` + ``cal_order``):
    regenerate RAW zero-shot forecasts then apply the winning transform --
    ``cal-before-blend`` (calibrate raw then blend with one-step
    persistence) or ``cal-after-blend`` (blend raw with persistence first,
    then global affine) -- no refitting. Run-1 artifacts (no ``calib_a``)
    use the legacy raw path.
    """
    import numpy as np

    from baselines._nesting import blend, persistence_pred
    from baselines._pretrained_ts import (
        backtest_train_per_cma,
        forecast_test_per_cma,
        load_pipeline,
        series_by_cma,
    )

    artifact = load_artifact("pretrained_ts", out_dir)  # validates reload
    assert artifact.get("variant") == "pretrained_ts"
    assert "test_forecasts" in artifact and "train_forecasts" in artifact
    best = artifact.get("best_params", {})
    choice = str(best.get("context", "full"))
    device = str(best.get("device", "cpu"))
    y_train, y_test = load_y(target)

    train_mask = y_train[target].notna().to_numpy()
    ytr = y_train.loc[train_mask].reset_index(drop=True)
    test_mask = y_test[target].notna().to_numpy()
    yte = y_test.loc[test_mask].reset_index(drop=True)
    train_series = series_by_cma(ytr, target)
    test_series = series_by_cma(yte, target)

    # Reload weights via from_pretrained (never pickled); stored
    # forecasts are the offline fallback so metrics stay reproducible.
    pipe = None
    for dev in ([device] if device == "cpu" else [device, "cpu"]):
        try:
            pipe = load_pipeline(dev)
            device = dev
            break
        except Exception as exc:  # noqa: BLE001
            print(f"pretrained_ts predict: from_pretrained failed on {dev}: {exc}")
    if pipe is None:
        test_fc = {c: np.asarray(a, dtype=float) for c, a in artifact["test_forecasts"].items()}
        train_fc = {c: np.asarray(a, dtype=float) for c, a in artifact["train_forecasts"].items()}
    else:
        test_fc = forecast_test_per_cma(pipe, train_series, test_series, choice)
        for cma in test_series:
            if cma not in test_fc:
                test_fc[cma] = np.full(len(test_series[cma]), np.nan)
        train_fc = backtest_train_per_cma(pipe, train_series, choice)

    train_preds = pd.Series(np.nan, index=ytr.index, dtype=float)
    for cma, g in ytr.groupby("cma_canonical"):
        gs = g.sort_values("date")
        arr = np.asarray(train_fc[str(cma)], dtype=float)
        train_preds.loc[gs.index] = arr[: len(gs)]
    train_preds_full = pd.Series(np.nan, index=y_train.index, dtype=float)
    train_preds_full.loc[ytr.index] = train_preds.values
    test_preds = pd.Series(np.nan, index=yte.index, dtype=float)
    for cma, g in yte.groupby("cma_canonical"):
        gs = g.sort_values("date")
        arr = np.asarray(test_fc[str(cma)], dtype=float)
        test_preds.loc[gs.index] = arr[: len(gs)]
    test_preds_full = pd.Series(np.nan, index=y_test.index, dtype=float)
    test_preds_full.loc[yte.index] = test_preds.values

    if "calib_a" in artifact:  # TRY-1 calibrated+blended path.
        calib_a = float(artifact.get("calib_a", 1.0))
        calib_b = float(artifact.get("calib_b", 0.0))
        blend_w = float(artifact.get("blend_w", 0.0))
        cal_order = str(artifact.get("cal_order", "cal-before-blend"))
        train_persist = persistence_pred(y_train)
        test_persist = persistence_pred(y_test)
        if cal_order == "cal-after-blend":
            _tr_blended = blend(train_preds_full, train_persist, blend_w)
            train_preds_full = pd.Series(
                calib_a * _tr_blended.to_numpy(dtype=float) + calib_b,
                index=y_train.index,
            )
            _te_blended = blend(test_preds_full, test_persist, blend_w)
            test_preds_full = pd.Series(
                calib_a * _te_blended.to_numpy(dtype=float) + calib_b,
                index=y_test.index,
            )
        else:
            _tr_cal = pd.Series(
                np.where(train_preds_full.notna().to_numpy(),
                         calib_a * train_preds_full.to_numpy(dtype=float) + calib_b,
                         np.nan),
                index=y_train.index,
            )
            train_preds_full = blend(_tr_cal, train_persist, blend_w)
            _te_cal = pd.Series(
                np.where(test_preds_full.notna().to_numpy(),
                         calib_a * test_preds_full.to_numpy(dtype=float) + calib_b,
                         np.nan),
                index=y_test.index,
            )
            test_preds_full = blend(_te_cal, test_persist, blend_w)
        train_preds_full.index = y_train.index
        test_preds_full.index = y_test.index

    train_eval = build_eval_frame(y_train, train_preds_full, target)
    test_eval = build_eval_frame(y_test, test_preds_full, target)
    return train_eval, test_eval


def _not_implemented(name: str):
    def _predictor(target: str, out_dir: Path):
        raise NotImplementedError(
            f"Variant '{name}' is registered as a placeholder and not implemented yet."
        )

    _predictor.__name__ = f"predict_{name}"
    return _predictor


PREDICTORS = {"persistence": predict_persistence, "linear_regression": predict_linear_regression, "gbt_ensemble": predict_gbt_ensemble, "prophet": predict_prophet, "lstm_rnn": predict_lstm_rnn, "pretrained_ts": predict_pretrained_ts}
for _name in FUTURE_VARIANTS:
    PREDICTORS[_name] = _not_implemented(_name)


def evaluate_variant(variant: str, target: str, out_dir: Path) -> dict:
    train_eval, test_eval = PREDICTORS[variant](target, out_dir)
    m_train = calculate_all_metrics(train_eval, target)
    m_test = calculate_all_metrics(test_eval, target)
    return {
        "entry": {
            "nrmse_train": float(m_train["NRMSE"]),
            "nrmse_test": float(m_test["NRMSE"]),
            "mda_test": float(m_test["MDA"]),
            "nmae_test": float(m_test["NMAE"]),
        },
        "n_test_rows": int(len(test_eval)),
        "n_test_valid": int(
            (
                test_eval[f"{target}_true"].notna()
                & test_eval[f"{target}_pred"].notna()
            ).sum()
        ),
    }


def discover_variants(out_dir: Path):
    """Variants with a dumped artifact, falling back to metrics.json keys."""
    stems = sorted(p.stem for p in out_dir.glob("*.joblib"))
    # Map aliased filenames (e.g. deep_learning.joblib -> lstm_rnn).
    reverse_alias = {v: k for k, v in ARTIFACT_FILENAMES.items()}
    found = []
    for stem in stems:
        name = reverse_alias.get(stem, stem)
        if name in PREDICTORS and name not in found:
            found.append(name)
    if found:
        return found
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            return [v for v in data if v in PREDICTORS]
        except json.JSONDecodeError:
            pass
    return ["persistence"] if "persistence" in PREDICTORS else []


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Recompute baseline metrics.")
    p.add_argument("--target", default="total")
    p.add_argument("--variant", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    variants = [args.variant] if args.variant else discover_variants(out_dir)
    models = {}
    n_pass = 0
    for variant in variants:
        if variant not in PREDICTORS:
            continue
        try:
            result = evaluate_variant(variant, args.target, out_dir)
            entry = result["entry"]
            models[variant] = entry
            if math.isfinite(entry.get("nrmse_test", float("nan"))):
                n_pass += 1
        except (FileNotFoundError, NotImplementedError):
            continue
    pass_rate = (n_pass / len(variants)) if variants else 0.0
    print(json.dumps({"pass_rate": pass_rate, "models": models}))
    return {"pass_rate": pass_rate, "models": models}


if __name__ == "__main__":
    main()
