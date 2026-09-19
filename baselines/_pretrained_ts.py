"""Shared helpers for the pretrained_ts baseline (variant V6).

Zero-shot multi-step forecasting with Amazon Chronos-Bolt-Small
(`amazon/chronos-bolt-small` via the `chronos-forecasting` package):
one `predict()` call per CMA, median quantile (0.5) as the point
forecast. Used by both baselines/train.py (context-length search +
final forecast) and baselines/predict.py (artifact reload +
forecast regeneration), so the windowing and median-extraction logic
live here once.

Artifact stores config + per-CMA forecast lists, NOT the ~191MB
weights file: predict.py reloads weights via `from_pretrained`.
"""

import numpy as np
import torch

MODEL_NAME = "amazon/chronos-bolt-small"
MODEL_RUN = "primary-chronos"  # researcher verdict: Chronos primary; TimesFM only on failure

MEDIAN_Q_INDEX = 4  # quantiles are [0.1 .. 0.9]; index 4 == 0.5
LAST120 = 120
CONTEXT_GRID = ["full", "last120"]


def truncate_context(vals: np.ndarray, choice: str) -> np.ndarray:
    """Return the model context for a context-length choice."""
    vals = np.asarray(vals, dtype=float)
    if choice == "last120":
        return vals[-LAST120:]
    return vals


def series_by_cma(y_df, target: str) -> dict:
    """Per-CMA date-ordered raw series (finite values only)."""
    out = {}
    for cma, g in y_df.groupby("cma_canonical"):
        g = g.sort_values("date")
        vals = g[target].to_numpy(dtype=float)
        out[str(cma)] = vals[np.isfinite(vals)]
    return out


def pooled_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE = sqrt(mse)/n, same definition as evaluation.calculate_all_metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    mse = float(np.mean((y_true - y_pred) ** 2))
    return float(np.sqrt(mse) / n)


def load_pipeline(device: str):
    """Load Chronos-Bolt-Small on `device` (weights from HF hub cache)."""
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(MODEL_NAME, device_map=device)


def forecast_median(pipeline, context_1d: np.ndarray, horizon: int) -> np.ndarray:
    """Median-quantile point forecast for one univariate context."""
    ctx = torch.tensor(
        np.asarray(context_1d, dtype=float).reshape(1, -1), dtype=torch.float32
    )
    out = pipeline.predict(ctx, prediction_length=int(horizon))
    arr = out.detach().cpu().numpy() if torch.is_tensor(out) else np.asarray(out)
    return np.asarray(arr[0, MEDIAN_Q_INDEX, :], dtype=float).reshape(-1)


def split_fit_val(vals: np.ndarray, frac: float = 0.2):
    """Chronological fit/val split (last `frac` per CMA -> val)."""
    n = len(vals)
    n_val = max(1, int(n * frac))
    return vals[: n - n_val], vals[n - n_val :]


def score_context_choice(pipeline, train_series: dict, choice: str):
    """Pooled validation NRMSE for one context choice (multi-step, one call/CMA).

    Falls back to persistence for a CMA when its fit context is too
    short (<3 points) to feed the model.
    """
    all_true, all_pred = [], []
    for vals in train_series.values():
        fit, val = split_fit_val(vals)
        ctx = truncate_context(fit, choice)
        if len(ctx) >= 3 and len(val) > 0:
            fc = forecast_median(pipeline, ctx, len(val))
        else:  # degenerate: persistence fallback (finite, leakage-free)
            fc = np.array(
                [fit[-1] if len(fit) else (val[0] if len(val) else np.nan)]
                * len(val),
                dtype=float,
            )
        all_true.append(np.asarray(val, dtype=float))
        all_pred.append(np.asarray(fc, dtype=float))
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=float)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=float)
    return pooled_nrmse(y_true, y_pred)


def forecast_test_per_cma(pipeline, train_series: dict, test_series: dict, choice: str) -> dict:
    """Final test forecast: one call per CMA, horizon = that CMA's test length."""
    out = {}
    for cma, vals in train_series.items():
        horizon = len(test_series.get(cma, np.array([], dtype=float)))
        if horizon == 0:
            out[cma] = np.array([], dtype=float)
            continue
        ctx = truncate_context(vals, choice)
        if len(ctx) >= 3:
            out[cma] = forecast_median(pipeline, ctx, horizon)
        else:  # degenerate: persistence fallback
            last = float(vals[-1]) if len(vals) else float("nan")
            out[cma] = np.full(horizon, last, dtype=float)
    return out


def backtest_train_per_cma(pipeline, train_series: dict, choice: str) -> dict:
    """Leakage-free train predictions via rolling-origin multi-step backtest.

    Per CMA, the train series is split into consecutive chunks of size
    n_val (same length as that CMA's validation tail); chunk 0 (no
    prior history) uses persistence, each later chunk is forecast from
    all prior observations truncated to the winning context choice.
    """
    out = {}
    for cma, vals in train_series.items():
        n = len(vals)
        n_val = max(1, int(n * 0.2))
        preds = np.full(n, np.nan, dtype=float)
        # Chunk 0: persistence fallback (mirrors _lstm_rnn first-row fallback).
        first = min(n_val, n)
        for j in range(first):
            preds[j] = vals[j - 1] if j > 0 else vals[0]
        # Later chunks: true multi-step Chronos forecasts from prior data.
        start = first
        while start < n:
            stop = min(start + n_val, n)
            ctx = truncate_context(vals[:start], choice)
            if len(ctx) >= 3:
                preds[start:stop] = forecast_median(pipeline, ctx, stop - start)
            else:
                for j in range(start, stop):
                    preds[j] = vals[j - 1]
            start = stop
        out[cma] = preds
    return out
