"""Nesting foundation: causal lag features + persistence blending.

Every later try imports these 5 helpers; no existing trainer depends on
them (importable with zero changes to the 6 trainers).

Causality contract: every feature for row t uses only target values
strictly before t in the same CMA (groupby+shift, never future values).
"""

import numpy as np
import pandas as pd

from prediction.prediction import calculate_persistence_predictions


def _infer_target(y: pd.DataFrame) -> str:
    """Return the value column (``total`` or the single non-id column)."""
    if "total" in y.columns:
        return "total"
    cands = [c for c in y.columns if c not in ("cma_canonical", "date")]
    if len(cands) == 1:
        return cands[0]
    if cands:
        return cands[0]
    raise ValueError(f"Cannot infer target column from {list(y.columns)}")


def build_lag_features(y, lags=(1, 2, 3, 6, 12), windows=(3, 6, 12)):
    """Build causal per-CMA lag/rolling features, index-aligned to ``y``.

    Each feature for row t uses only target values strictly before t in
    the same CMA (groupby+shift). Columns: ``lag_{k}`` for k in lags,
    ``rollmean_{w}`` / ``rollstd_{w}`` for w in windows, where the
    rolling stats are over the w values strictly before t
    (shift(1).rolling(w), min_periods=w so warmup is NaN).
    """
    target = _infer_target(y)
    if "cma_canonical" not in y.columns or "date" not in y.columns:
        raise ValueError("y must have columns cma_canonical, date, <target>")
    lags = tuple(int(k) for k in lags)
    windows = tuple(int(w) for w in windows)

    cmas = y["cma_canonical"].to_numpy()
    dates = pd.to_datetime(y["date"]).to_numpy()
    vals = y[target].to_numpy(dtype=float)
    work = pd.DataFrame(
        {"_cma": cmas, "_date": dates, "_val": vals}, index=y.index
    ).sort_values(["_cma", "_date"], kind="mergesort")

    feats = pd.DataFrame(index=work.index)
    grouped = work.groupby("_cma")["_val"]
    for k in lags:
        feats[f"lag_{k}"] = grouped.shift(k).astype(float)
    for w in windows:
        # shift(1) first -> strictly before t; min_periods=w -> full-window warmup NaN.
        feats[f"rollmean_{w}"] = work.groupby("_cma")["_val"].transform(
            lambda s, w=w: s.shift(1).rolling(window=w, min_periods=w).mean()
        ).astype(float)
        feats[f"rollstd_{w}"] = work.groupby("_cma")["_val"].transform(
            lambda s, w=w: s.shift(1).rolling(window=w, min_periods=w).std()
        ).astype(float)
    # Back to the caller's original row order/index.
    return feats.reindex(y.index)


def assemble(split_frames):
    """Concatenate train+test y, build lags ONCE, split back.

    The full per-CMA timeline is assembled so test warmup rows see the
    train-tail history. Features are strictly causal (shift-based), so no
    row ever sees its own or any future target: train rows only see past
    train values, test rows only see train-tail + earlier test values.
    Returns {split: feature DataFrame} index-aligned to each input frame.
    """
    keys = list(split_frames.keys())
    if not keys:
        return {}
    combined = pd.concat(
        [split_frames[k] for k in keys], keys=keys, names=["_split"]
    )
    feats_all = build_lag_features(combined)
    out = {}
    for k in keys:
        mask = combined.index.get_level_values("_split") == k
        sub = feats_all.loc[mask].copy()
        sub.index = sub.index.droplevel(0)
        out[k] = sub.reindex(split_frames[k].index)
    return out


def persistence_pred(y):
    """One-step persistence: y_hat_t = y_{t-1} per CMA, NaN first row."""
    target = _infer_target(y)
    return calculate_persistence_predictions(y, target)


def blend(model_pred, persist_pred, w):
    """Blend w*model + (1-w)*persist, propagating the persist NaN mask.

    Wherever persist is NaN the output is NaN regardless of w; w=0
    reproduces persistence bit-for-bit (returns a copy of persist_pred).
    """
    persist_s = (
        persist_pred.copy()
        if isinstance(persist_pred, pd.Series)
        else pd.Series(persist_pred)
    )
    wf = float(w)
    if wf == 0.0:
        return persist_s.copy()
    if isinstance(model_pred, pd.Series):
        if model_pred.index.equals(persist_s.index):
            model_s = model_pred
        elif len(model_pred) == len(persist_s):
            model_s = pd.Series(
                model_pred.to_numpy(dtype=float), index=persist_s.index
            )
        else:
            model_s = model_pred.reindex(persist_s.index)
    else:
        model_s = pd.Series(np.asarray(model_pred, dtype=float), index=persist_s.index)
    out = wf * model_s.astype(float) + (1.0 - wf) * persist_s.astype(float)
    out = pd.Series(out.to_numpy(dtype=float), index=persist_s.index)
    out[persist_s.isna().to_numpy()] = np.nan
    return out


def tune_blend(val_true, val_model, val_persist, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Grid-search the blend weight by repo NRMSE (sqrt(mse)/n, valid rows).

    Returns (best_w, best_nrmse); ties keep the first grid entry.
    """
    grid = tuple(float(g) for g in grid)
    true_a = np.asarray(val_true, dtype=float).ravel()
    model_a = np.asarray(val_model, dtype=float).ravel()
    persist_a = np.asarray(val_persist, dtype=float).ravel()
    best_w = grid[0]
    best_nrmse = float("inf")
    for g in grid:
        persist_s = pd.Series(persist_a)
        model_s = pd.Series(model_a)
        blended = blend(model_s, persist_s, g).to_numpy(dtype=float)
        valid = ~(np.isnan(true_a) | np.isnan(blended))
        n = int(valid.sum())
        if n == 0:
            nrmse = float("inf")
        else:
            diff = true_a[valid] - blended[valid]
            mse = float(np.mean(diff * diff))
            nrmse = float(np.sqrt(mse) / n)
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_w = g
    return float(best_w), float(best_nrmse)
