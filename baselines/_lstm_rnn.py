"""Shared helpers for the lstm_rnn baseline (variant V5).

Global (multi-CMA) one-step-ahead LSTM over lagged `total` windows with
per-CMA standardization. Used by both baselines/train.py (training +
hyperparameter search) and baselines/predict.py (artifact reload +
forecasting), so the architecture and windowing logic live here once.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

SEED = 42


class LSTMRegressor(nn.Module):
    """LSTM -> single fully-connected output (batch_first).

    ``input_size`` is 1 for the run-1 univariate model and n_features
    for the try-1 multivariate delta model (same class, wider input).
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 1,
                 input_size: int = 1):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=int(input_size), hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def select_device() -> str:
    """Return 'mps' if a full LSTM train step works there, else 'cpu'."""
    try:
        if torch.backends.mps.is_available():
            m = LSTMRegressor(hidden_size=8, num_layers=1).to("mps")
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            x = torch.randn(8, 6, 1, device="mps")
            y = torch.randn(8, 1, device="mps")
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            torch.mps.synchronize()
            return "mps"
    except Exception:
        pass
    return "cpu"


def fit_scalers(series_by_cma: dict) -> dict:
    """Per-CMA mean/std over the given raw series (std floored at 1e-8)."""
    scalers = {}
    for cma, vals in series_by_cma.items():
        vals = np.asarray(vals, dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if not np.isfinite(std) or std < 1e-8:
            std = 1e-8
        scalers[cma] = {"mean": mean, "std": std}
    return scalers


def make_windows(vals_std: np.ndarray, lookback: int):
    """Slide windows over a standardized series.

    Returns (X[N, L, 1], target_pos[N]) where target_pos[t] is the index
    (into vals_std) of the target row for window t.
    """
    n = len(vals_std)
    if n <= lookback:
        return (
            np.zeros((0, lookback, 1), dtype=np.float32),
            np.zeros((0,), dtype=int),
        )
    n_win = n - lookback
    X = np.zeros((n_win, lookback, 1), dtype=np.float32)
    pos = np.zeros((n_win,), dtype=int)
    for t in range(n_win):
        X[t, :, 0] = vals_std[t:t + lookback]
        pos[t] = t + lookback
    return X, pos


def pooled_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE = sqrt(mse)/n, same definition as evaluation.calculate_all_metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    mse = float(np.mean((y_true - y_pred) ** 2))
    return float(np.sqrt(mse) / n)


def train_config(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val_raw: np.ndarray,
    val_mean: dict,
    val_std: dict,
    val_rows: list,
    hidden_size: int,
    num_layers: int,
    lr: float,
    device: str,
    max_epochs: int = 60,
    patience: int = 10,
    batch_size: int = 256,
) -> tuple:
    """Train one LSTM config with early stopping on pooled val NRMSE.

    Returns (best_val_nrmse, best_epoch_1indexed, best_state_dict_on_cpu).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    model = LSTMRegressor(hidden_size=hidden_size, num_layers=num_layers)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(X_fit, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_fit, dtype=np.float32).reshape(-1, 1)),
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen
    )
    X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)

    best_nrmse = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        # Pooled validation NRMSE on the raw scale.
        model.eval()
        with torch.no_grad():
            pv = model(X_val_t).cpu().numpy().reshape(-1)
        pred_raw = np.array(
            [pv[i] * val_std[c] + val_mean[c] for i, c in enumerate(val_rows)],
            dtype=float,
        )
        nrmse = pooled_nrmse(y_val_raw, pred_raw)
        if np.isfinite(nrmse) and nrmse < best_nrmse - 1e-12:
            best_nrmse = float(nrmse)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is None:  # never improved (should not happen); keep last
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max_epochs
    return best_nrmse, best_epoch, best_state


def onestep_predict_frame(
    y_df,
    target: str,
    scalers: dict,
    model: LSTMRegressor,
    lookback: int,
    device: str,
):
    """One-step-ahead predictions for every row of y_df (finite everywhere).

    Uses true lagged values within y_df; the first `lookback` rows per CMA
    (no full window) fall back to persistence (previous value / first value).
    Rows with NaN target get NaN predictions (matching eval masking).
    """
    import pandas as pd

    preds = pd.Series(np.nan, index=y_df.index, dtype=float)
    model.eval()
    with torch.no_grad():
        for cma, g in y_df.groupby("cma_canonical"):
            g = g.sort_values("date")
            raw = g[target].to_numpy(dtype=float)
            idx = g.index.to_numpy()
            sc = scalers.get(str(cma))
            if sc is None:  # unseen CMA -> persistence fallback
                for j in range(len(raw)):
                    if np.isfinite(raw[j]):
                        preds.loc[idx[j]] = raw[j - 1] if j > 0 and np.isfinite(raw[j - 1]) else raw[j]
                continue
            mean, std = float(sc["mean"]), float(sc["std"])
            std = std if std >= 1e-8 else 1e-8
            stdzd = (raw - mean) / std
            # Rows with a full finite window -> model prediction.
            targets, positions = [], []
            for j in range(lookback, len(raw)):
                w = stdzd[j - lookback:j]
                if np.isfinite(w).all() and np.isfinite(raw[j]):
                    targets.append(w.reshape(1, lookback, 1).astype(np.float32))
                    positions.append(j)
            if targets:
                X = torch.from_numpy(np.concatenate(targets, axis=0)).to(device)
                pv = model(X).cpu().numpy().reshape(-1)
                for k, j in enumerate(positions):
                    preds.loc[idx[j]] = float(pv[k] * std + mean)
            # Fallback for rows without a model prediction but finite target.
            for j in range(len(raw)):
                if np.isnan(preds.loc[idx[j]]) and np.isfinite(raw[j]):
                    if j == 0:
                        preds.loc[idx[j]] = float(raw[j])
                    elif np.isfinite(raw[j - 1]):
                        preds.loc[idx[j]] = float(raw[j - 1])
                    else:
                        preds.loc[idx[j]] = float(raw[j])
    return preds


def build_model_from_state(state_dict: dict, hidden_size: int, num_layers: int,
                          device: str, input_size: int = 1):
    torch.manual_seed(SEED)
    model = LSTMRegressor(hidden_size=hidden_size, num_layers=num_layers,
                          input_size=input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def copy_state_cpu(model: LSTMRegressor) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Try-1 multivariate delta helpers (additive; run-1 univariate path above is
# untouched). Windows are rows [t-L+1 .. t] of the standardized deduped
# feature frame (contemporaneous exo + causal lags, same columns as the
# linear TRY-3 110-col set); the target is the standardized delta
# delta_t = total_t - lag_1_t, nested back to levels as lag_1 + delta.
# All scalers are fit on the fit split only (strict out-of-sample val).
# ---------------------------------------------------------------------------

def fit_feature_scalers(F_fit, cols) -> dict:
    """Per-feature global mean/std over fit rows (std floored at 1e-8)."""
    import pandas as pd

    scalers = {}
    arr = F_fit[cols] if isinstance(F_fit, pd.DataFrame) else F_fit
    for j, c in enumerate(cols):
        v = np.asarray(arr[c] if isinstance(arr, pd.DataFrame) else arr[:, j],
                       dtype=float)
        mean = float(np.mean(v))
        std = float(np.std(v))
        if not np.isfinite(std) or std < 1e-8:
            std = 1e-8
        scalers[str(c)] = {"mean": mean, "std": std}
    return scalers


def standardize_frame(F, cols, scalers) -> np.ndarray:
    """Standardize a feature frame to (T, D) float32 in column order."""
    import pandas as pd

    if isinstance(F, pd.DataFrame):
        M = F[cols].to_numpy(dtype=float)
    else:
        M = np.asarray(F, dtype=float)
    Z = np.empty_like(M, dtype=float)
    for j, c in enumerate(cols):
        sc = scalers[str(c)]
        Z[:, j] = (M[:, j] - float(sc["mean"])) / float(sc["std"])
    return Z.astype(np.float32)


def make_windows_multi(Z: np.ndarray, lookback: int):
    """Slide windows over a (T, D) standardized feature matrix.

    Returns (X[N, L, D], pos[N]) where window i covers rows
    [i .. i+L-1] and pos[i] = i+L-1 is the target row index.
    """
    Z = np.asarray(Z, dtype=np.float32)
    T, D = Z.shape
    L = int(lookback)
    if T < L:
        return (np.zeros((0, L, D), dtype=np.float32),
                np.zeros((0,), dtype=int))
    N = T - L + 1
    X = np.zeros((N, L, D), dtype=np.float32)
    pos = np.zeros((N,), dtype=int)
    for i in range(N):
        X[i] = Z[i:i + L]
        pos[i] = i + L - 1
    return X, pos


def _delta_to_level(d_std: np.ndarray, cma_list: list, delta_scalers: dict,
                    lag1_raw: np.ndarray, calib_a: float = 1.0,
                    calib_b: float = 0.0) -> np.ndarray:
    """Unstandardize delta preds and nest to levels: lag_1 + (a*d + b)."""
    d_std = np.asarray(d_std, dtype=float).reshape(-1)
    out = np.empty_like(d_std, dtype=float)
    for i, c in enumerate(cma_list):
        sc = delta_scalers[str(c)]
        d_raw = (float(d_std[i]) * float(sc["std"]) + float(sc["mean"]))
        out[i] = float(lag1_raw[i]) + float(calib_a) * d_raw + float(calib_b)
    return out


def train_delta_config(
    X_fit: np.ndarray,
    yd_fit: np.ndarray,
    X_val: np.ndarray,
    val_lag1_raw: np.ndarray,
    val_y_raw: np.ndarray,
    val_cmas: list,
    delta_scalers: dict,
    hidden_size: int,
    num_layers: int,
    input_size: int,
    lr: float,
    device: str,
    max_epochs: int = 60,
    patience: int = 10,
    batch_size: int = 256,
) -> tuple:
    """Train one multivariate delta-LSTM config.

    MSE on standardized deltas; early stopping on pooled LEVEL val NRMSE
    (lag_1 + unstandardized delta, repo definition). Returns
    (best_val_nrmse_level, best_epoch_1indexed, best_state_dict_on_cpu).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    model = LSTMRegressor(hidden_size=hidden_size, num_layers=num_layers,
                          input_size=input_size)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(X_fit, dtype=np.float32)),
        torch.from_numpy(np.asarray(yd_fit, dtype=np.float32).reshape(-1, 1)),
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen
    )
    X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)

    best_nrmse = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(X_val_t).cpu().numpy().reshape(-1)
        pred_level = _delta_to_level(pv, val_cmas, delta_scalers,
                                     val_lag1_raw)
        nrmse = pooled_nrmse(val_y_raw, pred_level)
        if np.isfinite(nrmse) and nrmse < best_nrmse - 1e-12:
            best_nrmse = float(nrmse)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is None:  # never improved (should not happen); keep last
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        best_epoch = max_epochs
    return best_nrmse, best_epoch, best_state


def delta_level_predict_windows(model, X: np.ndarray, lag1_raw: np.ndarray,
                                cma_list: list, delta_scalers: dict,
                                device: str, calib_a: float = 1.0,
                                calib_b: float = 0.0) -> np.ndarray:
    """Level preds for stacked windows: lag_1 + (a*delta_raw + b)."""
    import torch

    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        d_std = model(Xt).cpu().numpy().reshape(-1)
    return _delta_to_level(d_std, cma_list, delta_scalers, lag1_raw,
                           calib_a=calib_a, calib_b=calib_b)
