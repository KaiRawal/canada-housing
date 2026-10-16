"""Train single pooled SARIMAX-AR on the FULL 24-CMA panel (no ensembles).

Pinned config (val-selected, train-only; test scored once):
  features = [rollmean_3, rollmean_12] of chained deltas (p=2)
  estimator = Lasso(alpha=0.001, max_iter=50000), standardized, fit_intercept=True
  pooling = full-24, equal-row weighting (no sample_weight)

Conventions (frozen from docs/delta_search.md):
  boundary-chained deltas/lags on combined train+test timeline per CMA;
  test masks n16=1728 / n24=2398; naive floors recomputed chained.

Outputs:
  artifacts/sarimax_full.joblib
  artifacts/sarimax_full_metrics.json
  prediction/sarimax_full_predictions_{train,test}.csv (eval schema)

Search provenance: docs/sarimax_full_search.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "prediction"
ART_DIR = ROOT / "artifacts"

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

FEATURES = ["rollmean_3", "rollmean_12"]
ALPHA = 0.001
LAGS_REQUIRED = list(range(1, 13))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def build_combo() -> pd.DataFrame:
    ytr = pd.read_csv(PRED_DIR / "y_train_full_total.csv", parse_dates=["date"])
    yte = pd.read_csv(PRED_DIR / "y_test_full_total.csv", parse_dates=["date"])
    combo = pd.concat(
        [ytr.assign(_split="train"), yte.assign(_split="test")], ignore_index=True
    )
    combo = combo.sort_values(["cma_canonical", "date"], kind="mergesort").reset_index(
        drop=True
    )
    combo["delta"] = combo.groupby("cma_canonical")["total"].diff()
    for k in range(1, 25):
        combo[f"dlag_{k}"] = combo.groupby("cma_canonical")["delta"].shift(k)
    for k in (3, 6, 12):
        cols = [f"dlag_{i}" for i in range(1, k + 1)]
        combo[f"rollmean_{k}"] = combo[cols].mean(axis=1)
        combo[f"rollstd_{k}"] = combo[cols].std(axis=1, ddof=0)
    return combo


def main() -> None:
    combo = build_combo()
    lag_cols = [f"dlag_{k}" for k in LAGS_REQUIRED]

    test = combo[
        (combo._split == "test") & combo[lag_cols].notna().all(axis=1)
    ].copy()
    test16 = test[test.cma_canonical.isin(SUBSET16)].copy()
    assert len(test) == 2398, f"test24 mask broke: {len(test)}"
    assert len(test16) == 1728, f"test16 mask broke: {len(test16)}"

    naive16 = rmse(test16["delta"], np.zeros(len(test16)))
    naive24 = rmse(test["delta"], np.zeros(len(test)))
    print(f"naive16={naive16:.6f} naive24={naive24:.6f}")

    fit_mask = (
        (combo._split == "train")
        & combo["delta"].notna()
        & combo[FEATURES].notna().all(axis=1)
    )
    train = combo[fit_mask].copy()
    print(f"full-24 fit rows: {len(train)}")
    per_cma_fit = train.groupby("cma_canonical").size().sort_values()
    print(per_cma_fit.to_string())
    assert len(train) == 7744, f"fit-row count drift: {len(train)}"

    Xf = train[FEATURES].to_numpy(float)
    yf = train["delta"].to_numpy(float)
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd == 0] = 1.0
    model = Lasso(alpha=ALPHA, max_iter=50000).fit((Xf - mu) / sd, yf)
    nnz = int((np.abs(model.coef_) > 1e-12).sum())
    print(f"fitted: nnz={nnz}/{len(FEATURES)} intercept={model.intercept_:.4f}")

    def predict(df: pd.DataFrame) -> np.ndarray:
        return model.predict((df[FEATURES].to_numpy(float) - mu) / sd)

    pred16 = predict(test16)
    pred24 = predict(test)
    r16 = rmse(test16["delta"], pred16)
    r24 = rmse(test["delta"], pred24)
    print(f"test16 RMSE={r16:.6f} (gain {(naive16 - r16) / naive16 * 100:.2f}%)")
    print(f"test24 RMSE={r24:.6f} (gain {(naive24 - r24) / naive24 * 100:.2f}%)")

    # Poison check: same config fit on 16-only must not beat full-fit on 16 by >1%.
    tr16 = train[train.cma_canonical.isin(SUBSET16)].copy()
    Xc, yc = tr16[FEATURES].to_numpy(float), tr16["delta"].to_numpy(float)
    muc, sdc = Xc.mean(0), Xc.std(0)
    sdc[sdc == 0] = 1.0
    ctrl = Lasso(alpha=ALPHA, max_iter=50000).fit((Xc - muc) / sdc, yc)
    r16_ctrl = rmse(
        test16["delta"], ctrl.predict((test16[FEATURES].to_numpy(float) - muc) / sdc)
    )
    print(f"control 16-only fit -> test16 RMSE={r16_ctrl:.6f} (fit rows {len(tr16)})")
    assert r16 <= 1.01 * r16_ctrl, (
        f"poison: full-fit {r16:.4f} > 1% worse than 16-only {r16_ctrl:.4f}"
    )
    assert r24 < naive24, "gate: must beat naive on full panel"

    # Per-CMA test table (full panel).
    test = test.copy()
    test["pred"] = pred24
    per_cma = {}
    for cma, g in test.groupby("cma_canonical"):
        per_cma[cma] = {
            "rmse": rmse(g["delta"], g["pred"]),
            "naive": rmse(g["delta"], np.zeros(len(g))),
            "n": len(g),
        }

    # Level predictions in eval-schema (total_pred = y_prev + delta_hat).
    combo_s = combo.sort_values(["cma_canonical", "date"], kind="mergesort").copy()
    combo_s["y_prev"] = combo_s.groupby("cma_canonical")["total"].shift(1)
    tkey = list(
        zip(
            test["cma_canonical"],
            pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"),
        )
    )
    pred_d = dict(zip(tkey, map(float, pred24)))
    tr_sub = train.reset_index(drop=True)
    tr_pred = model.predict((tr_sub[FEATURES].to_numpy(float) - mu) / sd)
    trkey = list(
        zip(
            tr_sub["cma_canonical"],
            pd.to_datetime(tr_sub["date"]).dt.strftime("%Y-%m-%d"),
        )
    )
    tr_pred_d = dict(zip(trkey, map(float, tr_pred)))

    def frame_for(split: str, med: dict) -> pd.DataFrame:
        g = combo_s[combo_s._split == split].copy()
        kk = list(
            zip(g["cma_canonical"], pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d"))
        )
        dv = np.array([med.get(k, np.nan) for k in kk])
        yp = g["y_prev"].to_numpy(float)
        ok = np.isfinite(dv) & np.isfinite(yp)
        return pd.DataFrame(
            {
                "cma_canonical": g["cma_canonical"].values,
                "date": pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d"),
                "total_true": g["total"].to_numpy(float),
                "total_pred": np.where(ok, yp + dv, np.nan),
            }
        ).sort_values(["cma_canonical", "date"]).reset_index(drop=True)

    fr_test = frame_for("test", pred_d)
    fr_train = frame_for("train", tr_pred_d)
    fr_test.to_csv(PRED_DIR / "sarimax_full_predictions_test.csv", index=False)
    fr_train.to_csv(PRED_DIR / "sarimax_full_predictions_train.csv", index=False)
    print(
        f"wrote predictions test={len(fr_test)} "
        f"({int(fr_test['total_pred'].notna().sum())} valid) "
        f"train={len(fr_train)} ({int(fr_train['total_pred'].notna().sum())} valid)"
    )

    ART_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler_mu": mu,
            "scaler_sd": sd,
            "features": FEATURES,
            "alpha": ALPHA,
            "estimator": "Lasso",
            "pooling": "full-24 pooled, equal-row",
            "predictions_test": pred_d,
            "predictions_train": tr_pred_d,
            "mask": {"convention": "boundary-chained", "n16": 1728, "n24": 2398},
            "fit_rows": len(train),
            "per_cma_fit_rows": {k: int(v) for k, v in per_cma_fit.items()},
        },
        ART_DIR / "sarimax_full.joblib",
    )
    metrics = {
        "rmse_24": r24,
        "rmse_16": r16,
        "naive_24": naive24,
        "naive_16": naive16,
        "gain_pct_24": (naive24 - r24) / naive24 * 100,
        "gain_pct_16": (naive16 - r16) / naive16 * 100,
        "control_16only_rmse_16": r16_ctrl,
        "config": {
            "features": FEATURES,
            "alpha": ALPHA,
            "nnz": nnz,
            "intercept": float(model.intercept_),
            "coef": [float(c) for c in model.coef_],
        },
        "per_cma": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_cma.items()},
    }
    (ART_DIR / "sarimax_full_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved artifacts/sarimax_full.joblib + artifacts/sarimax_full_metrics.json")

    art = joblib.load(ART_DIR / "sarimax_full.joblib")
    chk = art["model"].predict(
        (train[FEATURES].to_numpy(float)[:32] - art["scaler_mu"]) / art["scaler_sd"]
    )
    assert np.array_equal(
        np.asarray(chk), model.predict((Xf[:32] - mu) / sd)
    ), "reload drift"
    assert abs(art["predictions_test"][tkey[0]] - float(pred24[0])) == 0.0
    print("artifact reload check: bit-exact")


if __name__ == "__main__":
    main()
