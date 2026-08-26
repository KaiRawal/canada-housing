"""
baselines_suite.py — T2.1: establish which SIMPLE baseline is the real bar.

Three baselines beyond persistence, all run through harness.evaluate() on the
identical 5 expanding folds, each logged as its own runs.jsonl row:

  1. T2.1-drift       Random walk with drift: per CMA, drift mu_c = mean
                      first difference of the LEVEL over the fold's train
                      window; forecast y(t) = y_last + h * mu_c where h =
                      months since the CMA's last train observation.
                      Level formulation: the harness scores the `total` index
                      LEVEL directly, so every baseline here forecasts levels;
                      the Δlog-vs-level question is studied properly in T2.2.
  2. T2.1-arima       Per-CMA ARIMA(order=(1,1,0)) fit on the train window
                      only (statsmodels; fixed small order a priori to keep
                      compute bounded — no auto-arima search). h-step forecasts
                      via res.forecast(steps=h_max) mapped onto val dates by
                      month arithmetic.
  3. T2.1-ridge-lags  ONE POOLED Ridge(alpha=1.0) over ALL CMAs on the existing
                      lag-1 features from X_train_full_total.csv plus CMA dummy
                      columns; StandardScaler + Ridge inside an sklearn
                      Pipeline fitted on the TRAIN WINDOW ONLY (no leakage).
                      Pooled-with-dummies rather than per-CMA because early
                      folds give each CMA only ~270 monthly rows against 176
                      features — per-CMA ridge would be hopelessly
                      underdetermined; pooling shares strength (whether spatial
                      pooling genuinely helps is probed separately in T2.3).
                      Late-starter CMAs unseen at train time are handled by
                      reindexing their dummy block to the train columns
                      (fill 0), so they are predicted from numeric lags alone.

Persistence is reported alongside EVERY model by the harness itself
(`_score_fold` computes it on the identical folds and identical row sets).
No test-split files are read anywhere in this script.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parent.parent
for p in (str(EXPERIMENTS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness  # noqa: E402
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    ID_COLS,
    TARGET,
    build_folds,
    current_git_sha,
    describe_folds,
    evaluate,
    load_train_data,
    log_run,
    save_results_json,
)

OUT_DIR = ARTIFACTS_DIR / "T2.1"
PERSISTENCE_REF_JSON = ARTIFACTS_DIR / "T1.2" / "baseline_validation_results.json"

ARIMA_ORDER = (1, 1, 0)          # fixed small order, decided a priori
RIDGE_ALPHA = 1.0                # default; tuning is T5's job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_number(s: pd.Series) -> np.ndarray:
    """Continuous month counter (year*12 + month) for month arithmetic."""
    d = pd.to_datetime(s)
    return (d.dt.year * 12 + d.dt.month).to_numpy()


# ---------------------------------------------------------------------------
# Model factories (harness model_fn contract)
# ---------------------------------------------------------------------------

def drift_model_fn(train_features: pd.DataFrame, train_target: pd.Series):
    """Random walk with drift, estimated per CMA on the train window."""
    df = pd.concat(
        [train_features[ID_COLS], train_target.rename("_y")], axis=1
    ).sort_values(ID_COLS)
    last_val = df.groupby("cma_canonical")["_y"].last()
    last_month = df.groupby("cma_canonical")["date"].agg(
        lambda s: int(s.iloc[-1].year * 12 + s.iloc[-1].month)
    )
    diffs = df.assign(_d=df.groupby("cma_canonical")["_y"].diff())
    drift = diffs.groupby("cma_canonical")["_d"].mean()

    def predict_fn(val_features: pd.DataFrame) -> pd.Series:
        vf = val_features[ID_COLS]
        cma = vf["cma_canonical"]
        y_last = cma.map(last_val)
        m_last = cma.map(last_month)
        mu = cma.map(drift)
        h = _month_number(vf["date"]) - m_last.to_numpy()
        return pd.Series(y_last.to_numpy() + h * mu.to_numpy(),
                         index=val_features.index)

    return predict_fn


def arima_model_fn(train_features: pd.DataFrame, train_target: pd.Series):
    """Per-CMA ARIMA(ARIMA_ORDER) fitted on the train window only."""
    from statsmodels.tsa.arima.model import ARIMA

    df = pd.concat(
        [train_features[ID_COLS], train_target.rename("_y")], axis=1
    ).sort_values(ID_COLS)
    fits = {}
    for cma, g in df.groupby("cma_canonical", sort=False):
        y = g["_y"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short series
            res = ARIMA(y, order=ARIMA_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        d = g["date"].iloc[-1]
        fits[cma] = (int(d.year * 12 + d.month), res)

    def predict_fn(val_features: pd.DataFrame) -> pd.Series:
        vf = val_features[ID_COLS]
        out = pd.Series(np.nan, index=val_features.index, dtype=float)
        v_month = pd.Series(_month_number(vf["date"]), index=vf.index)
        for cma, g in vf.groupby("cma_canonical", sort=False):
            if cma not in fits:
                continue  # no train history this fold -> left NaN, dropped by mask
            m_last, res = fits[cma]
            h = v_month.loc[g.index] - m_last
            assert (h >= 1).all(), "validation month not strictly after train"
            h_max = int(h.max())
            fc = np.asarray(res.forecast(steps=h_max), dtype=float)
            out.loc[g.index] = fc[h.to_numpy(dtype=int) - 1]
        return out

    return predict_fn


def _past_lag1(full_df: pd.DataFrame) -> pd.Series:
    """
    Actual y_{t-1} for every row, computed by a backward-looking within-CMA
    shift over the FULL train split. Uses past observations only (shift(1)),
    so reading it inside predict_fn is one-step-refreshed conditioning on the
    same information set persistence itself uses — no leakage.
    """
    return full_df.sort_values(ID_COLS).groupby("cma_canonical")[TARGET].shift(1)


def make_drift_recursive_model_fn(full_df: pd.DataFrame):
    """
    T2.1 critic fix #2: recursive-update drift. One-step-ahead forecast
    y_hat(t) = y_{t-1}(ACTUAL) + mu_c, i.e. truth-refreshed anchor instead of
    the frozen last-train-value + h*mu extrapolation. Same information set as
    persistence/ridge-on-lags (which both condition on lag-1 actuals), making
    cross-model ranking apples-to-apples.
    """
    lag1 = _past_lag1(full_df)

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        df = pd.concat(
            [train_features[ID_COLS], train_target.rename("_y")], axis=1
        ).sort_values(ID_COLS)
        diffs = df.assign(_d=df.groupby("cma_canonical")["_y"].diff())
        drift = diffs.groupby("cma_canonical")["_d"].mean()

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            prev = pd.to_numeric(lag1.loc[val_features.index],
                                 errors="coerce")
            mu = val_features["cma_canonical"].map(drift)
            return pd.Series(prev.to_numpy() + mu.to_numpy(),
                             index=val_features.index)

        return predict_fn

    return model_fn


def make_arima_recursive_model_fn(full_df: pd.DataFrame):
    """
    T2.1 critic fix #2: recursive-update ARIMA(1,1,0). Fit per CMA on the
    train window only, then forecast ONE step at a time across the validation
    window, appending each ACTUAL observation through t-1 before the next
    forecast (statsmodels .append(refit=False)). Same information set as
    persistence / ridge / recursive drift.
    """
    lag1 = _past_lag1(full_df)

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        df = pd.concat(
            [train_features[ID_COLS], train_target.rename("_y")], axis=1
        ).sort_values(ID_COLS)
        fits = {}
        for cma, g in df.groupby("cma_canonical", sort=False):
            y = g["_y"].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y, order=ARIMA_ORDER,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit()
            fits[cma] = res

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            out = pd.Series(np.nan, index=val_features.index, dtype=float)
            v = val_features[ID_COLS].copy()
            v["_lag1"] = lag1.loc[val_features.index]
            for cma, g in v.sort_values("date").groupby("cma_canonical",
                                                        sort=False):
                if cma not in fits:
                    continue  # no train history this fold -> NaN -> mask
                res = fits[cma]
                months = _month_number(g["date"])
                assert (np.diff(months) == 1).all(), \
                    f"non-contiguous monthly series for {cma}"
                for idx in g.index:
                    fc = float(np.asarray(res.forecast(steps=1),
                                          dtype=float)[0])
                    out[idx] = fc
                    res = res.append([float(v["_lag1"].loc[idx])],
                                     refit=False)
            return out

        return predict_fn

    return model_fn


def ridge_lags_model_fn(train_features: pd.DataFrame, train_target: pd.Series):
    """Pooled Ridge on lag features + CMA dummies, standardized train-only."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr = pd.get_dummies(
        train_features.drop(columns=["date"]), columns=["cma_canonical"]
    )
    if X_tr.isna().any().any() or not np.isfinite(X_tr.to_numpy(float)).all():
        raise ValueError("NaN/non-finite values in training features")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    pipe.fit(X_tr, train_target)

    def predict_fn(val_features: pd.DataFrame) -> pd.Series:
        X_va = pd.get_dummies(
            val_features.drop(columns=["date"]), columns=["cma_canonical"]
        ).reindex(columns=X_tr.columns, fill_value=0.0)
        preds = pipe.predict(X_va)
        return pd.Series(preds, index=val_features.index)

    return predict_fn


# ---------------------------------------------------------------------------
# Leaderboard assembly
# ---------------------------------------------------------------------------

def load_persistence_reference() -> dict:
    """Canonical persistence reference (Run B) from the corrected T1.2 artifact."""
    ref = json.loads(PERSISTENCE_REF_JSON.read_text())
    return ref["run_B_persistence_reference"]


def build_leaderboard(runs: dict[str, dict]) -> pd.DataFrame:
    """Rows: predictors incl. persistence; cols: metric mean±std + per-fold."""
    rows = []
    base_ref = load_persistence_reference()
    rows.append({
        "predictor": "persistence (canonical)",
        **{m: (base_ref["agg"]["baseline"][m]["mean"],
               base_ref["agg"]["baseline"][m]["std"])
           for m in harness.METRIC_NAMES},
        **{f"{m}_fold{pf['fold_id']}": pf["model"][m]
           for pf in base_ref["per_fold"] for m in harness.METRIC_NAMES},
        "fold5_eval_rows": base_ref["per_fold"][-1]["n_eval_rows"],
        "source": "T1.2 (mdafix row)",
    })
    for run_id, res in runs.items():
        agg = res["agg"]["model"]
        row = {"predictor": run_id,
               **{m: (agg[m]["mean"], agg[m]["std"]) for m in harness.METRIC_NAMES},
               **{f"{m}_fold{pf['fold_id']}": pf["model"][m]
                  for pf in res["per_fold"] for m in harness.METRIC_NAMES},
               "fold5_eval_rows": res["per_fold"][-1]["n_eval_rows"],
               "source": f"runs.jsonl:{run_id}"}
        rows.append(row)
        # paired persistence-on-the-same-mask block for THIS run
        b_agg = res["agg"]["baseline"]
        rows.append({"predictor": f"{run_id} [persistence, same mask]",
                     **{m: (b_agg[m]["mean"], b_agg[m]["std"])
                        for m in harness.METRIC_NAMES},
                     **{f"{m}_fold{pf['fold_id']}": pf["baseline"][m]
                        for pf in res["per_fold"] for m in harness.METRIC_NAMES},
                     "fold5_eval_rows": res["per_fold"][-1]["n_eval_rows"],
                     "source": f"runs.jsonl:{run_id}:baseline"})
    return pd.DataFrame(rows)


def fmt(mean_std) -> str:
    m, s = mean_std
    return f"{m:.4f}±{s:.4f}" if abs(m) > 1e-3 else f"{m:.3e}±{s:.3e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recursive", action="store_true",
                        help="run ONLY the recursive-update drift/ARIMA "
                             "variants (T2.1 critic fix #2) and append their "
                             "registry rows; original rows stay untouched")
    args = parser.parse_args()

    print("=== T2.1 baseline suite under the harness ===")
    df = load_train_data(TARGET)
    print(f"train split: {len(df)} rows, {df['cma_canonical'].nunique()} CMAs, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")

    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("\nfold design:\n" + describe_folds(folds))

    specs = [] if args.recursive else [
        ("T2.1-drift",
         drift_model_fn,
         "Random walk with drift: per-CMA mean first difference of the level "
         "estimated on the train window; forecast = last train value + h*drift "
         "(level formulation).",
         {"model": "drift_rw", "target_form": "level",
          "drift_estimator": "mean first-difference per CMA, train window"}),
        ("T2.1-arima",
         arima_model_fn,
         f"Per-CMA statsmodels ARIMA{ARIMA_ORDER} fit on the train window only; "
         "fixed small order a priori (bounded compute); h-step forecast mapped "
         "to validation months.",
         {"model": f"arima{ARIMA_ORDER}", "target_form": "level",
          "package": "statsmodels.tsa.arima.model.ARIMA",
          "order_policy": "fixed a priori, no search"}),
        ("T2.1-ridge-lags",
         ridge_lags_model_fn,
         f"Pooled Ridge(alpha={RIDGE_ALPHA}) on all lag-1 features + CMA "
         "dummies; StandardScaler inside sklearn Pipeline fitted on train "
         "window only; late-starter CMAs handled via dummy reindex(fill=0).",
         {"model": "ridge_on_lags", "alpha": RIDGE_ALPHA,
          "pooling": "pooled across CMAs with CMA dummies",
          "features": "all 176 lag-1 columns from X_train_full_total.csv",
          "scaling": "StandardScaler fit on train window only"}),
    ]
    if args.recursive:
        # Critic fix #2: truth-refreshed one-step-ahead variants so drift and
        # ARIMA compete on the SAME information set as persistence/ridge.
        specs = [
            ("T2.1-drift-recursive",
             make_drift_recursive_model_fn(df),
             "Recursive-update drift: y_hat(t) = ACTUAL y_{t-1} + mu_c "
             "(truth-refreshed anchor), one-step-ahead — same information set "
             "as persistence / ridge-on-lags.",
             {"model": "drift_rw_recursive", "target_form": "level",
              "update_rule": "one-step-ahead, actual y_{t-1} anchor",
              "critic_fix": "T2.1-fix-2"}),
            ("T2.1-arima-recursive",
             make_arima_recursive_model_fn(df),
             f"Recursive-update ARIMA{ARIMA_ORDER}: per-CMA fit on train "
             "window only, then rolling ONE-step-ahead forecasts appending "
             "actuals through t-1 within the validation window.",
             {"model": f"arima{ARIMA_ORDER}_recursive", "target_form": "level",
              "package": "statsmodels.tsa.arima.model.ARIMA",
              "update_rule": "rolling one-step-ahead, .append(actuals)",
              "order_policy": "fixed a priori, no search",
              "critic_fix": "T2.1-fix-2"}),
        ]

    runs = {}
    for run_id, fn, desc, cfg in specs:
        print(f"\n--- running {run_id} ---")
        res = evaluate(fn, df=df, folds=folds, target=TARGET)
        runs[run_id] = res

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        log_run(
            run_id=run_id, task="T2", sub="T2.1", description=desc,
            config={"git_sha_pre_commit": current_git_sha(), **cfg},
            results=res,
        )

        agg = res["agg"]
        print(f"  model      : " +
              "  ".join(f"{m}={fmt((agg['model'][m]['mean'], agg['model'][m]['std']))}"
                       for m in harness.METRIC_NAMES))
        print(f"  persistence: " +
              "  ".join(f"{m}={fmt((agg['baseline'][m]['mean'], agg['baseline'][m]['std']))}"
                        for m in harness.METRIC_NAMES))

    # -- Leaderboard / plot / summary (full suite only) ------------------------
    if args.recursive:
        # Registry rows + per-run artifacts already saved above; the original
        # leaderboard/plot artifacts are left untouched (append-only policy).
        summary = {
            "run_id": "T2.1-recursive-variants",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha_pre_commit": current_git_sha(),
            "runs": {rid: {"agg_model": res["agg"]["model"],
                           "per_fold": res["per_fold"]}
                     for rid, res in runs.items()},
            "verdict_hint": "see analyses/T2.1.md (recursive variants section)",
        }
        save_results_json(summary, OUT_DIR / "recursive_variants_summary.json")
        print("\n=== T2.1 recursive variants complete ===")
        return

    board = build_leaderboard(runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board_out = OUT_DIR / "leaderboard.csv"
    board.to_csv(board_out, index=False)
    print(f"\n[harness] saved leaderboard -> {board_out}")

    pretty = board.copy()
    for m in harness.METRIC_NAMES:
        pretty[m] = board[m].map(fmt)
    keep = ["predictor"] + harness.METRIC_NAMES + ["fold5_eval_rows"]
    print("\nLEADERBOARD (mean±std across the 5 canonical folds):\n" +
          pretty[keep].to_string(index=False))

    # -- Plot: per-fold MDA + aggregate comparison ----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors
    labels = ["persistence\n(canonical)", "drift", "arima(1,1,0)", "ridge-lags"]
    # gather per-fold MDA arrays
    base_ref = load_persistence_reference()
    per_fold_mda = {
        "persistence\n(canonical)": [pf["model"]["MDA"] for pf in base_ref["per_fold"]],
    }
    for i, run_id in enumerate(runs):
        per_fold_mda[labels[i + 1]] = [pf["model"]["MDA"] for pf in runs[run_id]["per_fold"]]

    ax = axes[0]
    for i, (lab, vals) in enumerate(per_fold_mda.items()):
        ax.plot(range(1, 6), vals, marker="o", color=colors[i % 10], label=lab)
    ax.set_xticks(range(1, 6),
                  [f"F{k}\n({2008 + 2*(k-1)}-{2009 + 2*(k-1)})" for k in range(1, 6)])
    ax.set_ylabel("MDA"); ax.set_ylim(0, 1)
    ax.set_title("Per-fold directional accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    means = [np.mean(v) for v in per_fold_mda.values()]
    stds = [np.std(v) for v in per_fold_mda.values()]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=[colors[i % 10] for i in range(len(labels))])
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("MDA mean ± std (5 folds)"); ax.set_ylim(0, 1)
    ax.set_title("Aggregate MDA vs persistence")
    for i, mv in enumerate(means):
        ax.text(i, mv + 0.02, f"{mv:.4f}", ha="center", fontsize=8)
    fig.suptitle("T2.1 baseline suite — which simple baseline is the bar?")
    fig.tight_layout()
    plot_path = OUT_DIR / "t2_1_leaderboard_mda.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[harness] saved plot -> {plot_path}")

    # -- Machine-readable summary ---------------------------------------------
    summary = {
        "run_id": "T2.1-baseline-suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": current_git_sha(),
        "arima_order": list(ARIMA_ORDER),
        "ridge_alpha": RIDGE_ALPHA,
        "leaderboard_mean_std": {
            row["predictor"]: {m: {"mean": row[m][0], "std": row[m][1]}
                               for m in harness.METRIC_NAMES}
            for _, row in board.iterrows()
        },
        "verdict_hint": "see analyses/T2.1.md",
    }
    save_results_json(summary, OUT_DIR / "suite_summary.json")
    print("\n=== T2.1 baseline suite complete ===")


if __name__ == "__main__":
    main()
