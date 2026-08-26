"""
harness.py — single source of truth for CV folds, metrics, and experiment logging.

Every later experiment (T2+) MUST go through this module so comparisons are
apples-to-apples:

  * Folds   : expanding-window over calendar years (~5 folds, 2-year validation
              windows anchored at the last year where >=50% of CMAs are still
              active), grouped by CMA.
              Random shuffles / KFold across time are banned.
  * Metrics : MDA / NMSE / NRMSE / NMAE reused from evaluation/evaluation.py.
              Persistence baseline computed on the IDENTICAL folds and scored
              on the IDENTICAL row sets.
  * Logging : one JSON line per run appended to prediction/experiments/runs.jsonl.

LEAKAGE RULES enforced here:
  * Only the train-split files (X_train_full_*, y_train_full_*) are ever loaded.
    Test-split files are forbidden until stage T8.
  * Model fitting sees ONLY rows with year < validation-window start.
  * The persistence baseline may use the last TRAIN observation to predict the
    first month of the validation window (past information), never future rows.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# experiments -> prediction -> repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluation import (  # noqa: E402  (needs ROOT on sys.path)
    calculate_all_metrics,
    mean_directional_accuracy_grouped,
)
PREDICTION_DIR = ROOT / "prediction"
EXPERIMENTS_DIR = ROOT / "prediction" / "experiments"
RUNS_JSONL = EXPERIMENTS_DIR / "runs.jsonl"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"

TARGET = "total"  # project rule: experiments use `total` ONLY
ID_COLS = ["cma_canonical", "date"]

METRIC_NAMES = ["MDA", "NMSE", "NRMSE", "NMAE"]


# ---------------------------------------------------------------------------
# Data loading (TRAIN SPLIT ONLY — test files are forbidden until T8)
# ---------------------------------------------------------------------------

def load_train_data(target: str = TARGET) -> pd.DataFrame:
    """
    Load the training split produced by imputation/data_imputation.py and return
    a single DataFrame: id columns + lagged feature columns + target column.

    X_train_full_{t}.csv and y_train_full_{t}.csv are row-aligned by position
    (the imputation export resets indices before saving), so we concatenate
    positionally after asserting equal length, then sort by (CMA, date).
    """
    x_path = PREDICTION_DIR / f"X_train_full_{target}.csv"
    y_path = PREDICTION_DIR / f"y_train_full_{target}.csv"

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path, parse_dates=["date"])

    missing = {"cma_canonical", "date", target} - set(y.columns)
    if missing:
        raise ValueError(f"{y_path.name} is missing columns {missing}")

    if len(X) != len(y):
        raise ValueError(
            f"Row misalignment: {x_path.name} ({len(X)}) vs {y_path.name} ({len(y)})"
        )

    df = pd.concat(
        [y[ID_COLS].reset_index(drop=True), X.reset_index(drop=True),
         y[[target]].reset_index(drop=True)],
        axis=1,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(ID_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold construction: expanding-window over years, grouped by CMA
# ---------------------------------------------------------------------------

def _default_anchor_year(df: pd.DataFrame, min_cma_fraction: float = 0.5) -> int:
    """
    Last calendar year in which at least `min_cma_fraction` of the CMAs still
    have observations. Anchoring validation windows here avoids degenerate
    folds covering years where only one or two late-starting CMAs exist
    (e.g. 2019-2022 in this dataset is Sherbrooke-only).
    """
    per_year = df.groupby(df["date"].dt.year)["cma_canonical"].nunique()
    threshold = int(np.ceil(min_cma_fraction * df["cma_canonical"].nunique()))
    eligible = per_year[per_year >= threshold]
    if eligible.empty:
        raise ValueError("No year meets the min-CMA-fraction requirement")
    return int(eligible.index.max())


def build_folds(df: pd.DataFrame, n_folds: int = 5,
                val_window_years: int = 2,
                anchor_year: int | None = None,
                min_cma_fraction: float = 0.5) -> list[dict]:
    """
    Expanding-window folds over calendar years. Validation windows are the last
    `n_folds * val_window_years` years up to `anchor_year` (defaults to
    _default_anchor_year), chopped into `n_folds` consecutive blocks; each fold
    trains on ALL rows strictly before its validation window (per CMA
    availability — late-starting CMAs simply contribute fewer rows and are
    skipped wherever they have no rows).

    Returns a list of fold dicts (chronological):
      {fold_id, val_start_year, val_end_year, train_index, val_index, ...counts}
    """
    if df["date"].isna().any():
        raise ValueError("NaT dates found — cannot build year-based folds")
    if anchor_year is None:
        anchor_year = _default_anchor_year(df, min_cma_fraction)
    years = df["date"].dt.year

    windows = []
    for k in range(n_folds):
        val_end = anchor_year - k * val_window_years
        val_start = val_end - val_window_years + 1
        windows.append((val_start, val_end))
    windows.reverse()  # chronological order

    folds = []
    for i, (val_start, val_end) in enumerate(windows, start=1):
        train_mask = years < val_start
        val_mask = years.between(val_start, val_end)

        # Sanity: strict temporal ordering, no overlap
        assert years[train_mask].max() < val_start
        assert not (train_mask & val_mask).any()
        assert val_mask.sum() > 0, f"Fold {i} has an empty validation window"

        folds.append({
            "fold_id": i,
            "val_start_year": val_start,
            "val_end_year": val_end,
            "anchor_year": anchor_year,
            "train_index": df.index[train_mask],
            "val_index": df.index[val_mask],
            "n_train_rows": int(train_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "n_train_cmas": int(df.loc[train_mask, "cma_canonical"].nunique()),
            "n_val_cmas": int(df.loc[val_mask, "cma_canonical"].nunique()),
        })
    return folds


def describe_folds(folds: list[dict]) -> str:
    return "\n".join(
        f"fold {f['fold_id']}: train years <{f['val_start_year']} "
        f"({f['n_train_rows']} rows / {f['n_train_cmas']} CMAs) | "
        f"val {f['val_start_year']}-{f['val_end_year']} "
        f"({f['n_val_rows']} rows / {f['n_val_cmas']} CMAs)"
        for f in folds
    )


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def persistence_predict(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        target: str = TARGET) -> pd.Series:
    """
    Persistence baseline: y_hat_t = y_{t-1} within the same CMA, where the
    observation at t-1 may lie in the train window (first val month) or in the
    validation window itself. Uses past observations only — no leakage.
    Returns a Series aligned to val_df.index; NaN where no prior obs exists.
    """
    combined = pd.concat(
        [train_df[ID_COLS + [target]], val_df[ID_COLS + [target]]]
    ).sort_values(ID_COLS)
    prev = combined.groupby("cma_canonical")[target].shift(1)
    return prev.reindex(val_df.index)


def last_value_model_fn(train_features: pd.DataFrame, train_target: pd.Series):
    """
    Naive 'frozen last-value' model used only as a harness smoke test: predicts
    each CMA's last TRAIN observation for every validation row (it cannot
    update within the window, unlike the true persistence baseline).
    Demonstrates the model_fn contract:
        model_fn(train_features_df_with_ids, train_y) -> predict_fn(val_X) -> Series
    """
    last = (
        pd.concat([train_features[ID_COLS], train_target.rename("_y")], axis=1)
        .sort_values(ID_COLS)
        .groupby("cma_canonical")["_y"].last()
    )

    def predict_fn(val_features: pd.DataFrame) -> pd.Series:
        return val_features["cma_canonical"].map(last)

    return predict_fn


# ---------------------------------------------------------------------------
# Metrics (wrapping evaluation/evaluation.py — never reimplemented here)
# ---------------------------------------------------------------------------

def _anchored_mda(train_df: pd.DataFrame, val_eval_df: pd.DataFrame,
                  pred: pd.Series, target: str = TARGET) -> float:
    """
    MDA including the train->validation boundary direction.

    For each CMA WITH train history we prepend its last TRAIN observation as an
    anchor row with true=pred=last-train-value. The anchor's own diff is NaN
    (first in group -> dropped by evaluation.mean_directional_accuracy_grouped)
    and contributes no scored direction, while the FIRST validation row's
    direction becomes well-defined instead of being lost.

    CMAs with NO train history in this fold (late starters whose entire series
    begins inside the validation window) cannot be anchored; instead they
    contribute UNANCHORED within-window directions, losing only their first
    in-window direction (its diff is NaN and dropped naturally). This keeps
    every predictable validation row represented in MDA.
    """
    train_last = (
        train_df.sort_values(ID_COLS)
        .groupby("cma_canonical", sort=False).tail(1)
        .set_index("cma_canonical")
    )
    frames = []
    for cma, g in val_eval_df.groupby("cma_canonical", sort=False):
        if cma not in train_last.index:
            # No train rows this fold -> unanchored contribution: the CMA's
            # first in-window direction is lost (NaN diff), the rest scored.
            frames.append(pd.DataFrame({
                "cma_canonical": g["cma_canonical"].tolist(),
                f"{target}_true": g[target].tolist(),
                f"{target}_pred": pred.loc[g.index].tolist(),
            }))
            continue
        a = train_last.loc[cma]
        frames.append(pd.DataFrame({
            "cma_canonical": [cma] + g["cma_canonical"].tolist(),
            f"{target}_true": [a[target]] + g[target].tolist(),
            f"{target}_pred": [a[target]] + pred.loc[g.index].tolist(),
        }))
    if not frames:
        return float("nan")
    anchored = pd.concat(frames, ignore_index=True)
    return mean_directional_accuracy_grouped(
        anchored, f"{target}_true", f"{target}_pred"
    )


def _score_fold(val_df: pd.DataFrame, train_df: pd.DataFrame,
                model_pred: pd.Series, baseline_pred: pd.Series,
                target: str = TARGET) -> dict:
    """
    Score model vs persistence on ONE fold.

    Both predictors are evaluated on the SAME row set: rows where the true
    value AND the model prediction AND the baseline prediction are all present
    (guarantees apples-to-apples even when persistence is NaN for a
    late-starting CMA's very first validation month).

    NMSE/NRMSE/NMAE come from evaluation.calculate_all_metrics on the
    unanchored validation rows (its grouped-MDA output is discarded in favour
    of the boundary-anchored MDA from _anchored_mda).
    """
    true = val_df[target]

    common_mask = true.notna() & model_pred.notna() & baseline_pred.notna()
    if common_mask.sum() == 0:
        raise ValueError("Empty common evaluation set for this fold")
    eval_idx = val_df.index[common_mask]

    val_eval = val_df.loc[eval_idx]

    def point_metrics(pred: pd.Series) -> dict:
        frame = pd.DataFrame({
            "cma_canonical": val_eval["cma_canonical"],
            "date": val_eval["date"],
            f"{target}_true": val_eval[target],
            f"{target}_pred": pred.loc[eval_idx],
        })
        m = calculate_all_metrics(frame, target)
        return {k: m[k] for k in ("NMSE", "NRMSE", "NMAE")}

    return {
        "n_eval_rows": int(common_mask.sum()),
        "model": {"MDA": _anchored_mda(train_df, val_eval, model_pred, target),
                  **point_metrics(model_pred)},
        "baseline": {"MDA": _anchored_mda(train_df, val_eval, baseline_pred, target),
                     **point_metrics(baseline_pred)},
    }


def _aggregate(per_fold: list[dict]) -> dict:
    """mean ± std across folds for each metric."""
    out = {}
    for block in ("model", "baseline"):
        out[block] = {}
        for m in METRIC_NAMES:
            vals = [f[block][m] for f in per_fold if np.isfinite(f[block][m])]
            out[block][m] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
            }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def evaluate(model_fn, df: pd.DataFrame | None = None,
             folds: list[dict] | None = None, target: str = TARGET) -> dict:
    """
    Run expanding-window CV for one model.

    Parameters
    ----------
    model_fn : callable(train_features_df, train_target_series) -> predict_fn
        predict_fn(val_features_df) -> array/Series aligned to val_features_df.
        Feature frames INCLUDE the id columns (cma_canonical, date); models
        needing purely numeric input should drop/encode them inside model_fn.
        Fitting sees ONLY the train window of each fold.
    df : prepared training DataFrame (loaded via load_train_data if None)
    folds : fold list (built via build_folds defaults if None)

    Returns
    -------
    {"folds": [...], "per_fold": [{fold_id, n_eval_rows, model, baseline}],
     "agg": {model: {M: {mean, std}}, baseline: {...}}, "config": {...}}
    """
    if df is None:
        df = load_train_data(target)
    if folds is None:
        folds = build_folds(df)

    per_fold = []
    for f in folds:
        train_df = df.loc[f["train_index"]]
        val_df = df.loc[f["val_index"]]

        predict_fn = model_fn(train_df.drop(columns=[target]), train_df[target])
        model_pred = pd.Series(predict_fn(val_df.drop(columns=[target])),
                               index=val_df.index)

        baseline_pred = persistence_predict(train_df, val_df, target)

        scores = _score_fold(val_df, train_df, model_pred, baseline_pred, target)
        per_fold.append({"fold_id": f["fold_id"], **scores})

    return {
        "folds": [
            {k: f[k] for k in ("fold_id", "val_start_year", "val_end_year",
                               "n_train_rows", "n_val_rows", "n_train_cmas",
                               "n_val_cmas")}
            for f in folds
        ],
        "per_fold": per_fold,
        "agg": _aggregate(per_fold),
        "config": {"n_folds": len(folds), "target": target,
                   "anchor_year": folds[0]["anchor_year"]},
    }


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def log_run(run_id: str, task: str, sub: str, description: str, config: dict,
            results: dict) -> None:
    """
    Append one JSON line per experiment run to prediction/experiments/runs.jsonl.
    Note: `git_sha`/`commit` record PRE-COMMIT HEAD (the sha of the worktree
    before this sub-stage's commit), not the final commit containing the run.
    Schema (superset of both the approved plan and the registry convention):
    {id, run_id, task, sub, description, config, metrics_per_fold, metrics_agg,
     baseline_metrics_agg, cv_metrics, baseline_metrics, timestamp, date,
     git_sha, commit}
    """
    now = datetime.now(timezone.utc)
    agg = results["agg"]
    cv_means = {m: agg["model"][m]["mean"] for m in METRIC_NAMES}
    base_means = {m: agg["baseline"][m]["mean"] for m in METRIC_NAMES}

    sha = current_git_sha()
    row = {
        "id": run_id,
        "run_id": run_id,
        "task": task,
        "sub": sub,
        "description": description,
        "config": {**config, **results["config"]},
        "metrics_per_fold": results["per_fold"],
        "metrics_agg": agg["model"],
        "baseline_metrics_agg": agg["baseline"],
        "cv_metrics": cv_means,
        "baseline_metrics": base_means,
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "git_sha": sha,
        "commit": sha,
    }

    RUNS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_JSONL, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(f"[harness] logged run '{run_id}' -> {RUNS_JSONL}")


def save_results_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[harness] saved results -> {path}")


# ---------------------------------------------------------------------------
# Smoke test: run the full machinery end-to-end with naive predictors
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== T1.1 harness smoke test ===")
    df = load_train_data(TARGET)
    print(f"train data: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df['cma_canonical'].nunique()} CMAs, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")

    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("\nfold design:\n" + describe_folds(folds))

    res = evaluate(last_value_model_fn, df=df, folds=folds, target=TARGET)

    def fmt(d):
        return (f"MDA={d['MDA']:.4f}  NMSE={d['NMSE']:.3e}  "
                f"NRMSE={d['NRMSE']:.4f}  NMAE={d['NMAE']:.4f}")

    print("\nper-fold (model = frozen last-value vs persistence baseline):")
    for pf in res["per_fold"]:
        print(f"  fold {pf['fold_id']} ({pf['n_eval_rows']} eval rows)")
        print(f"    model    : {fmt(pf['model'])}")
        print(f"    baseline : {fmt(pf['baseline'])}")

    print("\naggregate (mean ± std across folds):")
    for block in ("model", "baseline"):
        parts = [f"{m}={res['agg'][block][m]['mean']:.4f}"
                 f"±{res['agg'][block][m]['std']:.4f}" for m in METRIC_NAMES]
        print(f"  {block:9}: " + "  ".join(parts))

    save_results_json(res, ARTIFACTS_DIR / "T1.1" / "smoke_test_results.json")

    log_run(
        run_id="T1.1-harness-smoke",
        task="T1", sub="T1.1",
        description="Harness smoke test: frozen last-value predictor vs persistence "
                    "on the shared expanding-window folds (infrastructure check, "
                    "not a formal model result)",
        config={"model": "last_value_frozen", "val_window_years": 2},
        results=res,
    )
    print("\n=== smoke test complete ===")
