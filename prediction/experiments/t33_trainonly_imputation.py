"""
t33_trainonly_imputation.py — T3.3 HARD GATE: train-only re-imputation
sensitivity check (T1.2/T2.1 critic gate).

The upstream pipeline (imputation/data_imputation.py, executed pre-split)
fits annual->monthly distribution + bidirectional linear interpolation on
the TRAIN+TEST panel JOINTLY (impute_full_pipeline -> apply_all_rms_changes,
then splits). Any imputed train value may therefore be a function of
test-period observations, making CV absolutes optimistic.

This wrapper quantifies that optimism WITHOUT touching any file under
prediction/:
  1. Rebuilds the exact pre-split panel + per-CMA 80/20 split indices by
     executing the upstream module source UP TO (but excluding) the joint
     imputation call — byte-identical raw data, merges, structural column
     drops and split logic.
  2. Re-runs the IDENTICAL imputation estimators restricted to TRAIN ROWS
     ONLY (per CMA), producing an alternative training panel saved to
     artifacts/T3.3/X_train_trainonly_total.csv.gz (+ y ids file; gzipped to stay under the 10MB artifact cap).
  3. Verifies id alignment against the canonical y_train_full_total.csv and
     diffs every feature column against the joint-imputation panel.
  4. Re-runs the key reference models on the SAME canonical 5 folds using
     the train-only panel — persistence, ridge lag1-only, ridge +OWN — and
     logs T3.3-trainonly-* registry rows for head-to-head comparison.

The test-split files are NEVER loaded, touched, or regenerated here.
"""

import json
import sys
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
import temporal_features as tf  # noqa: E402
from baselines_suite import RIDGE_ALPHA, ridge_lags_model_fn  # noqa: E402
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

OUT_DIR = ARTIFACTS_DIR / "T3.3"
IMPUTATION_SRC = ROOT / "imputation" / "data_imputation.py"
RMS_PCT_GROUP = "RMS – Annual Average Rent Percent Change"


# ---------------------------------------------------------------------------
# 1-2. rebuild pre-split panel; train-only imputation
# ---------------------------------------------------------------------------

def load_upstream_namespace() -> dict:
    """
    Execute imputation/data_imputation.py source UP TO the joint imputation
    call, plus the prepare_data_for_feature_selection definition. Deliberately
    EXCLUDES everything after 'temp_df_imputed = impute_full_pipeline(' so no
    joint imputation runs and NO test-split code path executes.
    Returns the namespace (temp_df, columns_by_group, train_index, functions).
    """
    src = IMPUTATION_SRC.read_text()
    pre = src.split("temp_df_imputed = impute_full_pipeline(")[0]
    prep_start = src.index("def prepare_data_for_feature_selection")
    prep_end = src.index("#Ensure there are non-NA values")
    prep_src = src[prep_start:prep_end]

    ns: dict = {"__file__": str(IMPUTATION_SRC)}
    exec(compile(pre, str(IMPUTATION_SRC) + "[pre-imputation]", "exec"), ns)
    exec(compile(prep_src, str(IMPUTATION_SRC) + "[prepare_fn]", "exec"), ns)
    return ns


def trainonly_imputed_train_panel(ns: dict) -> pd.DataFrame:
    """
    Run the IDENTICAL imputation estimators (linear_impute per group +
    apply_all_rms_changes) fitted on TRAIN ROWS ONLY.
    """
    temp_df: pd.DataFrame = ns["temp_df"]
    columns_by_group: dict = ns["columns_by_group"]
    train_index: pd.Index = ns["train_index"]

    # restrict to train rows BEFORE any imputation transform sees the data
    work = temp_df.loc[train_index].copy()
    print(f"[trainonly] fitting imputation on {len(work)} train rows "
          f"(joint pipeline saw {len(temp_df)} rows)")
    for group, cols in columns_by_group.items():
        existing = [c for c in cols if c in work.columns]
        if not existing:
            continue
        if group == RMS_PCT_GROUP:
            continue  # derived after imputation, as upstream
        print(f"  -> {group} ({len(existing)} cols)")
        work = ns["linear_impute"](work, existing, distribute=False)
    work = ns["apply_all_rms_changes"](work, columns_by_group)

    # upstream post-imputation filters (identical)
    work = work.dropna(subset=["house", "land"])
    return work


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== T3.3 train-only re-imputation sensitivity (HARD GATE) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = current_git_sha()

    ns = load_upstream_namespace()
    work = trainonly_imputed_train_panel(ns)

    X_to, y_to, ids_to = ns["prepare_data_for_feature_selection"](
        work, target_col=TARGET, drop_target_nan=True)

    dropped_constant = list(X_to.columns[X_to.nunique() == 1])
    if dropped_constant:
        print(f"[trainonly] dropping constant features: {dropped_constant}")
        X_to = X_to.drop(columns=dropped_constant)

    # gzip keeps the committed artifact under the 10MB artifact cap
    X_to.to_csv(OUT_DIR / "X_train_trainonly_total.csv.gz",
                index=False, compression="gzip")
    y_export = pd.concat([ids_to.reset_index(drop=True),
                          y_to.reset_index(drop=True)], axis=1)
    y_export.to_csv(OUT_DIR / "y_train_trainonly_total.csv", index=False)

    # ---- alignment + column diff vs the canonical (joint) panel -------------
    y_canon = pd.read_csv(ROOT / "prediction" / "y_train_full_total.csv",
                          parse_dates=["date"])
    X_canon = pd.read_csv(ROOT / "prediction" / "X_train_full_total.csv")

    ids_new = y_export[ID_COLS].sort_values(ID_COLS).reset_index(drop=True)
    ids_old = y_canon[ID_COLS].sort_values(ID_COLS).reset_index(drop=True)
    ids_equal = ids_new.equals(ids_old)

    col_diff = {
        "n_cols_joint": int(X_canon.shape[1]),
        "n_cols_trainonly": int(X_to.shape[1]),
        "cols_only_in_joint": sorted(set(X_canon.columns) - set(X_to.columns)),
        "cols_only_in_trainonly": sorted(set(X_to.columns) - set(X_canon.columns)),
    }

    per_col_diff = []
    if ids_equal:
        common_cols = [c for c in X_canon.columns if c in X_to.columns]
        a = X_canon.set_index(
            [ids_old["cma_canonical"], ids_old["date"].astype(str)])
        b = X_to.set_index(
            [ids_new["cma_canonical"], ids_new["date"].astype(str)])
        for c in common_cols:
            va, vb = a[c].to_numpy(float), b[c].to_numpy(float)
            n_diff = int((~np.isclose(va, vb, rtol=0, atol=1e-12)).sum())
            if n_diff:
                per_col_diff.append({
                    "column": c, "n_cells_changed": n_diff,
                    "share_changed": round(n_diff / len(va), 4),
                    "mean_abs_delta": float(np.nanmean(np.abs(va - vb))),
                })
        per_col_diff.sort(key=lambda r: -r["n_cells_changed"])

    # ---- harness panel + folds ----------------------------------------------
    df_to = pd.concat([
        y_export[ID_COLS].reset_index(drop=True),
        X_to.reset_index(drop=True),
        y_export[TARGET].reset_index(drop=True),
    ], axis=1)
    df_to["date"] = pd.to_datetime(df_to["date"])
    df_to = df_to.sort_values(ID_COLS).reset_index(drop=True)

    df_joint = load_train_data(TARGET)
    folds_to = build_folds(df_to, n_folds=5, val_window_years=2)
    folds_joint = build_folds(df_joint, n_folds=5, val_window_years=2)
    fold_windows_match = [
        (a["val_start_year"], a["val_end_year"]) == (b["val_start_year"],
                                                     b["val_end_year"])
        for a, b in zip(folds_to, folds_joint)]
    assert all(fold_windows_match), "fold windows diverged between panels"
    print("fold design:\n" + describe_folds(folds_to))

    # ---- reference models on the train-only panel ----------------------------
    own_to, _ = tf.build_own_block(df_to)

    def make_persistence_model_fn(full_df):
        """True one-step persistence AS THE MODEL (y_hat_t = y_{t-1} within
        CMA, past-only), mirroring baselines_suite._past_lag1."""
        lag1 = full_df.sort_values(ID_COLS) \
            .groupby("cma_canonical")[TARGET].shift(1)

        def model_fn(tr_f, tr_y):
            def predict_fn(va_f):
                return pd.Series(lag1.loc[va_f.index].to_numpy(),
                                 index=va_f.index)
            return predict_fn
        return model_fn

    specs = [
        ("T3.3-trainonly-persistence", make_persistence_model_fn(df_to),
         "Persistence baseline recomputed on the train-only re-imputation "
         "panel (hard-gate sanity anchor: the target itself is NOT imputed "
         "upstream, so persistence must reproduce the canonical reference "
         "exactly)."),
        ("T3.3-trainonly-lag1only", ridge_lags_model_fn,
         "Pooled Ridge(1)-on-lags + CMA dummies on the TRAIN-ONLY "
         "re-imputation panel; identical canonical 5 folds."),
        ("T3.3-trainonly-add-own",
         tf.make_augmented_ridge_model_fn(own_to),
         "Pooled Ridge(1)-on-lags + CMA dummies +OWN temporal blocks on the "
         "TRAIN-ONLY re-imputation panel; identical canonical 5 folds."),
    ]
    res_by_id = {}
    for run_id, fn, desc in specs:
        print(f"\n--- running {run_id} ---")
        model_fn = fn
        res = evaluate(model_fn, df=df_to, folds=folds_to, target=TARGET)
        res_by_id[run_id] = res
        agg = res["agg"]
        fmt = lambda m, s: (f"{m:.4f}±{s:.4f}" if abs(m) > 1e-3
                            else f"{m:.3e}±{s:.3e}")
        print("  model      : " + "  ".join(
            f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}"
            for m in harness.METRIC_NAMES))
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        log_run(
            run_id=run_id, task="T3", sub="T3.3",
            description=desc, config={
                "git_sha_pre_commit": sha,
                "model": ("persistence_as_model" if "persistence" in run_id
                          else "ridge_on_lags+temporal"
                          if run_id.endswith("add-own") else "ridge_on_lags"),
                "alpha": RIDGE_ALPHA if "persistence" not in run_id else None,
                "panel": "train-only re-imputation "
                         "(imputation estimators fitted on train rows only)",
                "panel_artifact": "artifacts/T3.3/X_train_trainonly_total.csv.gz",
            }, results=res)

    # ---- head-to-head vs the corrected joint-imputation references ----------
    joint_ref = {
        "persistence": "T1.2-baseline-validation-mdazero",
        "lag1only": "T3.2-temporal-lag1only",   # latest-timestamp-wins
        "add_own": "T3.2-temporal-add-own",
    }
    canon_rows = [json.loads(l) for l in
                  open(harness.RUNS_JSONL)]
    latest = {}
    for r in canon_rows:
        latest[r["id"]] = r  # later lines overwrite earlier ones

    def agg_of(rid):
        r = latest[rid]
        return {m: r["metrics_agg"][m]["mean"] for m in harness.METRIC_NAMES}

    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "design": "identical canonical 5 folds (val 2008-09 .. 2016-17); "
                  "joint-imputation references = latest-timestamp-wins rows "
                  "of the named ids; train-only rows appended this stage",
        "pairs": [],
    }
    for label, to_id, joint_id in (
            ("persistence", "T3.3-trainonly-persistence",
             joint_ref["persistence"]),
            ("ridge_lag1only", "T3.3-trainonly-lag1only",
             joint_ref["lag1only"]),
            ("ridge_add_own", "T3.3-trainonly-add-own",
             joint_ref["add_own"])):
        to_m = {m: res_by_id[to_id]["agg"]["model"][m]["mean"]
                for m in harness.METRIC_NAMES}
        j_m = agg_of(joint_id)
        comparison["pairs"].append({
            "comparison": label,
            "trainonly_run_id": to_id,
            "joint_reference_run_id": joint_id,
            "trainonly": to_m,
            "joint": j_m,
            "delta_trainonly_minus_joint": {
                m: to_m[m] - j_m[m] for m in harness.METRIC_NAMES},
        })

    save_results_json(comparison,
                      OUT_DIR / "trainonly_vs_joint_comparison.json")
    save_results_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "ids_align_with_canonical_train_split": bool(ids_equal),
        "constant_columns_dropped_trainonly": dropped_constant,
        **col_diff,
        "n_columns_with_any_change": len(per_col_diff),
        "top_changed_columns": per_col_diff[:25],
        "note": "per-column diffs computed on aligned (cma, date) pairs; "
                "'changed' = |delta| > 1e-12",
    }, OUT_DIR / "panel_diff_summary.json")

    print("\nHEAD-TO-HEAD (means over the 5 canonical folds):")
    for p in comparison["pairs"]:
        print(f"  {p['comparison']:>16}: "
              + "  ".join(
                  f"{m} joint={p['joint'][m]:.5g} "
                  f"trainonly={p['trainonly'][m]:.5g} "
                  f"(d={p['delta_trainonly_minus_joint'][m]:+.2e})"
                  for m in harness.METRIC_NAMES))
    print(f"\nids align with canonical train split: {ids_equal}")
    print("=== T3.3 train-only gate complete ===")


if __name__ == "__main__":
    main()
