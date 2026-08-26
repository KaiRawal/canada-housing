"""
t23_corrections.py — T2.3 critic-fix regeneration sweep.

T2.3 critic fix #2 changed the canonical MDA scorer: harness.mda_zero_drop now
drops directions where the PREDICTED or TRUE direction is exactly 0, applied
SYMMETRICALLY to model and baseline. This removes the systematic anti-
persistence bias of the train->val boundary anchor (persistence's first
predicted direction is forced to 0 there => an automatic miss under the old
rule), which handicapped persistence by roughly 24 forced misses per fold.

Because MDA numbers moved, EVERY affected reference number is regenerated here
and appended to runs.jsonl as CORRECTED rows (runs.jsonl is strictly
append-only; old rows are never edited):

    T2.1-drift-mdazero            (= old T2.1-drift)
    T2.1-arima-mdazero            (= old T2.1-arima)
    T2.1-ridge-lags-mdazero       (= old T2.1-ridge-lags == T2.2-level)
    T2.1-drift-recursive-mdazero  (= old T2.1-drift-recursive)
    T2.1-arima-recursive-mdazero  (= old T2.1-arima-recursive)
    T2.2-dlog-mdazero             (= old T2.2-dlog)
    T2.2-yoy-mdazero              (= old T2.2-yoy)

Point metrics (NMSE/NRMSE/NMAE) are bit-identical to the pre-fix rows — ONLY
MDA moves. The canonical persistence reference is regenerated separately by
validate_baseline.py (row T1.2-baseline-validation-mdazero).

Derived artifacts whose headline numbers live in them are OVERWRITTEN with
corrected versions (git history + runs.jsonl preserve the old values):
artifacts/T2.1/*_results.json (+ leaderboard.csv, suite_summary.json,
recursive_variants_summary.json) and artifacts/T2.2/T2.2-{dlog,yoy}_results.json
(+ head_to_head.csv, regenerated with the fix-#4 scientific-notation
formatting). The two PNG plots are NOT regenerated (visual only; noted stale).

Also runs critic fix #1 — the DIRECT-SIGN diagnostic for the Δlog and YoY%
target forms: score sign(raw Δlog pred) and sign(YoY-implied monthly change)
== sign(YoY pred) DIRECTLY against sign(Δy_true) on the IDENTICAL common-mask
rows as the level-space scoring, with a momentum reference (persistence's
implied direction = sign of the previous actual change) on the same rows.
Logged as T2.2-directsign-dlog / T2.2-directsign-yoy.

No test-split file is read anywhere in this script.
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
import target_formulation as tf  # noqa: E402
from baselines_suite import (  # noqa: E402
    RIDGE_ALPHA,
    make_arima_recursive_model_fn,
    make_drift_recursive_model_fn,
    arima_model_fn,
    drift_model_fn,
    ridge_lags_model_fn,
)
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    ID_COLS,
    RUNS_JSONL,
    TARGET,
    build_folds,
    current_git_sha,
    describe_folds,
    load_train_data,
    log_run,
    persistence_predict,
    save_results_json,
    _aggregate,
    _score_fold,
)

OUT_T21 = ARTIFACTS_DIR / "T2.1"
OUT_T22 = ARTIFACTS_DIR / "T2.2"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3 else f"{mean:.3e}±{std:.3e}"


def old_agg_mda(path: Path) -> float | None:
    """
    Pre-fix aggregate model MDA for a run id, read from the ORIGINAL runs.jsonl
    row (first occurrence). Robust to artifacts having already been
    overwritten by a previous (partial) sweep execution.
    """
    orig_id = path.name.replace("_results.json", "")
    if not RUNS_JSONL.exists():
        return None
    with open(RUNS_JSONL) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") == orig_id:
                mda = (row.get("metrics_agg", {})
                         .get("MDA", {}).get("mean"))
                return float(mda) if mda is not None else None
    return None


# ---------------------------------------------------------------------------
# Direct-sign diagnostic (critic fix #1)
# ---------------------------------------------------------------------------

def direct_sign_diagnostic(df_t: pd.DataFrame, folds: list[dict], form: str):
    """
    Score the RAW transformed-target predictions' SIGN directly against
    sign(Δy_true) on the exact same rows the level-space scoring used
    (common mask: truth & inverted prediction & persistence all present).

      dlog : signal direction = sign(Δlog_pred)          (monthly change sign)
      yoy  : signal direction = sign(yoy_pred)           (YoY-implied monthly
             change = yoy_pred / 12 — same sign)

    Reference on the same rows: persistence's IMPLIED direction under the
    anchored convention = sign(y_{t-1} − y_{t-2}) (momentum copying).
    Accuracy counted where BOTH sides' directions are nonzero (same symmetric
    zero-drop rule as the canonical MDA).
    """
    transform_col = {"dlog": "_dlog", "yoy": "_yoy"}[form]
    helper_cols = [c for c in df_t.columns if c.startswith("_")]

    # Momentum reference over the full train split (backward shifts only).
    srt = df_t.sort_values(ID_COLS)
    g = srt.groupby("cma_canonical")[TARGET]
    momentum = np.sign(g.shift(1) - g.shift(2))
    momentum.index = srt.index

    per_fold = []
    for f in folds:
        train_df = df_t.loc[f["train_index"]]
        val_df = df_t.loc[f["val_index"]]
        fit_mask = train_df[transform_col].notna()
        predict_fn = ridge_lags_model_fn(
            train_df.loc[fit_mask].drop(columns=[TARGET] + helper_cols),
            train_df.loc[fit_mask, transform_col],
        )
        raw = pd.Series(predict_fn(
            val_df.drop(columns=[TARGET] + helper_cols)), index=val_df.index)
        baseline_pred = persistence_predict(train_df, val_df, TARGET)

        # EXACT common mask used by _score_fold for this form's level scoring.
        if form == "dlog":
            inverted = val_df["_lag1"] * np.exp(raw.astype(float))
        else:
            inverted = val_df["_lag12"] * (1.0 + raw.astype(float) / 100.0)
        mask = (val_df[TARGET].notna() & inverted.notna()
                & baseline_pred.notna())
        idx = val_df.index[mask]

        true_dir = np.sign(val_df.loc[idx, TARGET]
                           - val_df.loc[idx, "_lag1"])
        sig_dir = np.sign(raw.loc[idx])
        mom_dir = momentum.loc[idx]

        def acc(a: pd.Series, b: pd.Series) -> tuple[float, int]:
            ok = (a != 0) & (b != 0)
            n = int(ok.sum())
            return (float((a[ok] == b[ok]).mean()) if n else float("nan")), n

        # Each side drops ITS OWN zero directions (same symmetric rule as the
        # canonical MDA), so the scored sets can differ slightly in size —
        # both counts are recorded rather than asserted equal.
        acc_sig, n_dir = acc(sig_dir, true_dir)
        acc_mom, n_dir_m = acc(mom_dir, true_dir)

        per_fold.append({
            "fold_id": f["fold_id"], "n_eval_rows": int(mask.sum()),
            "n_sign_directions": n_dir,
            "n_momentum_directions": n_dir_m,
            "model": {"MDA": acc_sig, "NMSE": float("nan"),
                      "NRMSE": float("nan"), "NMAE": float("nan")},
            "baseline": {"MDA": acc_mom, "NMSE": float("nan"),
                         "NRMSE": float("nan"), "NMAE": float("nan")},
        })

    return {
        "folds": [{"fold_id": fl["fold_id"],
                   "val_start_year": fl["val_start_year"],
                   "val_end_year": fl["val_end_year"]} for fl in folds],
        "per_fold": per_fold,
        "agg": _aggregate(per_fold),
        "config": {"n_folds": len(folds), "target": TARGET,
                   "anchor_year": folds[0]["anchor_year"],
                   "target_form": form,
                   "metric_note": "MDA field = raw-sign directional accuracy "
                                  "(symmetric zero-drop); baseline = momentum "
                                  "reference sign(y_{t-1}-y_{t-2})"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="recompute everything needed for artifacts but "
                             "SKIP re-logging the corrected -mdazero registry "
                             "rows already appended by a previous partial run")
    args = parser.parse_args()

    print("=== T2.3 critic-fix regeneration sweep (zero-direction-drop MDA) ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))

    sha = current_git_sha()
    shifts = []  # old -> new MDA bookkeeping

    # ---- T2.1 variants -------------------------------------------------------
    specs_t21 = [
        ("T2.1-drift", drift_model_fn,
         "Random walk with drift (level formulation). CORRECTED row: "
         "zero-direction-drop MDA (T2.3 critic fix #2); point metrics "
         "unchanged vs the pre-fix row.",
         {"model": "drift_rw", "target_form": "level"}),
        ("T2.1-arima", arima_model_fn,
         "Per-CMA statsmodels ARIMA(1,1,0) fit on the train window only. "
         "CORRECTED row: zero-direction-drop MDA; point metrics unchanged.",
         {"model": "arima(1, 1, 0)", "target_form": "level",
          "package": "statsmodels.tsa.arima.model.ARIMA"}),
        ("T2.1-ridge-lags", ridge_lags_model_fn,
         "Pooled Ridge(alpha=1)-on-lags + CMA dummies. CORRECTED row: "
         "zero-direction-drop MDA; point metrics unchanged.",
         {"model": "ridge_on_lags", "alpha": RIDGE_ALPHA}),
        ("T2.1-drift-recursive", make_drift_recursive_model_fn(df),
         "Recursive-update drift (actual y_{t-1} anchor). CORRECTED row: "
         "zero-direction-drop MDA; point metrics unchanged.",
         {"model": "drift_rw_recursive", "target_form": "level",
          "update_rule": "one-step-ahead, actual y_{t-1} anchor"}),
        ("T2.1-arima-recursive", make_arima_recursive_model_fn(df),
         "Recursive-update ARIMA(1,1,0). CORRECTED row: zero-direction-drop "
         "MDA; point metrics unchanged.",
         {"model": "arima(1, 1, 0)_recursive", "target_form": "level",
          "update_rule": "rolling one-step-ahead, .append(actuals)"}),
    ]

    t21_results = {}
    for run_id, fn, desc, cfg_extra in specs_t21:
        print(f"\n--- re-running {run_id} ---")
        res = harness.evaluate(fn, df=df, folds=folds, target=TARGET)
        t21_results[run_id] = res
        path = OUT_T21 / f"{run_id}_results.json"
        old = old_agg_mda(path)
        new = res["agg"]["model"]["MDA"]["mean"]
        base_new = res["agg"]["baseline"]["MDA"]
        shifts.append({"run_id": run_id, "old_MDA": old, "new_MDA": new,
                       "new_baseline_MDA_mean": base_new["mean"],
                       "new_baseline_MDA_std": base_new["std"]})
        print(f"  model MDA: {old if old is None else round(old, 4)} -> "
              f"{new:.4f}   persistence (paired): "
              f"{base_new['mean']:.4f}±{base_new['std']:.4f}")

        save_results_json(res, path)  # overwrite with corrected numbers
        if not args.resume:
            log_run(
                run_id=f"{run_id}-mdazero", task="T2", sub="T2.1",
                description=desc,
                config={"git_sha_pre_commit": sha,
                        "critic_fix": "T2.3-fix-2-zero-drop-MDA", **cfg_extra,
                        **res["config"]},
                results=res,
            )
        else:
            print(f"  [resume] registry row {run_id}-mdazero already appended; "
                  f"not duplicated")

    # Rebuild the T2.1 leaderboard + machine-readable summaries from the
    # corrected results so no committed artifact carries stale headline MDA.
    from baselines_suite import build_leaderboard
    board = build_leaderboard(t21_results)
    board.to_csv(OUT_T21 / "leaderboard.csv", index=False)
    pretty = board.copy()
    for m in harness.METRIC_NAMES:
        pretty[m] = board[m].map(lambda ms: fmt(ms[0], ms[1]))
    print("\nCORRECTED LEADERBOARD (mean±std):\n"
          + pretty[["predictor"] + harness.METRIC_NAMES].to_string(index=False))

    save_results_json(
        {"run_id": "T2.1-baseline-suite", "regenerated_under":
         "zero-direction-drop MDA (T2.3 critic fix #2)",
         "generated_at": datetime.now(timezone.utc).isoformat(),
         "git_sha_pre_commit": sha, "arima_order": [1, 1, 0],
         "ridge_alpha": RIDGE_ALPHA,
         "leaderboard_mean_std": {
             row["predictor"]: {m: {"mean": row[m][0], "std": row[m][1]}
                                for m in harness.METRIC_NAMES}
             for _, row in board.iterrows()},
         "verdict_hint": "see analyses/T2.1.md"},
        OUT_T21 / "suite_summary.json")
    save_results_json(
        {"run_id": "T2.1-recursive-variants",
         "regenerated_under": "zero-direction-drop MDA (T2.3 critic fix #2)",
         "generated_at": datetime.now(timezone.utc).isoformat(),
         "git_sha_pre_commit": sha,
         "runs": {rid: {"agg_model": res["agg"]["model"],
                        "per_fold": res["per_fold"]}
                  for rid, res in t21_results.items() if "recursive" in rid},
         "verdict_hint": "see analyses/T2.1.md (recursive variants section)"},
        OUT_T21 / "recursive_variants_summary.json")

    # ---- T2.2 target forms ----------------------------------------------------
    print("\n--- re-running T2.2 target forms ---")
    df_t = tf.prepare_transforms(df)
    t22_specs = [
        ("T2.2-level", "level"), ("T2.2-dlog", "dlog"), ("T2.2-yoy", "yoy")]
    t22_results = {}
    for run_id, form in t22_specs:
        res = tf.evaluate_target_form(df_t, folds, form)
        t22_results[run_id] = res
        path = OUT_T22 / f"{run_id}_results.json"
        old = old_agg_mda(path)
        new = res["agg"]["model"]["MDA"]["mean"]
        shifts.append({"run_id": run_id, "old_MDA": old, "new_MDA": new})
        print(f"  {run_id}: model MDA "
              f"{old if old is None else round(old, 4)} -> {new:.4f}")
        save_results_json(res, path)  # overwrite
        if not args.resume:
            log_run(
                run_id=f"{run_id}-mdazero", task="T2", sub="T2.2",
                description=f"CORRECTED {run_id} (zero-direction-drop MDA, T2.3 "
                            "critic fix #2; point metrics unchanged). Per-fold "
                            "entries now also record n_train_rows_used (critic "
                            "fix #5).",
                config={"git_sha_pre_commit": sha,
                        "critic_fix": "T2.3-fix-2-zero-drop-MDA",
                        "model": "ridge_on_lags", "alpha": RIDGE_ALPHA,
                        "target_form": form, **res["config"]},
                results=res,
            )

    # Regenerate head_to_head.csv with corrected numbers + fix-#4 formatting.
    rows = []
    for run_id, form in t22_specs:
        res = t22_results[run_id]
        agg_m, agg_b = res["agg"]["model"], res["agg"]["baseline"]
        rows.append({
            "predictor": run_id,
            **{m: fmt(agg_m[m]["mean"], agg_m[m]["std"])
               for m in harness.METRIC_NAMES},
            **{f"{m}_vs_persist_delta":
               fmt(agg_m[m]["mean"] - agg_b[m]["mean"], agg_m[m]["std"])
               for m in harness.METRIC_NAMES},
        })
    pd.DataFrame(rows).to_csv(OUT_T22 / "head_to_head.csv", index=False)

    # ---- Direct-sign diagnostic (critic fix #1) --------------------------------
    print("\n--- direct-sign diagnostic (dlog / yoy raw heads) ---")
    diag_summaries = {}
    for form in ("dlog", "yoy"):
        diag = direct_sign_diagnostic(df_t, folds, form)
        agg = diag["agg"]
        diag_summaries[form] = {
            "signal_MDA_mean": agg["model"]["MDA"]["mean"],
            "signal_MDA_std": agg["model"]["MDA"]["std"],
            "momentum_ref_MDA_mean": agg["baseline"]["MDA"]["mean"],
            "momentum_ref_MDA_std": agg["baseline"]["MDA"]["std"],
            "n_directions_per_fold": [pf["n_sign_directions"]
                                      for pf in diag["per_fold"]],
        }
        print(f"  {form}: raw-sign accuracy "
              f"{agg['model']['MDA']['mean']:.4f}"
              f"±{agg['model']['MDA']['std']:.4f}  vs momentum ref "
              f"{agg['baseline']['MDA']['mean']:.4f}"
              f"±{agg['baseline']['MDA']['std']:.4f}")
        log_run(
            run_id=f"T2.2-directsign-{form}", task="T2", sub="T2.2",
            description=f"Direct-sign diagnostic (T2.3 critic fix #1): "
                        f"sign({'Δlog' if form == 'dlog' else 'YoY-implied monthly'}) "
                        "prediction scored DIRECTLY against sign(Δy_true) on "
                        "the identical common-mask rows as the level-space "
                        "scoring; reference = momentum "
                        "sign(y_{t-1}-y_{t-2}) (persistence's implied "
                        "direction) on the same rows; symmetric zero-drop.",
            config={"git_sha_pre_commit": sha, "critic_fix": "T2.3-fix-1",
                    "model": "ridge_on_lags", "alpha": RIDGE_ALPHA,
                    "target_form": form,
                    "scoring": "raw-sign vs sign(delta_y_true)"},
            results=diag,
        )
    save_results_json({"generated_at": datetime.now(timezone.utc).isoformat(),
                       "diagnostics": diag_summaries},
                      OUT_T22 / "direct_sign_diagnostic.json")

    # ---- Shift bookkeeping ------------------------------------------------------
    summary = {
        "purpose": "old -> new aggregate model MDA under the symmetric "
                   "zero-direction-drop MDA protocol (T2.3 critic fix #2)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "shifts": shifts,
        "note": "Point metrics (NMSE/NRMSE/NMAE) are bit-identical pre/post "
                "fix; ONLY MDA moved. Canonical persistence reference is the "
                "T1.2-baseline-validation-mdazero row. PNG plots in "
                "artifacts/T2.1 and T2.2 were NOT regenerated (stale "
                "visually; tables/JSONs above are authoritative).",
    }
    save_results_json(summary, ARTIFACTS_DIR / "T2.3" / "mda_correction_summary.json")

    print("\n=== regeneration sweep complete ===")


if __name__ == "__main__":
    main()
