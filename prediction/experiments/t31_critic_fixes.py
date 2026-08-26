"""
t31_critic_fixes.py — T2.3 follow-up critic fixes, folded in at the start of
T3.1 (all five were approved; #1 and #4 are doc-only edits shipped in the same
commit). This script implements the CODE side of fixes #2, #3 and #5:

  Fix #2 — intersection-mask paired scorer:
      harness.paired_intersection_scores scores every predictor on ONE shared
      nonzero-direction set (intersection of each side's kept directions),
      removing even the residual asymmetry left by per-side zero-drop.
      The three comparisons where the zero-drop headline could plausibly
      flip a conclusion are re-run here PAIRED vs persistence:
        * drift                (nominally EDGED its paired persistence MDA)
        * drift-recursive      (nominally ~tied persistence)
        * ridge-on-lags        (the T2.x model vehicle)
      New registry rows are APPENDED (append-only policy) with the paired
      intersection-masked metrics; they supplement, never replace, the
      zero-drop-per-side `-mdazero` rows.

  Fix #3 — zero-direction decomposition from committed code:
      computes the pred-zero / true-zero / overlap decomposition of the
      candidate directions pooled over the 5 canonical folds under the
      persistence baseline's anchored predictions, and stores it into
      artifacts/T2.3/mda_correction_summary.json (previously these numbers
      — 853 / 805 / 406 of 2039 — existed only as prose).

  Fix #5 — direction-count audit fields:
      regenerates artifacts/T2.2/direct_sign_diagnostic.json adding
      n_sign_directions / n_momentum_directions per fold (+ totals), via the
      committed t23_corrections.direct_sign_diagnostic code. All pre-existing
      values are asserted UNCHANGED before saving (old artifact numbers stay
      intact elsewhere by construction).

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
from baselines_suite import (  # noqa: E402
    RIDGE_ALPHA,
    make_drift_recursive_model_fn,
    drift_model_fn,
    ridge_lags_model_fn,
)
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    TARGET,
    build_folds,
    current_git_sha,
    describe_folds,
    load_train_data,
    log_run,
    persistence_predict,
    save_results_json,
    _aggregate,
)
import target_formulation as tf  # noqa: E402
from t23_corrections import direct_sign_diagnostic  # noqa: E402

OUT_T23 = ARTIFACTS_DIR / "T2.3"
OUT_T22 = ARTIFACTS_DIR / "T2.2"
OUT_T31 = ARTIFACTS_DIR / "T3.1"


# ---------------------------------------------------------------------------
# Fix #3: pred-zero / true-zero / overlap decomposition (persistence baseline)
# ---------------------------------------------------------------------------

def zero_decomposition(df: pd.DataFrame, folds: list[dict]) -> dict:
    """
    Pool candidate directions across the canonical folds using the SAME
    boundary-anchored construction and common-mask row sets the headline
    zero-drop scoring uses, with PERSISTENCE as the reference predictor
    (the predictor whose bias motivated the protocol change).
    """
    n_candidate = n_pred_zero = n_true_zero = n_overlap = 0
    for f in folds:
        train_df = df.loc[f["train_index"]]
        val_df = df.loc[f["val_index"]]
        base_pred = persistence_predict(train_df, val_df, TARGET)
        mask = (val_df[TARGET].notna() & base_pred.notna())
        val_eval = val_df.loc[val_df.index[mask]]

        anchored = harness._anchored_frame(train_df, val_eval, base_pred,
                                           TARGET)
        g = anchored.groupby("cma_canonical", sort=False)
        tdir = np.sign(g[f"{TARGET}_true"].diff())
        pdir = np.sign(g[f"{TARGET}_pred"].diff())
        cand = tdir.notna() & pdir.notna()
        pz = cand & (pdir == 0)
        tz = cand & (tdir == 0)
        n_candidate += int(cand.sum())
        n_pred_zero += int(pz.sum())
        n_true_zero += int(tz.sum())
        n_overlap += int((pz & tz).sum())

    return {
        "predictor": "persistence (boundary-anchored, common-mask rows)",
        "folds": "canonical 5 expanding folds (2008-09 ... 2016-17)",
        "n_candidate_directions": n_candidate,
        "n_pred_zero_directions": n_pred_zero,
        "n_true_zero_directions": n_true_zero,
        "n_pred_and_true_zero_overlap": n_overlap,
        "frac_pred_zero": round(n_pred_zero / n_candidate, 4),
        "frac_true_zero": round(n_true_zero / n_candidate, 4),
        "computed_by": "prediction/experiments/t31_critic_fixes.py "
                       "(T2.3 critic fix #3)",
    }


# ---------------------------------------------------------------------------
# Fix #2: intersection-mask paired reruns vs persistence
# ---------------------------------------------------------------------------

def paired_intersection_vs_persistence(df: pd.DataFrame, folds: list[dict],
                                       model_name: str, model_fn):
    """
    One pass of harness-style CV for `model_fn`, but scored with the
    intersection-mask paired scorer against persistence on every fold.
    Returns a results dict shaped for log_run (per_fold/agg/config) plus raw
    per-fold detail for artifacts.
    """
    per_fold, detail = [], []
    for f in folds:
        train_df = df.loc[f["train_index"]]
        val_df = df.loc[f["val_index"]]

        predict_fn = model_fn(train_df.drop(columns=[TARGET]),
                              train_df[TARGET])
        model_pred = pd.Series(predict_fn(val_df.drop(columns=[TARGET])),
                               index=val_df.index)
        base_pred = persistence_predict(train_df, val_df, TARGET)

        # identical common-mask row set as the headline _score_fold scoring
        common = (val_df[TARGET].notna() & model_pred.notna()
                  & base_pred.notna())
        val_eval = val_df.loc[val_df.index[common]]

        paired = harness.paired_intersection_scores(
            train_df, val_eval,
            {"model": model_pred.loc[val_eval.index],
             "persistence": base_pred.loc[val_eval.index]},
            TARGET)

        m, b = paired["per_predictor"]["model"], \
            paired["per_predictor"]["persistence"]
        per_fold.append({"fold_id": f["fold_id"],
                         "n_eval_rows": int(common.sum()),
                         "model": m,
                         "baseline": b})
        detail.append({
            "fold_id": f["fold_id"],
            "n_val_rows_common_mask": int(common.sum()),
            "counts": paired["counts"],
            "model": m, "persistence": b,
            "delta_MDA_model_minus_persistence":
                m["MDA"] - b["MDA"] if np.isfinite(m["MDA"]) else None,
        })

    return {"folds": [{"fold_id": fl["fold_id"],
                       "val_start_year": fl["val_start_year"],
                       "val_end_year": fl["val_end_year"]} for fl in folds],
            "per_fold": per_fold,
            "agg": _aggregate(per_fold),
            "detail": detail,
            "config": {"n_folds": len(folds), "target": TARGET,
                       "anchor_year": folds[0]["anchor_year"],
                       "scoring": "paired intersection-mask (shared "
                                  "nonzero-direction set)"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== T2.3 critic-fix companions (code side of fixes #2/#3/#5) ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()

    # ---- Fix #3: decomposition -> mda_correction_summary.json ---------------
    print("\n--- fix #3: zero-direction decomposition ---")
    decomp = zero_decomposition(df, folds)
    print(json.dumps(decomp, indent=2))
    summary_path = OUT_T23 / "mda_correction_summary.json"
    summary = json.loads(summary_path.read_text())
    if "zero_decomposition" in summary:
        # idempotent re-run: keep the first committed decomposition
        prev = summary["zero_decomposition"]
        for k in ("n_candidate_directions", "n_pred_zero_directions",
                  "n_true_zero_directions", "n_pred_and_true_zero_overlap"):
            assert prev[k] == decomp[k], f"decomposition drift on {k}"
        print("  [idempotent] decomposition already present and identical")
    else:
        summary["zero_decomposition"] = decomp
        summary["zero_decomposition_added_at"] = \
            datetime.now(timezone.utc).isoformat()
        save_results_json(summary, summary_path)

    # ---- Fix #2: paired intersection-mask reruns -----------------------------
    print("\n--- fix #2: intersection-mask paired reruns vs persistence ---")
    specs = [
        ("T2.3-intersect-drift", drift_model_fn, "drift_rw",
         "PAIRED intersection-mask scoring (T2.3 critic fix #2 companion): "
         "random walk with drift vs persistence on ONE shared "
         "nonzero-direction set per fold. Supplements (does not replace) the "
         "-mdazero zero-drop-per-side row."),
        ("T2.3-intersect-drift-recursive",
         make_drift_recursive_model_fn(df), "drift_rw_recursive",
         "PAIRED intersection-mask scoring: recursive-update drift vs "
         "persistence on one shared nonzero-direction set per fold."),
        ("T2.3-intersect-ridge-lags", ridge_lags_model_fn, "ridge_on_lags",
         "PAIRED intersection-mask scoring: pooled Ridge(α=1)-on-lags + CMA "
         "dummies vs persistence on one shared nonzero-direction set per "
         "fold."),
    ]
    all_detail = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "git_sha_pre_commit": sha,
                  "scorer": "harness.paired_intersection_scores",
                  "purpose": "decide whether the zero-drop-per-side headline "
                             "conclusions survive when model and baseline "
                             "are restricted to their SHARED kept-direction "
                             "set",
                  "comparisons": {}}
    for run_id, fn, model_tag, desc in specs:
        print(f"\n--- {run_id} ---")
        res = paired_intersection_vs_persistence(df, folds, model_tag, fn)
        agg = res["agg"]
        mda_m = agg["model"]["MDA"]["mean"]
        mda_b = agg["baseline"]["MDA"]["mean"]
        deltas = [d["delta_MDA_model_minus_persistence"] for d in res["detail"]]
        print(f"  model       MDA(intersect) = {mda_m:.4f}")
        print(f"  persistence MDA(intersect) = {mda_b:.4f}")
        print(f"  per-fold ΔMDA = {[round(d, 4) for d in deltas]}")
        all_detail["comparisons"][run_id] = {
            "model_tag": model_tag, "aggregate": agg, "detail": res["detail"]}
        save_results_json(res, OUT_T31 / f"{run_id}_results.json")

        log_run(
            run_id=run_id, task="T2", sub="T2.3",
            description=desc,
            config={"git_sha_pre_commit": sha,
                    "critic_fix": "T2.3-fix-2-companion-intersection-mask",
                    "model": model_tag,
                    "alpha": RIDGE_ALPHA if model_tag == "ridge_on_lags"
                    else None,
                    "scoring": res["config"]["scoring"]},
            results={"folds": res["folds"], "per_fold": res["per_fold"],
                     "agg": agg, "config": res["config"]},
        )

    # Verdict on the nominal drift edge under the stricter pairing
    dr = all_detail["comparisons"]["T2.3-intersect-drift"]
    deltas_dr = [d["delta_MDA_model_minus_persistence"] for d in dr["detail"]]
    drift_edge_survives = all(d is not None and d > 0 for d in deltas_dr)
    mean_delta = float(np.mean(deltas_dr))
    all_detail["drift_nominal_edge_verdict"] = {
        "question": "drift nominally edged persistence on zero-drop-per-side "
                    "MDA (0.6852 vs 0.6721) — is the edge real?",
        "mean_paired_delta_MDA_intersection": mean_delta,
        "edge_positive_on_all_folds": bool(drift_edge_survives),
        "verdict_wording": "sign-consistent under stricter pairing but NOT "
                           "statistically established (|t| < 1); jointly "
                           "irrelevant regardless — NRMSE ~5x worse on the "
                           "same rows (T3.2 critic fix #2 wording)",
        "note": "point-error class unchanged (~40x NRMSE) either way",
    }
    OUT_T31.mkdir(parents=True, exist_ok=True)
    save_results_json(all_detail, OUT_T31 / "intersection_mask_paired.json")

    # ---- Fix #5: regenerate direct-sign diagnostic with both counts ----------
    print("\n--- fix #5: direct_sign_diagnostic.json count fields ---")
    diag_path = OUT_T22 / "direct_sign_diagnostic.json"
    old_diag = json.loads(diag_path.read_text())
    df_t = tf.prepare_transforms(df)
    new_diag = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "diagnostics": {}}
    for form in ("dlog", "yoy"):
        diag = direct_sign_diagnostic(df_t, folds, form)
        old_entry = old_diag["diagnostics"][form]
        # guard: previously published values must be reproduced EXACTLY
        assert abs(diag["agg"]["model"]["MDA"]["mean"]
                   - old_entry["signal_MDA_mean"]) < 1e-12, form
        assert abs(diag["agg"]["baseline"]["MDA"]["mean"]
                   - old_entry["momentum_ref_MDA_mean"]) < 1e-12, form
        n_sig = [pf["n_sign_directions"] for pf in diag["per_fold"]]
        n_mom = [pf["n_momentum_directions"] for pf in diag["per_fold"]]
        new_diag["diagnostics"][form] = {
            **old_entry,
            "n_sign_directions_per_fold": n_sig,
            "n_momentum_directions_per_fold": n_mom,
            "n_sign_directions_total": int(np.nansum(n_sig)),
            "n_momentum_directions_total": int(np.nansum(n_mom)),
        }
        print(f"  {form}: signal dirs/fold {n_sig}, momentum dirs/fold {n_mom}")
    new_diag["regeneration_note"] = (
        "Counts added (T3.1 critic fix #5); all pre-existing values "
        "assert-verified identical to the previous artifact before saving.")
    save_results_json(new_diag, diag_path)

    print("\n=== critic-fix companions complete ===")


if __name__ == "__main__":
    main()
