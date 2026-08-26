"""
validate_baseline.py — T1.2: validate the CV harness by formally reproducing
the persistence-baseline metrics under it.

What it does
------------
Run A (exact reproduction): re-runs the T1.1 smoke configuration
(`harness.evaluate` + `last_value_model_fn`) so the persistence baseline is
scored on the IDENTICAL folds and row sets as the smoke test, then asserts BOTH
the per-fold model (frozen last-value) AND baseline metric blocks match
artifacts/T1.1/smoke_test_results.json exactly.

Run B (canonical persistence reference): scores persistence AS BOTH the model
and the baseline (`_score_fold` with model_pred == baseline_pred), i.e. the
persistence predictor run through the harness on its own maximal evaluation
set. This produces the reference numbers all later models are judged against,
plus per-fold eval-row attrition (val rows lost to NaN persistence predictions).

T1.2 critic-fix re-run (2026-08-26): `_anchored_mda` now lets CMAs with no
train history contribute unanchored within-window directions (losing only
their first in-window direction), so fold 5's MDA includes the six
late-starter CMAs instead of silently dropping them. This script regenerates
the canonical reference numbers and appends a CORRECTED run row to runs.jsonl
(never edits old rows). Folds 1-4 are unchanged; only fold-5 MDA can move.

Cross-checks / sanity checks (all asserted, all reported):
  1. Persistence definition matches prediction/prediction.py exactly:
     y_hat_t = y_{t-1} within CMA, computed by a GLOBAL shift(1) over the whole
     sorted train split — compared element-wise against the harness's
     fold-local concatenated shift on every scored row (verifies the
     train->val boundary month is handled identically).
  2. No NaN among scored predictions/truths (guaranteed by the common mask;
     asserted post-hoc).
  3. Every fold's validation window starts strictly after its training window,
     including PER CMA (a CMA's last train date < its first val date).
  4. Eval-row attrition is accounted for exactly per fold:
     n_val_rows = n_eval_rows + dropped_truth_NaN + dropped_baseline_NaN,
     with a per-CMA breakdown of the baseline-NaN drops (late-starting CMAs
     whose first-ever observation falls inside the validation window).

Usage: python prediction/experiments/validate_baseline.py
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
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    ID_COLS,
    TARGET,
    _aggregate,
    _score_fold,
    build_folds,
    current_git_sha,
    describe_folds,
    evaluate,
    last_value_model_fn,
    load_train_data,
    log_run,
    persistence_predict,
    save_results_json,
)

RUN_ID = "T1.2-baseline-validation-mdafix"
SMOKE_JSON = ARTIFACTS_DIR / "T1.1" / "smoke_test_results.json"
OUT_JSON = ARTIFACTS_DIR / "T1.2" / "baseline_validation_results.json"


# ---------------------------------------------------------------------------
# Cross-check: prediction.py persistence convention, computed independently
# ---------------------------------------------------------------------------

def attach_global_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute persistence EXACTLY the way prediction/prediction.py does
    (global sort by (CMA, date), groupby.shift(1)) over the WHOLE train split,
    then merge it onto the harness DataFrame by (cma_canonical, date).
    """
    from prediction.prediction import calculate_persistence_predictions

    y_frame = df[ID_COLS + [TARGET]].copy()
    y_frame["global_pred"] = calculate_persistence_predictions(y_frame, TARGET)

    # Independent recomputation of the same convention (belt & braces):
    manual = (
        y_frame.sort_values(ID_COLS)
        .groupby("cma_canonical")[TARGET].shift(1)
        .rename("manual_pred")
    )
    y_frame["manual_pred"] = manual.reindex(y_frame.index)
    assert y_frame["global_pred"].equals(y_frame["manual_pred"]), \
        "prediction.py shift(1) convention does not match a manual grouped shift"

    return df.merge(
        y_frame[ID_COLS[:1] + ["date", "global_pred"]], on=ID_COLS, how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------------
# Attrition accounting
# ---------------------------------------------------------------------------

def attrition_for_fold(fold: dict, df: pd.DataFrame) -> dict:
    """Why did val rows fall out of the scored set? (truth NaN / baseline NaN)"""
    val_df = df.loc[fold["val_index"]]
    base_pred = persistence_predict(df.loc[fold["train_index"]], val_df, TARGET)
    truth_ok = val_df[TARGET].notna()
    base_ok = base_pred.notna()

    dropped_base = val_df.index[truth_ok & ~base_ok]
    per_cma = (
        val_df.loc[dropped_base]
        .groupby("cma_canonical")["date"].agg(["count", "min"])
        .reset_index().rename(columns={"count": "n_rows", "min": "first_date"})
    )
    return {
        "n_val_rows": int(len(val_df)),
        "n_dropped_truth_nan": int((~truth_ok).sum()),
        "n_dropped_baseline_nan": int((truth_ok & ~base_ok).sum()),
        "n_eval_rows": int((truth_ok & base_ok).sum()),
        "dropped_baseline_nan_by_cma": [
            {"cma": r["cma_canonical"], "n_rows": int(r["n_rows"]),
             "first_date": str(r["first_date"].date())}
            for _, r in per_cma.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== T1.2 persistence-baseline validation ===")
    df = load_train_data(TARGET)
    print(f"train split: {len(df)} rows, {df['cma_canonical'].nunique()} CMAs, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")

    df = attach_global_persistence(df)

    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("\nfold design:\n" + describe_folds(folds))

    # -- Sanity check 3: strict temporal ordering, globally and per CMA -------
    temporal_checks = []
    for f in folds:
        train_df, val_df = df.loc[f["train_index"]], df.loc[f["val_index"]]
        glob_ok = train_df["date"].max() < val_df["date"].min()
        per_cma_ok = True
        shared = set(train_df["cma_canonical"]) & set(val_df["cma_canonical"])
        for cma in shared:
            t_max = train_df.loc[train_df["cma_canonical"] == cma, "date"].max()
            v_min = val_df.loc[val_df["cma_canonical"] == cma, "date"].min()
            if not t_max < v_min:
                per_cma_ok = False
        temporal_checks.append({"fold_id": f["fold_id"],
                                "global_train_before_val": bool(glob_ok),
                                "per_cma_train_before_val": bool(per_cma_ok)})
        assert glob_ok and per_cma_ok, f"temporal ordering violated in fold {f['fold_id']}"
    print("\nsanity: train strictly precedes val in every fold "
          "(globally AND per CMA): OK")

    # -- Cross-check: harness persistence == prediction.py global persistence -
    mismatch = 0
    for f in folds:
        val_df = df.loc[f["val_index"]]
        base_pred = persistence_predict(df.loc[f["train_index"]], val_df, TARGET)
        scored = base_pred.notna()
        cmp_ = np.allclose(base_pred[scored].to_numpy(float),
                           val_df.loc[scored, "global_pred"].to_numpy(float),
                           rtol=0, atol=0)
        assert cmp_, f"persistence mismatch vs prediction.py in fold {f['fold_id']}"
        mismatch += int((base_pred[scored].to_numpy(float)
                         != val_df.loc[scored, "global_pred"].to_numpy(float)).sum())
    print(f"sanity: harness persistence == prediction.py global shift(1) on ALL "
          f"scored rows across 5 folds ({mismatch} mismatches): OK")

    # -- Run A: exact reproduction of the T1.1 smoke run ----------------------
    print("\nRun A: evaluate(last_value_model_fn) — reproduces the T1.1 smoke "
          "configuration exactly")
    res_a = evaluate(last_value_model_fn, df=df.drop(columns=["global_pred"]),
                     folds=folds, target=TARGET)

    smoke = json.loads(SMOKE_JSON.read_text())
    assert len(res_a["per_fold"]) == len(smoke["per_fold"]), (
        f"fold-list length mismatch: harness produced {len(res_a['per_fold'])} "
        f"folds but smoke JSON has {len(smoke['per_fold'])}"
    )
    repro_report = []
    for pf_a, pf_s in zip(res_a["per_fold"], smoke["per_fold"]):
        assert pf_a["fold_id"] == pf_s["fold_id"]
        for block in ("model", "baseline"):  # model = frozen last-value block
            for metric in harness.METRIC_NAMES:
                a, s = pf_a[block][metric], pf_s[block][metric]
                assert a == s, (f"{block} {metric} differs from smoke test in fold "
                                f"{pf_a['fold_id']}: {a} vs {s}")
        repro_report.append({
            "fold_id": pf_a["fold_id"],
            "matches_smoke_exactly": True,
            "n_eval_rows": pf_a["n_eval_rows"],
            "smoke_n_eval_rows": pf_s["n_eval_rows"],
        })
    print("Run A: per-fold model AND baseline metrics match artifacts/T1.1/"
          "smoke_test_results.json EXACTLY on all folds: OK")

    # -- Run B: persistence AS the model (canonical reference) ----------------
    print("\nRun B: persistence scored as both model and baseline "
          "(maximal evaluation set)")
    per_fold_b = []
    for f in folds:
        train_df, val_df = df.loc[f["train_index"]], df.loc[f["val_index"]]
        base_pred = persistence_predict(train_df, val_df, TARGET)
        scores = _score_fold(val_df, train_df, base_pred, base_pred.copy(), TARGET)

        # Sanity check 2: nothing NaN ever enters the scored set
        scored_idx = val_df.index[
            val_df[TARGET].notna() & base_pred.notna()
        ]
        assert val_df.loc[scored_idx, TARGET].notna().all()
        assert base_pred.loc[scored_idx].notna().all()
        assert np.isfinite(base_pred.loc[scored_idx].to_numpy(float)).all()

        # Sanity: scored preds coincide with prediction.py's global persistence
        assert np.allclose(
            base_pred.loc[scored_idx].to_numpy(float),
            df.loc[scored_idx, "global_pred"].to_numpy(float), rtol=0, atol=0,
        )

        per_fold_b.append({"fold_id": f["fold_id"], **scores})

    agg_b = _aggregate(per_fold_b)
    for block in ("model", "baseline"):
        assert agg_b[block] == _aggregate(
            [{k: v for k, v in pf.items() if k != "fold_id"} for pf in per_fold_b]
        )[block]

    def fmt(d):
        return (f"MDA={d['MDA']:.4f} NMSE={d['NMSE']:.3e} "
                f"NRMSE={d['NRMSE']:.4f} NMAE={d['NMAE']:.4f}")

    print("\nper-fold persistence reference (model == baseline):")
    for pf in per_fold_b:
        print(f"  fold {pf['fold_id']} ({pf['n_eval_rows']} eval rows): {fmt(pf['baseline'])}")
    print("\naggregate (mean ± std across folds):")
    parts = [f"{m}={agg_b['baseline'][m]['mean']:.4f}±{agg_b['baseline'][m]['std']:.4f}"
             for m in harness.METRIC_NAMES]
    print("  persistence: " + "  ".join(parts))

    # -- Attrition accounting --------------------------------------------------
    attrition = [dict(fold_id=f["fold_id"], **attrition_for_fold(f, df))
                 for f in folds]
    for a, pf in zip(attrition, per_fold_b):
        assert a["n_eval_rows"] == pf["n_eval_rows"]
        assert (a["n_val_rows"] == a["n_eval_rows"]
                + a["n_dropped_truth_nan"] + a["n_dropped_baseline_nan"])
    print("\nsanity: eval-row attrition accounts exactly for every val row "
          "(see artifact JSON for per-CMA detail)")

    # -- Assemble + persist artifact ------------------------------------------
    results_b = {
        "folds": res_a["folds"],
        "per_fold": per_fold_b,
        "agg": agg_b,
        "config": {"n_folds": len(folds), "target": TARGET,
                   "anchor_year": folds[0]["anchor_year"],
                   "predictor": "persistence (y_{t-1} within CMA)"},
    }
    artifact = {
        "run_id": RUN_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": current_git_sha(),
        "data_summary": {
            "file": "prediction/X_train_full_total.csv + y_train_full_total.csv",
            "n_rows": int(len(df)), "target": TARGET,
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "n_cmas": int(df["cma_canonical"].nunique()),
        },
        "fold_design": res_a["folds"],
        "temporal_ordering_checks": temporal_checks,
        "crosscheck_vs_prediction_py": {
            "method": "harness persistence_predict vs global groupby.shift(1) "
                      "(prediction.py convention), element-wise on every scored row",
            "n_exact_mismatches": mismatch,
            "status": "PASS",
        },
        "run_A_smoke_reproduction": {
            "description": "evaluate(last_value_model_fn); baseline block compared "
                           "to artifacts/T1.1/smoke_test_results.json",
            "per_fold": repro_report,
            "status": "PASS (exact float equality on all folds/metrics)",
            "agg_baseline": res_a["agg"]["baseline"],
        },
        "run_B_persistence_reference": results_b,
        "eval_row_attrition": attrition,
    }
    save_results_json(artifact, OUT_JSON)

    # -- Registry ---------------------------------------------------------------
    log_run(
        run_id=RUN_ID,
        task="T1", sub="T1.2",
        description="CORRECTED T1.2 reference (critic fix: _anchored_mda now "
                    "scores unanchored within-window directions for CMAs with "
                    "no train history, instead of dropping them): persistence "
                    "baseline reproduced through the harness on all 5 expanding "
                    "folds; per-fold model AND baseline blocks match the T1.1 "
                    "smoke run exactly (Run A eval sets unchanged); persistence "
                    "== prediction.py global shift(1) on every scored row. Run B "
                    "(model==baseline==persistence) is the canonical reference; "
                    "only fold-5 MDA changed vs the original row.",
        config={"model": "persistence", "validation_run": "A: smoke repro; "
                "B: persistence-as-model canonical reference"},
        results=results_b,
    )

    # Append a human-readable summary next to the JSON artifact
    lines = [
        "# T1.2 persistence-baseline validation — machine-readable summary",
        "",
        "| fold | val years | val rows | eval rows | dropped (NaN baseline pred) |"
        " MDA | NMSE | NRMSE | NMAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, f, pf in zip(attrition, folds, per_fold_b):
        b = pf["baseline"]
        lines.append(
            f"| {f['fold_id']} | {f['val_start_year']}-{f['val_end_year']} | "
            f"{a['n_val_rows']} | {a['n_eval_rows']} | "
            f"{a['n_dropped_baseline_nan']} | {b['MDA']:.4f} | {b['NMSE']:.3e} | "
            f"{b['NRMSE']:.4f} | {b['NMAE']:.4f} |"
        )
    ab = agg_b["baseline"]
    lines += [
        "",
        f"Aggregate: MDA={ab['MDA']['mean']:.4f}±{ab['MDA']['std']:.4f}, "
        f"NMSE={ab['NMSE']['mean']:.3e}±{ab['NMSE']['std']:.3e}, "
        f"NRMSE={ab['NRMSE']['mean']:.4f}±{ab['NRMSE']['std']:.4f}, "
        f"NMAE={ab['NMAE']['mean']:.4f}±{ab['NMAE']['std']:.4f}",
    ]
    summary_path = ARTIFACTS_DIR / "T1.2" / "baseline_validation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n[harness] saved summary -> {summary_path}")
    print("=== T1.2 validation complete ===")


if __name__ == "__main__":
    main()
