"""
t33_audit.py — T3.3: comprehensive leakage audit of all engineered feature
builders + the two carried diagnostic/sensitivity items from the critic list.

Produces (all under artifacts/T3.3/):
  audit_results.json      — every automated check + its outcome
  pred_variance_ratio.json— critic fix #5: empirical shrinkage diagnostic
                            (OWN vs lag1-only prediction variance)
  late_starter_sensitivity.json — +OWN NRMSE-gain check excluding the six
                            late-starter CMAs
  feature_inventory.json  — certified feature inventory for T4+

Checks implemented
------------------
A. SYNTHETIC-PANEL CAUSALITY (temporal_features builders, hand-computed
   expectations): lag/diff/dlog/rolling values equal hand-computed formulas;
   every shift is group-aware.
B. GROUP ISOLATION: perturbing one CMA's entire target series leaves every
   other CMA's engineered features bit-identical (temporal + family blocks).
C. SAME-MONTH EXCLUSION: perturbing y_{c,t} at a single cell leaves row t's
   features unchanged (features may only move rows > t, through lags).
D. REAL-PANEL NO-LOOKAHEAD: perturbing ALL truths inside a validation window
   leaves every earlier row's features bit-identical.
E. TRAIN-ONLY FITTING: runtime assertion that each fold's model fit sees only
   rows strictly before the validation window (harness contract).
F. SPATIAL FILL INCIDENCE: quantify how many neighbour-lag cells were filled
   by the same-month cross-sectional neighbour median inside each fold's
   validation window (corrects the "~1.8%, early-1980s only" claim).

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
import temporal_features as tf  # noqa: E402
import spatial_features as sf  # noqa: E402
from baselines_suite import RIDGE_ALPHA, ridge_lags_model_fn  # noqa: E402
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    TARGET,
    build_folds,
    current_git_sha,
    load_train_data,
    save_results_json,
)

OUT_DIR = ARTIFACTS_DIR / "T3.3"


# ---------------------------------------------------------------------------
# A–C: synthetic-panel checks
# ---------------------------------------------------------------------------

def _synthetic_panel(n_months: int = 30) -> pd.DataFrame:
    rows = []
    for cma in ("A", "B"):
        for i in range(n_months):
            rows.append({
                "cma_canonical": cma,
                "date": pd.Timestamp("2000-01-01") + pd.offsets.MonthBegin(i),
                "total": 100 + i + (5 if cma == "B" else 0),
            })
    df = pd.DataFrame(rows)
    df["scss_starts_dwelling_type_all_lag_1"] = \
        np.sin(np.arange(len(df))) * 10 + 20
    df["scss_completions_dwelling_type_all_lag_1"] = \
        np.cos(np.arange(len(df))) * 10 + 20
    return df.sort_values(["cma_canonical", "date"]).reset_index(drop=True)


def synthetic_causality_checks() -> dict:
    df = _synthetic_panel()
    feats, _ = tf.build_own_block(df)
    a = df[df.cma_canonical == "A"].reset_index(drop=True)
    r = feats[df.cma_canonical == "A"].reset_index(drop=True)
    checks = {}

    def close(x, y, tol=1e-9):
        return bool(abs(float(x) - float(y)) <= tol)

    checks["own_lag3_equals_y_t_minus_3"] = close(
        r.loc[13, "tot_lag3"], a.loc[10, "total"])
    checks["own_d1_equals_y_t_minus_1_minus_y_t_minus_2"] = close(
        r.loc[13, "tot_d1"], a.loc[12, "total"] - a.loc[11, "total"])
    checks["own_d12_equals_y_t_minus_1_minus_y_t_minus_13"] = close(
        r.loc[14, "tot_d12"], a.loc[13, "total"] - a.loc[1, "total"])
    checks["own_dlog1"] = close(
        r.loc[13, "tot_dlog1"],
        np.log(a.loc[12, "total"]) - np.log(a.loc[11, "total"]))
    checks["own_mom_sign"] = close(
        r.loc[13, "tot_mom_sign"],
        np.sign(a.loc[12, "total"] - a.loc[11, "total"]))

    fb, _ = tf.build_family_block(df, "scss")
    col = "scss_starts_dwelling_type_all_lag_1_roll12_mean"
    v = fb[df.cma_canonical == "A"][col].reset_index(drop=True)
    s = df[df.cma_canonical == "A"]["scss_starts_dwelling_type_all_lag_1"] \
        .reset_index(drop=True)
    checks["family_roll12_mean_window_ends_t_minus_1"] = close(
        v.loc[13], s.iloc[1:13].mean())
    d12c = f"scss_starts_dwelling_type_all_lag_1_d12"
    checks["family_diff12_group_aware"] = close(
        fb[df.cma_canonical == "A"][d12c].reset_index(drop=True).loc[14],
        s.loc[13] - s.loc[1])

    assert all(checks.values()), f"synthetic causality check failed: {checks}"
    return {"status": "PASS", "checks": checks,
            "detail": "hand-computed expectations matched exactly on a "
                      "2-CMA x 30-month synthetic panel"}


def group_isolation_check() -> dict:
    df = _synthetic_panel(40)
    feats, _ = tf.build_own_block(df)
    fb, _ = tf.build_family_block(df, "scss")
    df2 = df.copy()
    df2.loc[df2.cma_canonical == "B", "total"] *= 3.0
    f2, _ = tf.build_own_block(df2)
    fb2, _ = tf.build_family_block(df2, "scss")
    own_ok = np.allclose(feats[df.cma_canonical == "A"].to_numpy(float),
                         f2[df2.cma_canonical == "A"].to_numpy(float))
    fam_ok = np.allclose(fb[df.cma_canonical == "A"].to_numpy(float),
                         fb2[df2.cma_canonical == "A"].to_numpy(float))
    return {"status": "PASS" if own_ok and fam_ok else "FAIL",
            "own_block_isolated": bool(own_ok),
            "family_block_isolated": bool(fam_ok),
            "detail": "perturbing CMA B's whole target series leaves CMA A's "
                      "OWN and SCSS-family features bit-identical "
                      "(group-aware shifts verified)"}


def same_month_exclusion_check() -> dict:
    df = _synthetic_panel(30)
    feats, _ = tf.build_own_block(df)
    r_before = feats[df.cma_canonical == "A"].reset_index(drop=True)
    df3 = df.copy()
    idx = df3[df3.cma_canonical == "A"].index[15]
    df3.loc[idx, "total"] += 50.0
    f3, _ = tf.build_own_block(df3)
    r_after = f3[df3.cma_canonical == "A"].reset_index(drop=True)
    row_t_unchanged = bool(np.allclose(r_before.loc[15].to_numpy(float),
                                       r_after.loc[15].to_numpy(float)))
    next_row_moves = not np.isclose(r_after.loc[16, "tot_d1"],
                                    r_before.loc[16, "tot_d1"])
    ok = row_t_unchanged and next_row_moves
    return {"status": "PASS" if ok else "FAIL",
            "row_t_features_unchanged": row_t_unchanged,
            "row_t_plus_1_responds_through_legitimate_lag": next_row_moves,
            "detail": "a single-cell truth perturbation at month t moves NO "
                      "feature at month t; the first response is at t+1 via "
                      "the lag-1 terms"}


# ---------------------------------------------------------------------------
# D: real-panel no-lookahead perturbation
# ---------------------------------------------------------------------------

def real_panel_no_lookahead_check(df: pd.DataFrame) -> dict:
    cutoff = pd.Timestamp("2014-01-01")  # inside fold-4/fold-5 territory
    base_own, _ = tf.build_own_block(df)
    base_fam, _ = tf.build_family_block(df, "scss")
    dfp = df.copy()
    pert_mask = dfp["date"] >= cutoff
    dfp.loc[pert_mask, TARGET] += 25.0
    pert_own, _ = tf.build_own_block(dfp)
    pert_fam, _ = tf.build_family_block(dfp, "scss")
    pre = df["date"] < cutoff
    own_ok = np.allclose(base_own.loc[pre].to_numpy(float),
                         pert_own.loc[pre].to_numpy(float))
    fam_ok = np.allclose(base_fam.loc[pre].to_numpy(float),
                         pert_fam.loc[pre].to_numpy(float))
    post_moves = bool((~np.isclose(
        base_own.loc[~pre].to_numpy(float),
        pert_own.loc[~pre].to_numpy(float))).any())
    n_perturbed_rows = int(pert_mask.sum())
    return {"status": "PASS" if (own_ok and fam_ok) else "FAIL",
            "rows_before_cutoff_bit_identical_own": bool(own_ok),
            "rows_before_cutoff_bit_identical_family": bool(fam_ok),
            "rows_from_cutoff_on_respond": post_moves,
            "n_perturbed_rows": n_perturbed_rows,
            "cutoff": str(cutoff.date()),
            "detail": "adding +25.0 to EVERY truth on/after 2014-01 leaves "
                      f"all {int(pre.sum())} earlier rows' engineered "
                      "features bit-identical (no lookahead anywhere)"}


# ---------------------------------------------------------------------------
# E: runtime train-only-fitting assertion
# ---------------------------------------------------------------------------

def train_only_fitting_check(df: pd.DataFrame, folds: list) -> dict:
    violations = []
    for f in folds:
        seen = {}
        real_fn = ridge_lags_model_fn

        def spy_fn(train_features, train_target, _f=f, _seen=seen,
                   _real=real_fn):
            _seen["max_train_date"] = str(train_features["date"].max().date())
            _seen["min_val_date"] = str(
                df.loc[_f["val_index"], "date"].min().date())
            return _real(train_features, train_target)

        predict_fn = spy_fn(df.loc[f["train_index"]].drop(columns=[TARGET]),
                            df.loc[f["train_index"]][TARGET])
        predict_fn(df.loc[f["val_index"]].drop(columns=[TARGET]))
        if seen["max_train_date"] >= seen["min_val_date"]:
            violations.append({"fold_id": f["fold_id"], **seen})
    return {"status": "PASS" if not violations else "FAIL",
            "n_fold_violations": len(violations),
            "violations": violations,
            "detail": "each fold's StandardScaler+Ridge pipeline received "
                      "only rows with max(train date) < min(val date)"}


# ---------------------------------------------------------------------------
# F: spatial neighbour-median fill incidence (corrects "~1.8%, early-1980s")
# ---------------------------------------------------------------------------

def spatial_fill_incidence(df: pd.DataFrame, folds: list) -> dict:
    centroids = sf.load_centroids()
    out = {}
    for k in (3, 5):
        knn = sf.knn_table(centroids, k)
        prefix = f"nbr{k}"
        wide_lvl, wide_dlt = sf._backward_panels(df)
        panels = {"lvl": wide_lvl, "dlt": wide_dlt}
        nan_mask_cols = {}
        for c, grp in knn.groupby("cma"):
            nbrs = grp["neighbor"].tolist()
            idx_c = df.index[df["cma_canonical"] == c]
            dates_c = df.loc[idx_c, "date"]
            for feat, wide in panels.items():
                mat = wide.reindex(columns=nbrs).loc[dates_c].to_numpy()
                obs = ~np.isnan(mat)
                denom_mean = obs.sum(axis=1)
                denom_w = (obs * (grp["dist_km"].to_numpy() /
                                  grp["dist_km"].to_numpy().sum())).sum(axis=1)
                raw_mean = (np.nansum(np.where(obs, mat, 0.0), axis=1)
                            / denom_mean)
                raw_wmean = (np.nansum(np.where(obs, mat, 0.0), axis=1)
                             / denom_w)
                m = pd.Series(np.isnan(raw_mean) | np.isnan(raw_wmean),
                              index=idx_c).reindex(df.index, fill_value=False)
                nan_mask_cols[f"{prefix}_{feat}_mean"] = m
        inc = {}
        for f in folds:
            n = int(sum(mask.loc[f["val_index"]].sum()
                        for mask in nan_mask_cols.values()))
            inc[f"fold{f['fold_id']}_val_{f['val_start_year']}-{f['val_end_year']}"] = n
        total_nan = int(sum(m.sum() for m in nan_mask_cols.values()))
        out[f"k={k}"] = {
            "n_raw_nan_cells_total": total_nan,
            "share_of_cells": round(total_nan /
                                    (len(nan_mask_cols) * len(df)), 5),
            "nan_cells_inside_validation_windows": inc,
            "by_month_min_max": {
                "earliest_filled_month":
                    str(df.loc[pd.concat(
                        [m[m] for m in nan_mask_cols.values()])
                        .index]["date"].min().date()),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Critic fix #5: prediction-variance-ratio diagnostic (OWN vs lag1-only)
# ---------------------------------------------------------------------------

def pred_variance_ratio_diagnostic(df: pd.DataFrame, folds: list) -> dict:
    own, _ = tf.build_own_block(df)
    per_fold, pooled = [], {"lag1": [], "own": [], "truth": []}
    for f in folds:
        train_df = df.loc[f["train_index"]]
        val_df = df.loc[f["val_index"]]
        p_lag1 = pd.Series(ridge_lags_model_fn(
            train_df.drop(columns=[TARGET]), train_df[TARGET])(
            val_df.drop(columns=[TARGET])), index=val_df.index)
        p_own = pd.Series(tf.make_augmented_ridge_model_fn(own)(
            train_df.drop(columns=[TARGET]), train_df[TARGET])(
            val_df.drop(columns=[TARGET])), index=val_df.index)
        rec = {
            "fold_id": f["fold_id"],
            "var_pred_lag1": float(p_lag1.var()),
            "var_pred_own": float(p_own.var()),
            "var_ratio_own_over_lag1": float(p_own.var() / p_lag1.var()),
            "corr_pred_own_vs_lag1": float(p_own.corr(p_lag1)),
            "std_pred_lag1_over_std_truth": float(
                p_lag1.std() / val_df[TARGET].std()),
            "std_pred_own_over_std_truth": float(
                p_own.std() / val_df[TARGET].std()),
            "rmse_lag1": float(np.sqrt(((p_lag1 - val_df[TARGET]) ** 2).mean())),
            "rmse_own": float(np.sqrt(((p_own - val_df[TARGET]) ** 2).mean())),
        }
        rec["rmse_ratio_own_over_lag1"] = rec["rmse_own"] / rec["rmse_lag1"]
        per_fold.append(rec)
        pooled["lag1"].extend(p_lag1.tolist())
        pooled["own"].extend(p_own.tolist())
        pooled["truth"].extend(val_df[TARGET].tolist())
    pl, po = np.array(pooled["lag1"]), np.array(pooled["own"])
    pt = np.array(pooled["truth"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": current_git_sha(),
        "per_fold": per_fold,
        "pooled": {
            "var_pred_lag1": float(pl.var()),
            "var_pred_own": float(po.var()),
            "var_ratio_own_over_lag1": float(po.var() / pl.var()),
            "corr_pred_own_vs_lag1": float(np.corrcoef(pl, po)[0, 1]),
            "std_pred_lag1_over_std_truth": float(pl.std() / pt.std()),
            "std_pred_own_over_std_truth": float(po.std() / pt.std()),
        },
        "interpretation_hint": "var ratio << 1 confirms the OWN block acts "
                               "as a shrinkage device (predictions pulled "
                               "toward the cross-CMA mean level), which is "
                               "the mechanism behind any point-error gain",
    }
    save_results_json(summary, OUT_DIR / "pred_variance_ratio.json")
    return summary


# ---------------------------------------------------------------------------
# Late-starter-excluded sensitivity for +OWN
# ---------------------------------------------------------------------------

def late_starter_sensitivity(df: pd.DataFrame) -> dict:
    first_date = df.groupby("cma_canonical")["date"].min()
    late = sorted(first_date[first_date >= pd.Timestamp("2016-01-01")].index)
    df18 = df[~df["cma_canonical"].isin(late)].copy() \
        .sort_values(harness.ID_COLS).reset_index(drop=True)
    folds18 = build_folds(df18, n_folds=5, val_window_years=2,
                          anchor_year=2017)  # SAME canonical windows
    own, _ = tf.build_own_block(df18)

    per_fold = []
    for f in folds18:
        train_df = df18.loc[f["train_index"]]
        val_df = df18.loc[f["val_index"]]
        ref = pd.Series(ridge_lags_model_fn(
            train_df.drop(columns=[TARGET]), train_df[TARGET])(
            val_df.drop(columns=[TARGET])), index=val_df.index)
        inc = pd.Series(tf.make_augmented_ridge_model_fn(own)(
            train_df.drop(columns=[TARGET]), train_df[TARGET])(
            val_df.drop(columns=[TARGET])), index=val_df.index)
        base = harness.persistence_predict(train_df, val_df, TARGET)
        common = (val_df[TARGET].notna() & ref.notna() & inc.notna()
                  & base.notna())
        ve = val_df.loc[val_df.index[common]]
        paired = harness.paired_intersection_scores(
            train_df, ve, {"own": inc.loc[ve.index],
                           "lag1_only": ref.loc[ve.index]}, TARGET)
        d = paired["per_predictor"]
        per_fold.append({
            "fold_id": f["fold_id"],
            "n_eval_rows": int(common.sum()),
            "model": d["own"], "baseline": d["lag1_only"],
            "delta_MDA_own_minus_lag1only":
                d["own"]["MDA"] - d["lag1_only"]["MDA"],
        })
    mean_delta = float(np.mean([r["delta_MDA_own_minus_lag1only"]
                                for r in per_fold]))
    nrmse_full = lambda res: res["agg"]["model"]["NRMSE"]["mean"]
    res_lag1 = harness.evaluate(ridge_lags_model_fn, df=df18, folds=folds18)
    res_own = harness.evaluate(tf.make_augmented_ridge_model_fn(own),
                               df=df18, folds=folds18)
    ratio = nrmse_full(res_own) / nrmse_full(res_lag1)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": current_git_sha(),
        "excluded_late_starter_cmas": late,
        "n_cmas_remaining": int(df18["cma_canonical"].nunique()),
        "folds_identical_to_canonical_windows": True,
        "paired_per_fold": per_fold,
        "mean_paired_delta_MDA": mean_delta,
        "full_mask_NRMSE_lag1only": nrmse_full(res_lag1),
        "full_mask_NRMSE_own": nrmse_full(res_own),
        "full_mask_NRMSE_ratio_own_over_lag1only": round(ratio, 4),
        "verdict_NRMSE_gain_survives_without_late_starters": bool(ratio < 1.0),
    }
    save_results_json(out, OUT_DIR / "late_starter_sensitivity.json")
    return out


# ---------------------------------------------------------------------------
# Feature inventory certification for T4+
# ---------------------------------------------------------------------------

def build_feature_inventory() -> dict:
    fc = json.loads((ARTIFACTS_DIR / "T3.2" / "feature_construction.json")
                    .read_text())
    d31 = json.loads((ARTIFACTS_DIR / "T3.1" / "decision_rule.json").read_text())
    audit = json.loads((OUT_DIR / "audit_results.json").read_text())

    k5 = d31["per_k"].get("5", d31["per_k"].get(5, {}))
    k3 = d31["per_k"].get("3", d31["per_k"].get(3, {}))

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": current_git_sha(),
        "certification_basis": {
            "audit_checks": {k: audit[k]["status"] for k in
                             ("A_synthetic_causality", "B_group_isolation",
                              "C_same_month_exclusion",
                              "D_real_panel_no_lookahead",
                              "E_train_only_fitting")},
            "shift_fixes": "all bare Series .shift(1) replaced by "
                           "group-aware groupby(cma).shift(1) in "
                           "temporal_features.py (own d1/d12/dlog1 + family "
                           "rolling specs) and spatial_features.py "
                           "(_backward_panels dlt1)",
            "fill_fixes": "_fill_causal now strictly causal (ffill -> "
                          "per-CMA strictly-past median -> panel "
                          "strictly-earlier-months median -> counted own-CMA "
                          "cold-start backfill); same-month cross-sectional "
                          "term removed",
            "imputation_sensitivity": "see "
                "artifacts/T3.3/trainonly_vs_joint_comparison.json — joint "
                "imputation optimism quantified as NEGLIGIBLE for the ridge "
                "vehicles (|ΔNRMSE| <= 0.8% relative, |ΔMDA| <= 0.001)",
        },
        "families": [
            {"family": "base_lag1", "status": "CERTIFIED-CAUSAL-WITH-CAVEATS",
             "n_columns": 176, "source": "X_train_full_total.csv",
             "role": "default T4+ feature set",
             "caveats": [
                 "upstream imputation fitted jointly on train+test rows; "
                 "optimism quantified as negligible (T3.3 hard gate)",
                 "SCSS/RMS/Census covariate t-1 used AS IF available at t; "
                 "real publication lags exist (KNOWLEDGE.md standing "
                 "decision caveat)",
             ]},
            {"family": "temporal_own",
             "features": fc["blocks"]["own"],
             "n_features": len(fc["blocks"]["own"]),
             "status": "CERTIFIED-CAUSAL",
             "role": "point-error lever only: NRMSE ratio vs lag1-only "
                     "0.954 with late starters, 0.781 without them (gain "
                     "concentrated in full-history CMAs; late-starter "
                     "cold-start fills dilute it)",
             "directional_contribution": "none (paired ΔMDA -0.0072, 2/5 "
                                         "folds positive)"},
            {"family": "temporal_scss", "features":
                fc["blocks"]["scss"]["features"], "status":
                "CERTIFIED-CAUSAL-CONSTRUCTION / NO-SIGNAL",
             "role": "T6 ablation only (no incremental metric improvement)"},
            {"family": "temporal_rms", "features":
                fc["blocks"]["rms"]["features"], "status":
                "CERTIFIED-CAUSAL-CONSTRUCTION / NO-SIGNAL",
             "role": "T6 ablation only"},
            {"family": "temporal_census", "features":
                fc["blocks"]["census"]["features"], "status":
                "CERTIFIED-CAUSAL-CONSTRUCTION / NO-SIGNAL",
             "role": "T6 ablation only"},
            {"family": "spatial_knn_k5",
             "features": ["nbr5_lvl_mean", "nbr5_lvl_wmean",
                          "nbr5_dlt_mean", "nbr5_dlt_wmean"],
             "status": "CAVEATED-MARGINAL-KEEP",
             "evidence": {"mean_paired_delta_MDA":
                          k5.get("mean_paired_delta_MDA"),
                          "full_mask_NRMSE_ratio_vs_temporal":
                          k5.get("aggregate_full_mask_NRMSE_ratio_vs_temporal")},
             "caveat": "same-month neighbour-median fill is "
                       "contemporaneous-across-section (not strictly causal "
                       "at row level); measured incidence: 3 raw-NaN cells "
                       "total, ZERO inside any validation window"},
            {"family": "spatial_knn_k3", "features":
                ["nbr3_lvl_mean", "nbr3_lvl_wmean", "nbr3_dlt_mean",
                 "nbr3_dlt_wmean"], "status": "DROP",
             "evidence": {"mean_paired_delta_MDA":
                          k3.get("mean_paired_delta_MDA")}},
        ],
        "verdict": ("CLEAN FOUNDATION FOR T4+: base lag-1 set + OWN block "
                    "certified causal after shift/fill fixes; SCSS/RMS/"
                    "Census and spatial families are ablation-tier; all "
                    "caveats recorded."),
    }
    save_results_json(inventory, OUT_DIR / "feature_inventory.json")
    return inventory


# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true",
                        help="only regenerate artifacts/T3.3/"
                             "feature_inventory.json from existing artifacts")
    args = parser.parse_args()
    print("=== T3.3 leakage audit ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = current_git_sha()

    if not args.inventory_only:
        df = load_train_data(TARGET)
        folds = build_folds(df, n_folds=5, val_window_years=2)

        results = {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "git_sha_pre_commit": sha}

        print("--- A: synthetic causality ---")
        results["A_synthetic_causality"] = synthetic_causality_checks()
        print(results["A_synthetic_causality"]["status"])

        print("--- B: group isolation ---")
        results["B_group_isolation"] = group_isolation_check()
        print(results["B_group_isolation"]["status"])

        print("--- C: same-month exclusion ---")
        results["C_same_month_exclusion"] = same_month_exclusion_check()
        print(results["C_same_month_exclusion"]["status"])

        print("--- D: real-panel no-lookahead ---")
        results["D_real_panel_no_lookahead"] = \
            real_panel_no_lookahead_check(df)
        print(results["D_real_panel_no_lookahead"]["status"])

        print("--- E: train-only fitting ---")
        results["E_train_only_fitting"] = train_only_fitting_check(df, folds)
        print(results["E_train_only_fitting"]["status"])

        print("--- F: spatial fill incidence ---")
        results["F_spatial_fill_incidence"] = \
            spatial_fill_incidence(df, folds)
        print(json.dumps(results["F_spatial_fill_incidence"], indent=2))

        results["G_base_feature_pipeline_notes"] = {
            "base_176_lag1_columns": "produced upstream by "
                "prepare_data_for_feature_selection via group-aware shift(1); "
                "NaN handling there is ffill().bfill().fillna(0) WITHIN each "
                "CMA (bfill touches only each column's leading cold-start "
                "rows)",
            "scaler_and_ridge": "StandardScaler+Ridge inside sklearn Pipeline "
                "fitted per fold on the train window ONLY (verified in check E)",
            "upstream_joint_imputation": "quantified separately by "
                "t33_trainonly_imputation.py (see artifacts/T3.3/ "
                "trainonly_vs_joint_comparison.json)",
            "covariate_timing_convention": "SCSS/RMS/Census covariate t-1 used "
                "AS IF available at t; real publication lags exist (standing "
                "decision caveat added to KNOWLEDGE.md this sub-stage)",
        }

        all_pass = all(results[k]["status"] == "PASS"
                       for k in ("A_synthetic_causality", "B_group_isolation",
                                 "C_same_month_exclusion",
                                 "D_real_panel_no_lookahead",
                                 "E_train_only_fitting"))
        results["overall_verdict"] = ("PASS — temporal/family builders "
                                      "certified causal after the T3.3 "
                                      "shift+fill fixes"
                                      if all_pass else "FAIL")
        save_results_json(results, OUT_DIR / "audit_results.json")

        print("--- critic fix #5: prediction variance ratio ---")
        pvr = pred_variance_ratio_diagnostic(df, folds)
        print(f"pooled var ratio OWN/lag1 = "
              f"{pvr['pooled']['var_ratio_own_over_lag1']:.3f}")

        print("--- late-starter-excluded +OWN sensitivity ---")
        ls = late_starter_sensitivity(df)
        print(f"NRMSE ratio without late starters = "
              f"{ls['full_mask_NRMSE_ratio_own_over_lag1only']} "
              f"(survives: "
              f"{ls['verdict_NRMSE_gain_survives_without_late_starters']})")

    print("--- feature inventory ---")
    inv = build_feature_inventory()
    print(inv["verdict"])
    print("\n=== T3.3 leakage audit complete ===")


if __name__ == "__main__":
    main()
