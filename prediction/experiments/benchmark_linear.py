"""
benchmark_linear.py — T4.1: tuned-ish LINEAR-FAMILY benchmark (Ridge /
ElasticNet) on certified feature sets, identical canonical folds.

Mini-goal: establish the linear-family ceiling on certified features under
identical folds, for the T4 leaderboard.

FEATURE SETS (fixed + NAMED for reuse in T4.4; source: artifacts/T3.3/
feature_inventory.json):
    A_base_lag1          : the 176 lag-1 columns from X_train_full_total.csv
                           (CERTIFIED-CAUSAL-WITH-CAVEATS)
    B_base_lag1_own      : A + 8 OWN-target temporal features
                           (tot_lag{3,6,12,24}, tot_d1/d12/dlog1/mom_sign;
                           CERTIFIED-CAUSAL after T3.3 shift/fill fixes)
    C_base_lag1_own_nbr5 : B + 4 k=5 neighbour-lag spatial features
                           (CAVEATED-MARGINAL-KEEP: same-month neighbour-
                           median fill, zero incidence inside val windows)

MODELS (standardization INSIDE the fold pipeline; CMA dummies as in
baselines_suite.ridge_lags_model_fn; late-starter CMAs handled unchanged via
dummy reindex fill 0):
    ridge_cv : sklearn GridSearchCV over a log-spaced alpha grid.
    enet     : GridSearchCV over a small alpha x l1_ratio grid.
    Both are Pipeline(StandardScaler_on_X, model) wrapped in
    TransformedTargetRegressor(StandardScaler_on_y): the TARGET is also
    standardized inside the fold pipeline because coordinate-descent
    ElasticNet does not converge at practical iteration budgets on raw
    NHPI levels (~100-350); with a standardized target it converges in
    <=~1200 iterations (equivalent model class up to an alpha rescaling;
    both scalers are fitted on the TRAIN WINDOW ONLY).
INNER-FIT DISCIPLINE (no leakage): every candidate is scored by an
EXPANDING, TIME-ORDERED inner CV computed WITHIN each fold's train window
ONLY — the final 4 blocks of 12 train months serve successively as inner
validation sets, training on strictly earlier calendar months, pooled across
CMAs (calendar cutoffs respect both time ordering and CMA grouping, unlike
KFold shuffles which are banned project-wide). The winning pipeline (X/y
scalers + model TOGETHER, so no statistic ever crosses the inner boundary)
is then refit on the FULL outer train window. Chosen hyperparameters are
captured per fold and saved.

DECISION RULE (stated up front, per approved plan): winner = best mean MDA
subject to mean NRMSE not >10% worse than the best mean NRMSE across the six
linear combos; ties broken by the simpler model (Ridge < ElasticNet;
smaller feature set < larger). Multi-metric standing bar still applies vs
persistence: a directional edge alone never constitutes "better".

EVALUATION: all set x model combos through harness.evaluate on the canonical
5 expanding folds; persistence scored alongside on identical folds/rows by
the harness itself; PAIRED intersection-mask deltas (harness.
paired_intersection_scores) vs BOTH the ridge alpha=1 lag1-only baseline
(MDA 0.6051 / NRMSE 0.00169) and persistence (MDA 0.6771±0.0474 / NRMSE
0.00113), all predictors anchored identically.

No test-split file is read anywhere in this script.
"""

import argparse
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
import temporal_features as tf  # noqa: E402
import spatial_features as sf  # noqa: E402
from baselines_suite import RIDGE_ALPHA, ridge_lags_model_fn  # noqa: E402
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    TARGET,
    build_folds,
    current_git_sha,
    describe_folds,
    evaluate,
    load_train_data,
    log_run,
    save_results_json,
)

OUT_DIR = ARTIFACTS_DIR / "T4.1"

# ---- Hyperparameter grids -------------------------------------------------
RIDGE_ALPHAS = list(np.logspace(-3, 3, 25))
ENET_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
ENET_L1_RATIOS = [0.05, 0.5, 0.95]

# ---- Inner-CV design (train window ONLY, time-ordered, expanding) ---------
INNER_N_BLOCKS = 4
INNER_BLOCK_MONTHS = 12
INNER_MIN_HISTORY_MONTHS = 24


def inner_time_splits(dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding, time-ordered inner CV within one fold's train window:
    the LAST INNER_N_BLOCKS blocks of INNER_BLOCK_MONTHS train months are,
    successively, inner-validation sets; training uses strictly earlier
    calendar months (pooled across CMAs — calendar cutoffs cannot leak
    across CMA groups because every CMA shares the same month axis).
    Returns (train_positions, val_positions) pairs into the frame.
    """
    per = dates.dt.to_period("M")
    months = sorted(per.unique())  # Period objects sort chronologically
    pos = np.arange(len(dates))
    vals = per.to_numpy()
    splits = []
    end = len(months)
    for _ in range(INNER_N_BLOCKS):
        vstart = end - INNER_BLOCK_MONTHS
        if vstart < INNER_MIN_HISTORY_MONTHS:
            break
        va_mask = (vals >= months[vstart]) & (vals <= months[end - 1])
        tr_mask = vals < months[vstart]
        assert tr_mask.sum() > 0 and va_mask.sum() > 0
        splits.append((pos[tr_mask], pos[va_mask]))
        end = vstart
    assert len(splits) >= 2, "inner time splits collapsed"
    return splits


# ---------------------------------------------------------------------------
# Tuned linear model factories (harness model_fn contract)
# ---------------------------------------------------------------------------

def _design(train_or_val: pd.DataFrame,
            extra: pd.DataFrame | None) -> pd.DataFrame:
    fr = train_or_val.join(extra) if extra is not None else train_or_val
    return pd.get_dummies(fr.drop(columns=["date"]),
                          columns=["cma_canonical"])


def make_tuned_linear_fn(blocks: tuple[pd.DataFrame, ...], kind: str,
                         chosen_log: list, pred_stash: list | None = None):
    """
    kind='ridge_cv' -> Ridge alpha grid; kind='enet' -> ElasticNet
    alpha x l1_ratio grid. Appends precomputed engineered blocks (OWN /
    spatial) exactly like temporal_features.make_augmented_ridge_model_fn.
    Records each fold's chosen hyperparameters into chosen_log (fold order)
    and, if `pred_stash` is a list, each predict_fn call's predictions (used
    by the paired-delta block to avoid refitting).
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    extra = pd.concat(blocks, axis=1) if blocks else None

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        X_tr = _design(train_features, extra)
        if X_tr.isna().any().any() or \
                not np.isfinite(X_tr.to_numpy(float)).all():
            raise ValueError("NaN/non-finite values in training features")
        splits = inner_time_splits(train_features["date"])

        if kind == "ridge_cv":
            inner = Pipeline([("scaler", StandardScaler()),
                              ("model", Ridge())])
            grid = {"regressor__model__alpha": RIDGE_ALPHAS}
        elif kind == "enet":
            inner = Pipeline([("scaler", StandardScaler()),
                              ("model", ElasticNet(max_iter=50000))])
            grid = {"regressor__model__alpha": ENET_ALPHAS,
                    "regressor__model__l1_ratio": ENET_L1_RATIOS}
        else:
            raise ValueError(kind)

        # y-standardization lives INSIDE the fold pipeline (train-only):
        # required for ElasticNet coordinate-descent convergence on raw
        # NHPI levels; equivalent model class up to alpha rescaling.
        est = TransformedTargetRegressor(regressor=inner,
                                         transformer=StandardScaler())
        gs = GridSearchCV(est, grid, cv=splits,
                          scoring="neg_mean_squared_error",
                          refit=True, n_jobs=-1)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", UserWarning)
            gs.fit(X_tr.to_numpy(float), train_target.to_numpy())
            n_conv_warnings = sum(
                1 for wi in w if issubclass(wi.category, ConvergenceWarning))
            n_user_warnings = sum(
                1 for wi in w if issubclass(wi.category, UserWarning))
        # n_iter_ is available on the fitted ElasticNet after refit on full
        # outer train window (coordinate-descent iteration count; evidence that
        # target standardization fixed convergence).
        try:
            # fitted pipeline lives on regressor_ (with trailing underscore)
            fitted_model = gs.best_estimator_.regressor_.named_steps["model"]
            n_iter_val = getattr(fitted_model, "n_iter_", None)
            if n_iter_val is not None:
                n_iter_val = int(np.asarray(n_iter_val).max()) \
                    if np.asarray(n_iter_val).size > 1 else int(n_iter_val)
        except Exception:
            n_iter_val = None
        chosen_log.append({
            "best_params": {k: (float(v) if isinstance(v, (int, float,
                                                            np.floating))
                                else v)
                            for k, v in gs.best_params_.items()},
            "best_inner_cv_nmse": float(-gs.best_score_),
            "n_inner_splits": len(splits),
            "n_convergence_warnings": int(n_conv_warnings),
            "n_user_warnings": int(n_user_warnings),
            "n_iter_": n_iter_val,
        })

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            X_va = _design(val_features, extra) \
                .reindex(columns=X_tr.columns, fill_value=0.0)
            preds = pd.Series(gs.predict(X_va.to_numpy(float)),
                              index=val_features.index)
            if pred_stash is not None:
                pred_stash.append(preds.copy())
            return preds

        return predict_fn

    return model_fn


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3 \
        else f"{mean:.3e}±{std:.3e}"


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true",
                        help="recompute artifacts but skip runs.jsonl rows")
    parser.add_argument("--post-only", action="store_true",
                        help="skip CV/pairing; rebuild combo table, decision "
                             "rule and plot from the saved per-run result "
                             "JSONs (used after a mid-run crash)")
    args = parser.parse_args()

    print("=== T4.1 linear-family benchmark (Ridge/ElasticNet) ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Feature sets (fixed + named for T4.4 reuse) -----------------------
    own, _ = tf.build_own_block(df)
    centroids = sf.load_centroids()
    knn5 = sf.knn_table(centroids, 5)
    spat5 = sf.build_spatial_features_strict(df, knn5, "nbr5")

    feature_sets = {
        "A_base_lag1": (),
        "B_base_lag1_own": (own,),
        "C_base_lag1_own_nbr5": (own, spat5),
    }
    save_results_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "source_inventory": "artifacts/T3.3/feature_inventory.json",
        "sets": {name: {"n_engineered_extra": sum(b.shape[1]
                                                  for b in blocks),
                        "engineered_columns": [
                            c for b in blocks for c in b.columns]}
                 for name, blocks in feature_sets.items()},
        "note": "base lag-1 = the 176 columns of X_train_full_total.csv "
                "(plus 24 CMA dummies inside the pipeline); reused VERBATIM "
                "by name in the T4.4 leaderboard.",
    }, OUT_DIR / "feature_sets.json")

    # ---- Ridge alpha=1 lag1-only reference predictions (pairing anchor) ---
    # ---- Set x model CV runs ----------------------------------------------
    combos = [(f"{name.split('_')[0]}-{kind}", name, kind)
              for name in feature_sets for kind in ("ridge_cv", "enet")]

    if not args.post_only:
        ref_preds = {}
        for f in folds:
            train_df = df.loc[f["train_index"]]
            val_df = df.loc[f["val_index"]]
            fn = ridge_lags_model_fn(train_df.drop(columns=[TARGET]),
                                     train_df[TARGET])
            ref_preds[f["fold_id"]] = pd.Series(
                fn(val_df.drop(columns=[TARGET])), index=val_df.index)

        runs, chosen_by_combo, preds_by_combo = {}, {}, {}
        for combo_id, set_name, kind in combos:
            chosen_log = []
            stash: list = []
            fn = make_tuned_linear_fn(feature_sets[set_name], kind,
                                      chosen_log, pred_stash=stash)
            run_id = f"T4.1-{combo_id}"
            print(f"\n--- running {run_id} ({set_name} x {kind}) ---")
            res = evaluate(fn, df=df, folds=folds, target=TARGET)
            runs[run_id] = res
            chosen_by_combo[run_id] = chosen_log
            preds_by_combo[run_id] = stash
            agg = res["agg"]
            print("  model      : " +
                  "  ".join(f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}"
                            for m in harness.METRIC_NAMES))
            print("  persistence: " +
                  "  ".join(f"{m}={fmt(agg['baseline'][m]['mean'], agg['baseline'][m]['std'])}"
                            for m in harness.METRIC_NAMES))
            print(f"  chosen per fold: "
                  f"{[c['best_params'] for c in chosen_log]}")

            cfg = {
                "git_sha_pre_commit": sha,
                "model": ("ridge_alpha_grid" if kind == "ridge_cv"
                          else "elasticnet_grid"),
                "feature_set": set_name,
                "n_base_features": 176,
                "n_engineered_extra": sum(b.shape[1] for b in
                                          feature_sets[set_name]),
                "ridge_alphas": RIDGE_ALPHAS if kind == "ridge_cv" else None,
                "enet_alphas": ENET_ALPHAS if kind == "enet" else None,
                "enet_l1_ratios": ENET_L1_RATIOS if kind == "enet" else None,
                "inner_cv": {
                    "scheme": "expanding time-ordered calendar-cutoff blocks "
                              "WITHIN the fold train window ONLY",
                    "n_blocks": INNER_N_BLOCKS,
                    "block_months": INNER_BLOCK_MONTHS,
                    "min_history_months": INNER_MIN_HISTORY_MONTHS,
                    "scoring": "neg_mean_squared_error",
                    "refit": "winning pipeline (X-scaler + y-standardizer + "
                             "model) on full outer train window; no statistic "
                             "crosses the inner boundary",
                    "target_handling":
                        "TransformedTargetRegressor(StandardScaler) inside "
                        "the fold pipeline (train-only); needed for "
                        "ElasticNet convergence, equivalent model class up "
                        "to alpha rescaling",
                    "chosen_per_fold": chosen_log,
                },
                "late_starter_handling": "unchanged: dummy reindex fill 0",
            }
            save_results_json(res, OUT_DIR / f"{run_id}_results.json")
            if not args.no_log:
                log_run(
                    run_id=run_id, task="T4", sub="T4.1",
                    description=(
                        f"Linear-family benchmark '{combo_id}': "
                        f"{'Ridge alpha-grid' if kind == 'ridge_cv' else 'ElasticNet alpha x l1_ratio grid'} "
                        f"(inner expanding time-ordered CV inside the train "
                        f"window only, scaler inside pipeline) on feature set "
                        f"{set_name}; canonical 5 folds; persistence scored "
                        "alongside by the harness."),
                    config=cfg, results=res,
                )

        # ---- Paired intersection-mask deltas vs ridge-a1 AND persistence ---
        print("\n--- paired intersection-mask deltas ---")
        delta_rows, paired_summary = [], {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha_pre_commit": sha,
            "references": {
                "ridge_a1_lag1only": {"MDA": 0.6051, "NRMSE": 0.00169},
                "persistence": {"MDA": 0.6771, "NRMSE": 0.00113},
            },
            "per_combo": {},
        }
        for combo_id, set_name, kind in combos:
            detail = []
            run_id = f"T4.1-{combo_id}"
            for fi, f in enumerate(folds):
                train_df = df.loc[f["train_index"]]
                val_df = df.loc[f["val_index"]]
                m_pred = preds_by_combo[run_id][fi]  # stashed CV predictions
                r_pred = ref_preds[f["fold_id"]]
                b_pred = harness.persistence_predict(train_df, val_df, TARGET)

                common = (val_df[TARGET].notna() & m_pred.notna()
                          & r_pred.notna() & b_pred.notna())
                ve = val_df.loc[val_df.index[common]]
                paired = harness.paired_intersection_scores(
                    train_df, ve,
                    {"model": m_pred.loc[ve.index],
                     "ridge_a1_lag1only": r_pred.loc[ve.index],
                     "persistence": b_pred.loc[ve.index]}, TARGET)
                pp = paired["per_predictor"]
                detail.append({
                    "combo_id": combo_id,
                    "fold_id": f["fold_id"],
                    "val_window": f"{f['val_start_year']}-{f['val_end_year']}",
                    "n_directions_intersection":
                        paired["counts"]["n_scored_directions_intersection"],
                    "model_MDA": pp["model"]["MDA"],
                    "ridge_a1_MDA": pp["ridge_a1_lag1only"]["MDA"],
                    "persistence_MDA_paired": pp["persistence"]["MDA"],
                    "delta_MDA_vs_ridge_a1":
                        pp["model"]["MDA"] - pp["ridge_a1_lag1only"]["MDA"],
                    "delta_MDA_vs_persistence":
                        pp["model"]["MDA"] - pp["persistence"]["MDA"],
                    "model_NRMSE": pp["model"]["NRMSE"],
                    "ridge_a1_NRMSE": pp["ridge_a1_lag1only"]["NRMSE"],
                    "persistence_NRMSE_paired": pp["persistence"]["NRMSE"],
                    "delta_NRMSE_vs_ridge_a1":
                        pp["model"]["NRMSE"] - pp["ridge_a1_lag1only"]["NRMSE"],
                    "delta_NRMSE_vs_persistence":
                        pp["model"]["NRMSE"] - pp["persistence"]["NRMSE"],
                })
                print(f"  {combo_id} fold {f['fold_id']}: "
                      f"dMDA vs a1 = "
                      f"{detail[-1]['delta_MDA_vs_ridge_a1']:+.4f}, "
                      f"vs pers = "
                      f"{detail[-1]['delta_MDA_vs_persistence']:+.4f}")
            d = pd.DataFrame(detail)
            delta_rows.extend(detail)
            paired_summary["per_combo"][combo_id] = {
                "mean_delta_MDA_vs_ridge_a1":
                    float(d["delta_MDA_vs_ridge_a1"].mean()),
                "folds_positive_vs_ridge_a1":
                    f"{int((d['delta_MDA_vs_ridge_a1'] > 0).sum())}/"
                    f"{len(d)}",
                "mean_delta_MDA_vs_persistence":
                    float(d["delta_MDA_vs_persistence"].mean()),
                "folds_positive_vs_persistence":
                    f"{int((d['delta_MDA_vs_persistence'] > 0).sum())}/"
                    f"{len(d)}",
                "mean_delta_NRMSE_vs_ridge_a1":
                    float(d["delta_NRMSE_vs_ridge_a1"].mean()),
                "mean_delta_NRMSE_vs_persistence":
                    float(d["delta_NRMSE_vs_persistence"].mean()),
                "folds_positive_NRMSE_vs_persistence":
                    f"{int((d['delta_NRMSE_vs_persistence'] < 0).sum())}/"
                    f"{len(d)}",
                "detail": detail,
            }
        pd.DataFrame(delta_rows)[
            ["combo_id", "fold_id", "val_window",
             "n_directions_intersection", "model_MDA", "ridge_a1_MDA",
             "persistence_MDA_paired", "delta_MDA_vs_ridge_a1",
             "delta_MDA_vs_persistence", "model_NRMSE", "ridge_a1_NRMSE",
             "persistence_NRMSE_paired",
             "delta_NRMSE_vs_ridge_a1", "delta_NRMSE_vs_persistence"]
        ].to_csv(OUT_DIR / "paired_delta_table.csv", index=False)
        save_results_json(paired_summary, OUT_DIR / "paired_deltas.json")
    else:
        # --post-only: reload saved per-run results (after a mid-run crash)
        import json as _json
        runs, chosen_by_combo, preds_by_combo = {}, {}, {}
        for combo_id, _, _ in combos:
            run_id = f"T4.1-{combo_id}"
            with open(OUT_DIR / f"{run_id}_results.json") as fh:
                runs[run_id] = _json.load(fh)

    first_run_id = f"T4.1-{combos[0][0]}"
    # ---- Combo table + DECISION RULE --------------------------------------
    KIND_SIMPLICITY = {"ridge_cv": 0, "enet": 1}
    combo_meta = {cid: (sname, kind) for cid, sname, kind in combos}
    SIZE_RANK = {"A": 0, "B": 1, "C": 2}
    board_rows = [{
        "combo": "persistence",
        **{m: fmt(runs[first_run_id]["agg"]["baseline"][m]["mean"],
                  runs[first_run_id]["agg"]["baseline"][m]["std"])
           for m in harness.METRIC_NAMES},
        "_MDA_mean":
            runs[first_run_id]["agg"]["baseline"]["MDA"]["mean"],
        "_NRMSE_mean":
            runs[first_run_id]["agg"]["baseline"]["NRMSE"]["mean"],
    }]
    for combo_id, _, _ in combos:
        m = runs[f"T4.1-{combo_id}"]["agg"]["model"]
        board_rows.append({
            "combo": combo_id,
            **{mm: fmt(m[mm]["mean"], m[mm]["std"])
               for mm in harness.METRIC_NAMES},
            "_MDA_mean": m["MDA"]["mean"],
            "_NRMSE_mean": m["NRMSE"]["mean"],
        })
    board = pd.DataFrame(board_rows)
    board.to_csv(OUT_DIR / "combo_table.csv", index=False)
    print("\nCOMBO TABLE (mean±std over the 5 canonical folds):")
    print(board[[c for c in board.columns if not c.startswith("_")]]
          .to_string(index=False))

    lin = board[board["combo"] != "persistence"]
    best_nrmse = lin["_NRMSE_mean"].min()
    eligible = lin[lin["_NRMSE_mean"] <= 1.10 * best_nrmse]
    ranked = eligible.sort_values(
        ["_MDA_mean"], ascending=False)
    # tie-break by simplicity: max MDA within 1e-4 treated as tie
    top_mda = ranked["_MDA_mean"].max()
    tied = ranked[ranked["_MDA_mean"] >= top_mda - 1e-4]
    tied = tied.assign(_simp=[
        SIZE_RANK[combo_meta[c][0].split("_")[0]]
        + KIND_SIMPLICITY[combo_meta[c][1]]
        for c in tied["combo"]])
    winner = tied.sort_values("_simp").iloc[0]["combo"]

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "decision_rule": "winner = best mean MDA subject to mean NRMSE not "
                         ">10% worse than the best mean NRMSE across the six "
                         "linear combos; ties (within 1e-4 MDA) broken by "
                         "the simpler model (Ridge < ElasticNet; smaller "
                         "feature set first)",
        "best_nrmse_among_combos": float(best_nrmse),
        "eligibility_threshold_nrmse": float(1.10 * best_nrmse),
        "eligible_combos": eligible["combo"].tolist(),
        "tied_on_MDA_within_1e-4": tied["combo"].tolist(),
        "winner_for_T4_leaderboard": winner,
        "winner_metrics": {
            m: runs[f"T4.1-{winner}"]["agg"]["model"][m]
            for m in harness.METRIC_NAMES},
        "caveats": [
            "no linear combo beats persistence MDA 0.6771 (standing bar)",
            "joint-imputation panels canonical (<=~1% optimism caveat, "
            "T3.3 hard gate)",
            "publication-lag caveat on SCSS/RMS/Census covariates does not "
            "bind here (sets A-C contain no SCSS/RMS/Census blocks)",
            "gate-scope caveat: eligibility uses GLOBAL-best NRMSE across all "
            "six combos (best=0.0010728 C-enet, threshold=0.0011801); "
            "A-enet NRMSE 0.0011897 is 1.109x global-best (excluded) but only "
            "1.052x vs persistence 0.0011305 — under a persistence-relative "
            "10% gate (0.0012436) A-enet WOULD be eligible and would be the "
            "directional winner (MDA 0.6490). Programmatic consumers must not "
            "assume the global gate is the only defensible scope.",
        ],
    }
    save_results_json(decision, OUT_DIR / "decision_rule.json")
    print(f"\nDECISION RULE -> winner: {winner}")

    # ---- Plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10.colors
    ax = axes[0]
    ax.plot(range(1, 6),
            [pf["baseline"]["MDA"]
             for pf in runs[first_run_id]["per_fold"]],
            marker="x", color="black", linewidth=2, alpha=0.7,
            label="persistence (same mask)")
    for i, (combo_id, _, _) in enumerate(combos):
        ax.plot(range(1, 6),
                [pf["model"]["MDA"]
                 for pf in runs[f"T4.1-{combo_id}"]["per_fold"]],
                marker=["o", "s"][i % 2],
                linestyle=["-", "--"][(i // 2) % 2],
                color=colors[i % 10], label=combo_id)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("fold (validation window)")
    ax.set_ylabel("boundary-anchored zero-drop MDA")
    ax.set_title("T4.1 linear family — per-fold MDA")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["persistence"] + [c[0] for c in combos]
    means = ([runs[first_run_id]["agg"]["baseline"]["MDA"]["mean"]]
             + [runs[f"T4.1-{c[0]}"]["agg"]["model"]["MDA"]["mean"]
                for c in combos])
    stds = ([runs[first_run_id]["agg"]["baseline"]["MDA"]["std"]]
            + [runs[f"T4.1-{c[0]}"]["agg"]["model"]["MDA"]["std"]
               for c in combos])
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=["gray"] + [colors[i % 10] for i in range(len(combos))])
    ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8,
                  ha="right")
    ax.set_ylabel("MDA mean ± std (5 folds)")
    ax.axhline(means[0], color="black", linewidth=1, linestyle=":")
    for i, mv in enumerate(means):
        ax.text(i, mv + 0.01, f"{mv:.4f}", ha="center", fontsize=8)
    ax.set_title("Aggregate MDA vs persistence")
    fig.tight_layout()
    plot_path = OUT_DIR / "t4_1_linear_benchmark.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[harness] saved plot -> {plot_path}")

    print("\n=== T4.1 linear-family benchmark complete ===")


if __name__ == "__main__":
    main()
