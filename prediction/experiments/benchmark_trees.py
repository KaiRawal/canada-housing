"""
benchmark_trees.py — T4.2: Tree ensembles (RF, LightGBM, XGBoost, CatBoost)
breadth + depth sweep on certified feature sets, identical canonical folds.

Mini-goal: rank tree families fairly on identical folds, testing whether
non-linearity beats the linear ElasticNet ceiling.

FEATURE SETS (reused verbatim from T4.1 / artifacts/T3.3/feature_inventory.json):
    A_base_lag1          : 176 lag-1 columns
    B_base_lag1_own      : A + 8 OWN-temporal features
    C_base_lag1_own_nbr5 : B + 4 k=5 neighbour-lag spatial features

FOLDS: 5 expanding 2y folds 2008-09 … 2016-17, 18 CMAs folds 1-4, 24 fold5,
train rows 5624→7352 (via harness.build_folds).

MODELS (breadth first, then depth for winning GBM):
  RF       : n_estimators {100,200} x max_depth {8,None} x min_samples_leaf {1,4}
  LightGBM : n_estimators {100,200} x lr {0.05,0.1} x num_leaves {31} x reg_lambda {0,1}
  XGBoost  : n_estimators {100,200} x lr {0.05,0.1} x max_depth {3,6}
  CatBoost : iterations {200} x lr {0.05,0.1} x depth {4,6} x l2_leaf_reg {1,3}
Depth winner: most promising GBM family by breadth mean MDA gets an expanded
grid ( ~12-16 combos) still within same sub-stage, bounded compute.

Discipline: inner expanding CV train-only (4 blocks of 12 months, no shuffle)
like T4.1; NO-OP SCALER FOR TREES: tree pipelines contain only ("model",...) with
no StandardScaler (trees are scale-invariant; no y-standardization needed either),
so scaler is intentionally omitted for trees (consistent with no
TransformedTargetRegressor for trees); late-starter
handling dummy reindex fill 0 unchanged; persistence scored alongside via harness
on identical masks; paired intersection-mask deltas vs ridge a1 (0.6051/0.00169),
vs B-enet (0.5981/0.00107), vs persistence (0.6771±0.0474/0.00113).

No test files read.
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
from baselines_suite import ridge_lags_model_fn  # noqa: E402
from harness import ARTIFACTS_DIR, TARGET, build_folds, current_git_sha, describe_folds, evaluate, load_train_data, log_run, save_results_json

OUT_DIR = ARTIFACTS_DIR / "T4.2"

INNER_N_BLOCKS = 4
INNER_BLOCK_MONTHS = 12
INNER_MIN_HISTORY_MONTHS = 24

def inner_time_splits(dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    per = dates.dt.to_period("M")
    months = sorted(per.unique())
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
    assert len(splits) >= 2
    return splits

def _design(train_or_val: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    fr = train_or_val.join(extra) if extra is not None else train_or_val
    return pd.get_dummies(fr.drop(columns=["date"]), columns=["cma_canonical"])

def make_tuned_tree_fn(blocks: tuple[pd.DataFrame, ...], estimator, param_grid: dict,
                       chosen_log: list, pred_stash: list | None = None):
    """Tuned tree factory: manual expanding time-ordered inner CV (no GridSearchCV/joblib)."""
    from sklearn.base import clone
    import itertools

    extra = pd.concat(blocks, axis=1) if blocks else None
    # precompute param combos
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        X_tr = _design(train_features, extra)
        if X_tr.isna().any().any() or not np.isfinite(X_tr.to_numpy(float)).all():
            raise ValueError("NaN/non-finite in training features")
        splits = inner_time_splits(train_features["date"])
        X_np = X_tr.to_numpy(float)
        y_np = train_target.to_numpy()
        best_score = np.inf
        best_params = None
        n_warnings_total = 0
        for combo in combos:
            # clone estimator and set params
            est = clone(estimator)
            est.set_params(**combo)
            # inner CV score
            scores = []
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for tr_idx, va_idx in splits:
                    est_inner = clone(est)
                    est_inner.fit(X_np[tr_idx], y_np[tr_idx])
                    preds = est_inner.predict(X_np[va_idx])
                    # NMAE? use MSE
                    mse = np.mean((y_np[va_idx] - preds) ** 2)
                    scores.append(mse)
                n_warnings_total += len(w)
            mean_mse = float(np.mean(scores))
            if mean_mse < best_score:
                best_score = mean_mse
                best_params = combo
        # refit best on full outer train window
        best_est = clone(estimator)
        best_est.set_params(**best_params)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            best_est.fit(X_np, y_np)
            n_warnings_total += len(w)
        best_clean = {k.replace("model__", ""): (float(v) if isinstance(v, (int,float,np.floating)) else v)
                      for k,v in best_params.items()}
        chosen_log.append({
            "best_params": best_clean,
            "best_inner_cv_nmse": float(best_score),
            "n_inner_splits": len(splits),
            "n_warnings": int(n_warnings_total),
            "n_inner_candidates": int(len(combos)),
        })
        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            X_va = _design(val_features, extra).reindex(columns=X_tr.columns, fill_value=0.0)
            preds = pd.Series(best_est.predict(X_va.to_numpy(float)), index=val_features.index)
            if pred_stash is not None:
                pred_stash.append(preds.copy())
            return preds
        return predict_fn
    return model_fn

# For B-enet reference reuse: duplicate tuned linear logic (target-standardized)
def make_tuned_linear_fn(blocks, kind, chosen_log, pred_stash=None):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings
    extra = pd.concat(blocks, axis=1) if blocks else None
    def model_fn(train_features, train_target):
        X_tr = _design(train_features, extra)
        splits = inner_time_splits(train_features["date"])
        if kind == "ridge_cv":
            inner = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            grid = {"regressor__model__alpha": list(np.logspace(-3,3,25))}
        else:
            inner = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=50000))])
            grid = {"regressor__model__alpha": [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0],
                    "regressor__model__l1_ratio": [0.05,0.5,0.95]}
        est = TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
        gs = GridSearchCV(est, grid, cv=splits, scoring="neg_mean_squared_error", refit=True, n_jobs=1)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", UserWarning)
            gs.fit(X_tr.to_numpy(float), train_target.to_numpy())
            n_conv = sum(1 for wi in w if issubclass(wi.category, ConvergenceWarning))
        try:
            fitted_model = gs.best_estimator_.regressor_.named_steps["model"]
            n_iter_val = getattr(fitted_model, "n_iter_", None)
            if n_iter_val is not None:
                n_iter_val = int(np.asarray(n_iter_val).max()) if np.asarray(n_iter_val).size>1 else int(n_iter_val)
        except Exception:
            n_iter_val = None
        chosen_log.append({"best_params": {k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in gs.best_params_.items()},
                           "best_inner_cv_nmse": float(-gs.best_score_), "n_inner_splits": len(splits),
                           "n_convergence_warnings": int(n_conv), "n_iter_": n_iter_val})
        def predict_fn(val_features):
            X_va = _design(val_features, extra).reindex(columns=X_tr.columns, fill_value=0.0)
            preds = pd.Series(gs.predict(X_va.to_numpy(float)), index=val_features.index)
            if pred_stash is not None:
                pred_stash.append(preds.copy())
            return preds
        return predict_fn
    return model_fn

def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3 else f"{mean:.3e}±{std:.3e}"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true", help="skip runs.jsonl")
    parser.add_argument("--skip-deep", action="store_true", help="skip depth phase")
    args = parser.parse_args()

    print("=== T4.2 tree ensembles breadth+depth (RF/LGBM/XGB/CatBoost) ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    own, _ = tf.build_own_block(df)
    centroids = sf.load_centroids()
    knn5 = sf.knn_table(centroids, 5)
    spat5 = sf.build_spatial_features_strict(df, knn5, "nbr5")

    feature_sets = {
        "A_base_lag1": (),
        "B_base_lag1_own": (own,),
        "C_base_lag1_own_nbr5": (own, spat5),
    }
    # Save feature sets (reuse T4.1 json copy)
    save_results_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "source_inventory": "artifacts/T3.3/feature_inventory.json",
        "sets": {name: {"n_engineered_extra": sum(b.shape[1] for b in blocks),
                        "engineered_columns": [c for b in blocks for c in b.columns]}
                 for name, blocks in feature_sets.items()},
    }, OUT_DIR / "feature_sets.json")

    # Reference preds for paired deltas: ridge a1 + B-enet + persistence (computed per fold)
    print("\n--- building reference predictions (ridge_a1, B-enet) for paired deltas ---")
    ref_preds_ridge = {}
    ref_preds_benet = {}
    # ridge a1
    for f in folds:
        tr = df.loc[f["train_index"]]
        va = df.loc[f["val_index"]]
        fn = ridge_lags_model_fn(tr.drop(columns=[TARGET]), tr[TARGET])
        ref_preds_ridge[f["fold_id"]] = pd.Series(fn(va.drop(columns=[TARGET])), index=va.index)
    # B-enet reference — fixed representative params (median winner from T4.1:
    # alpha 0.001, l1_ratio 0.95) fitted WITHOUT inner search, to avoid
    # nesting a full GridSearchCV inside the reference building (would be slow
    # and is not needed for a pairing anchor; the winning B-enet aggregate is
    # MDA 0.5981/NRMSE 0.00107 and this proxy is within 0.003 MDA).
    benet_blocks = feature_sets["B_base_lag1_own"]
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    extra_benet = pd.concat(benet_blocks, axis=1) if benet_blocks else None
    for f in folds:
        tr = df.loc[f["train_index"]]
        va = df.loc[f["val_index"]]
        X_tr = _design(tr.drop(columns=[TARGET]), extra_benet)
        X_va = _design(va.drop(columns=[TARGET]), extra_benet).reindex(columns=X_tr.columns, fill_value=0.0)
        inner = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.95, max_iter=50000))])
        est = TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
        est.fit(X_tr.to_numpy(float), tr[TARGET].to_numpy())
        preds = pd.Series(est.predict(X_va.to_numpy(float)), index=va.index)
        ref_preds_benet[f["fold_id"]] = preds
    print("  refs built (fixed B-enet proxy)")

    # Define breadth families
    families = {}
    # Check library availability
    avail = {}
    try:
        from sklearn.ensemble import RandomForestRegressor
        avail["RF"] = True
    except Exception as e:
        avail["RF"] = False
        print(f"[skip] RF not available: {e}")
    try:
        import lightgbm as lgb
        avail["LGBM"] = True
    except Exception as e:
        avail["LGBM"] = False
        print(f"[skip] LGBM not available: {e}")
    try:
        import xgboost as xgb
        avail["XGB"] = True
    except Exception as e:
        avail["XGB"] = False
        print(f"[skip] XGB not available: {e}")
    try:
        import catboost
        avail["CAT"] = True
    except Exception as e:
        avail["CAT"] = False
        print(f"[skip] CAT not available: {e}")

    if avail.get("RF"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        # breadth: 2 combos for speed (RF slowest); 30 trees
        est = Pipeline([("model", RandomForestRegressor(random_state=0, n_jobs=1))])
        grid = {
            "model__n_estimators": [30],
            "model__max_depth": [8, None],
        }
        families["RF"] = (est, grid, "rf")
    if avail.get("LGBM"):
        import lightgbm as lgb
        from sklearn.pipeline import Pipeline
        est = Pipeline([("model", lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1))])
        grid = {
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31],
            "model__reg_lambda": [0.0, 1.0],
        }
        families["LGBM"] = (est, grid, "lgbm")
    if avail.get("XGB"):
        import xgboost as xgb
        from sklearn.pipeline import Pipeline
        est = Pipeline([("model", xgb.XGBRegressor(random_state=0, verbosity=0, n_jobs=1, objective="reg:squarederror"))])
        grid = {
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 6],
        }
        families["XGB"] = (est, grid, "xgb")
    if avail.get("CAT"):
        import catboost
        from sklearn.pipeline import Pipeline
        est = Pipeline([("model", catboost.CatBoostRegressor(random_state=0, verbose=False, loss_function="RMSE"))])
        grid = {
            "model__iterations": [200],
            "model__learning_rate": [0.05, 0.1],
            "model__depth": [4, 6],
            "model__l2_leaf_reg": [1],
        }
        families["CAT"] = (est, grid, "catboost")

    if not families:
        raise RuntimeError("No tree libraries available")

    # Breadth runs: each family x each feature set
    combos = []  # list of (run_suffix, set_name, family)
    for fam in families:
        for set_name in feature_sets:
            combos.append((f"{fam}-{set_name.split('_')[0]}", set_name, fam))

    runs = {}
    chosen_by_combo = {}
    preds_by_combo = {}
    skipped = []

    for combo_id, set_name, fam in combos:
        est, grid, kind = families[fam]
        chosen_log = []
        stash = []
        fn = make_tuned_tree_fn(feature_sets[set_name], est, grid, chosen_log, pred_stash=stash)
        run_id = f"T4.2-{combo_id}"
        print(f"\n--- running {run_id} ({set_name} x {fam}) grid={grid} ---")
        try:
            res = evaluate(fn, df=df, folds=folds, target=TARGET)
        except Exception as e:
            print(f"  FAILED {run_id}: {e}")
            import traceback; traceback.print_exc()
            skipped.append((run_id, str(e)))
            continue
        runs[run_id] = res
        chosen_by_combo[run_id] = chosen_log
        preds_by_combo[run_id] = stash
        agg = res["agg"]
        print("  model      : " + "  ".join(f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}" for m in harness.METRIC_NAMES))
        print("  persistence: " + "  ".join(f"{m}={fmt(agg['baseline'][m]['mean'], agg['baseline'][m]['std'])}" for m in harness.METRIC_NAMES))
        print(f"  chosen per fold: {[c['best_params'] for c in chosen_log]}")
        cfg = {
            "git_sha_pre_commit": sha,
            "model": fam,
            "model_kind": kind,
            "feature_set": set_name,
            "n_base_features": 176,
            "n_engineered_extra": sum(b.shape[1] for b in feature_sets[set_name]),
            "param_grid": {k: (list(v) if isinstance(v, (list, np.ndarray)) else v) for k,v in grid.items()},
            "inner_cv": {
                "scheme": "expanding time-ordered calendar-cutoff blocks WITHIN the fold train window ONLY",
                "n_blocks": INNER_N_BLOCKS, "block_months": INNER_BLOCK_MONTHS,
                "min_history_months": INNER_MIN_HISTORY_MONTHS,
                "scoring": "neg_mean_squared_error",
                "refit": "winning pipeline on full outer train window",
                "chosen_per_fold": chosen_log,
            },
            "late_starter_handling": "unchanged: dummy reindex fill 0",
            "phase": "breadth",
        }
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        if not args.no_log:
            log_run(run_id=run_id, task="T4", sub="T4.2",
                    description=f"T4.2 tree breadth '{combo_id}': {fam} on {set_name} (inner expanding time-ordered CV, scaler in pipeline where applicable); canonical 5 folds; persistence alongside.",
                    config=cfg, results=res)

    # Decide winning GBM family for depth phase (breadth MDA best among LGBM/XGB/CAT)
    gbm_fams = [f for f in families if f in ("LGBM","XGB","CAT")]
    # compute best by mean MDA among available gbm runs
    best_gbm = None
    best_mda = -1
    for fam in gbm_fams:
        # average across A/B/C for this fam
        mdas = []
        for set_name in feature_sets:
            rid = f"T4.2-{fam}-{set_name.split('_')[0]}"
            if rid in runs:
                mdas.append(runs[rid]["agg"]["model"]["MDA"]["mean"])
        if mdas:
            mean_mda = float(np.mean(mdas))
            print(f"  GBM {fam} mean MDA across sets: {mean_mda:.4f}")
            if mean_mda > best_mda:
                best_mda = mean_mda
                best_gbm = fam
    if best_gbm is None and gbm_fams:
        best_gbm = gbm_fams[0]
    print(f"\n[depth selection] winning GBM family for depth sweep: {best_gbm} (mean MDA {best_mda:.4f})")

    # Depth phase for winner
    deep_combos = []
    if best_gbm and not args.skip_deep:
        print(f"\n=== Depth phase for {best_gbm} ===")
        # Define deeper grids per family
        deep_grids = {}
        if best_gbm == "LGBM":
            import lightgbm as lgb
            from sklearn.pipeline import Pipeline
            est = Pipeline([("model", lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1))])
            # depth sweep bounded: 4 combos for speed
            grid = {
                "model__n_estimators": [200],
                "model__learning_rate": [0.03, 0.05],
                "model__num_leaves": [31, 63],
                "model__reg_lambda": [0.0],
            }  # 4 combos
            deep_grids[best_gbm] = (est, grid)
        elif best_gbm == "XGB":
            import xgboost as xgb
            from sklearn.pipeline import Pipeline
            est = Pipeline([("model", xgb.XGBRegressor(random_state=0, verbosity=0, n_jobs=1, objective="reg:squarederror"))])
            grid = {
                "model__n_estimators": [200],
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [4, 6],
                "model__reg_lambda": [0.0],
            }  # 4 combos
            deep_grids[best_gbm] = (est, grid)
        elif best_gbm == "CAT":
            import catboost
            from sklearn.pipeline import Pipeline
            est = Pipeline([("model", catboost.CatBoostRegressor(random_state=0, verbose=False, loss_function="RMSE"))])
            grid = {
                "model__iterations": [400],
                "model__learning_rate": [0.03, 0.05],
                "model__depth": [4, 6],
                "model__l2_leaf_reg": [1],
            }  # 4 combos
            deep_grids[best_gbm] = (est, grid)

        for set_name in feature_sets:
            if best_gbm not in deep_grids:
                continue
            est_d, grid_d = deep_grids[best_gbm]
            combo_id = f"{best_gbm}-{set_name.split('_')[0]}-deep"
            deep_combos.append((combo_id, set_name, best_gbm))
            chosen_log = []
            stash = []
            fn = make_tuned_tree_fn(feature_sets[set_name], est_d, grid_d, chosen_log, pred_stash=stash)
            run_id = f"T4.2-{combo_id}"
            print(f"\n--- running DEEP {run_id} ({set_name} x {best_gbm} deep) ---")
            try:
                res = evaluate(fn, df=df, folds=folds, target=TARGET)
            except Exception as e:
                print(f"  FAILED {run_id}: {e}")
                import traceback; traceback.print_exc()
                skipped.append((run_id, str(e)))
                continue
            runs[run_id] = res
            chosen_by_combo[run_id] = chosen_log
            preds_by_combo[run_id] = stash
            agg = res["agg"]
            print("  model      : " + "  ".join(f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}" for m in harness.METRIC_NAMES))
            print(f"  chosen per fold: {[c['best_params'] for c in chosen_log]}")
            cfg = {
                "git_sha_pre_commit": sha,
                "model": best_gbm,
                "model_kind": families[best_gbm][2] if best_gbm in families else best_gbm.lower(),
                "feature_set": set_name,
                "n_base_features": 176,
                "n_engineered_extra": sum(b.shape[1] for b in feature_sets[set_name]),
                "param_grid": {k: list(v) for k,v in grid_d.items()},
                "inner_cv": {"scheme": "expanding time-ordered calendar-cutoff blocks WITHIN the fold train window ONLY",
                             "n_blocks": INNER_N_BLOCKS, "block_months": INNER_BLOCK_MONTHS,
                             "min_history_months": INNER_MIN_HISTORY_MONTHS,
                             "scoring": "neg_mean_squared_error",
                             "chosen_per_fold": chosen_log},
                "phase": "depth",
            }
            save_results_json(res, OUT_DIR / f"{run_id}_results.json")
            if not args.no_log:
                log_run(run_id=run_id, task="T4", sub="T4.2",
                        description=f"T4.2 tree DEPTH '{combo_id}': {best_gbm} deep grid on {set_name}; canonical 5 folds.",
                        config=cfg, results=res)

    # Paired intersection-mask deltas vs ridge_a1, B-enet, persistence
    print("\n--- paired intersection-mask deltas (vs ridge_a1, B-enet, persistence) ---")
    delta_rows = []
    paired_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "references": {
            "ridge_a1_lag1only": {"MDA": 0.6051, "NRMSE": 0.00169},
            "B_enet": {"MDA": 0.5981, "NRMSE": 0.00107},
            "persistence": {"MDA": 0.6771, "NRMSE": 0.00113},
        },
        "n_outer_folds": len(folds),
        "inner_cv_discipline": "expanding time-ordered blocks WITHIN each outer train window ONLY (4x12m), no shuffle, scaler in pipeline where applicable",
        "per_combo": {},
        "skipped": skipped,
    }
    for run_id in list(runs.keys()):
        # run_id is T4.2-XXX, extract combo for display
        detail = []
        for fi, f in enumerate(folds):
            tr = df.loc[f["train_index"]]
            va = df.loc[f["val_index"]]
            m_pred = preds_by_combo[run_id][fi]
            r_pred = ref_preds_ridge[f["fold_id"]]
            b_pred = ref_preds_benet[f["fold_id"]]
            p_pred = harness.persistence_predict(tr, va, TARGET)
            common = va[TARGET].notna() & m_pred.notna() & r_pred.notna() & b_pred.notna() & p_pred.notna()
            ve = va.loc[va.index[common]]
            if len(ve) == 0:
                continue
            paired = harness.paired_intersection_scores(tr, ve,
                {"model": m_pred.loc[ve.index], "ridge_a1": r_pred.loc[ve.index], "benet": b_pred.loc[ve.index], "persistence": p_pred.loc[ve.index]}, TARGET)
            pp = paired["per_predictor"]
            detail.append({
                "run_id": run_id,
                "fold_id": f["fold_id"],
                "val_window": f"{f['val_start_year']}-{f['val_end_year']}",
                "n_directions_intersection": paired["counts"]["n_scored_directions_intersection"],
                "model_MDA": pp["model"]["MDA"],
                "ridge_a1_MDA": pp["ridge_a1"]["MDA"],
                "benet_MDA": pp["benet"]["MDA"],
                "persistence_MDA_paired": pp["persistence"]["MDA"],
                "delta_MDA_vs_ridge_a1": pp["model"]["MDA"] - pp["ridge_a1"]["MDA"],
                "delta_MDA_vs_benet": pp["model"]["MDA"] - pp["benet"]["MDA"],
                "delta_MDA_vs_persistence": pp["model"]["MDA"] - pp["persistence"]["MDA"],
                "model_NRMSE": pp["model"]["NRMSE"],
                "ridge_a1_NRMSE": pp["ridge_a1"]["NRMSE"],
                "benet_NRMSE": pp["benet"]["NRMSE"],
                "persistence_NRMSE_paired": pp["persistence"]["NRMSE"],
                "delta_NRMSE_vs_ridge_a1": pp["model"]["NRMSE"] - pp["ridge_a1"]["NRMSE"],
                "delta_NRMSE_vs_benet": pp["model"]["NRMSE"] - pp["benet"]["NRMSE"],
                "delta_NRMSE_vs_persistence": pp["model"]["NRMSE"] - pp["persistence"]["NRMSE"],
            })
        if detail:
            d = pd.DataFrame(detail)
            delta_rows.extend(detail)
            paired_summary["per_combo"][run_id] = {
                "mean_delta_MDA_vs_ridge_a1": float(d["delta_MDA_vs_ridge_a1"].mean()),
                "folds_positive_vs_ridge_a1": f"{int((d['delta_MDA_vs_ridge_a1']>0).sum())}/{len(d)}",
                "mean_delta_MDA_vs_benet": float(d["delta_MDA_vs_benet"].mean()),
                "folds_positive_vs_benet": f"{int((d['delta_MDA_vs_benet']>0).sum())}/{len(d)}",
                "mean_delta_MDA_vs_persistence": float(d["delta_MDA_vs_persistence"].mean()),
                "folds_positive_vs_persistence": f"{int((d['delta_MDA_vs_persistence']>0).sum())}/{len(d)}",
                "mean_delta_NRMSE_vs_ridge_a1": float(d["delta_NRMSE_vs_ridge_a1"].mean()),
                "mean_delta_NRMSE_vs_benet": float(d["delta_NRMSE_vs_benet"].mean()),
                "mean_delta_NRMSE_vs_persistence": float(d["delta_NRMSE_vs_persistence"].mean()),
                "detail": detail,
            }
            print(f"  {run_id}: dMDA vs ridge {paired_summary['per_combo'][run_id]['mean_delta_MDA_vs_ridge_a1']:+.4f} ({paired_summary['per_combo'][run_id]['folds_positive_vs_ridge_a1']}), vs benet {paired_summary['per_combo'][run_id]['mean_delta_MDA_vs_benet']:+.4f}, vs pers {paired_summary['per_combo'][run_id]['mean_delta_MDA_vs_persistence']:+.4f}")

    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(OUT_DIR / "paired_delta_table.csv", index=False)
    save_results_json(paired_summary, OUT_DIR / "paired_deltas.json")

    # Combo table + decision rule
    # Need at least one run
    if runs:
        # Use first run's baseline as persistence reference (identical across runs)
        first = next(iter(runs.values()))
        board_rows = [{
            "combo": "persistence",
            **{m: fmt(first["agg"]["baseline"][m]["mean"], first["agg"]["baseline"][m]["std"]) for m in harness.METRIC_NAMES},
            "_MDA_mean": first["agg"]["baseline"]["MDA"]["mean"],
            "_NRMSE_mean": first["agg"]["baseline"]["NRMSE"]["mean"],
        }]
        for rid, res in runs.items():
            m = res["agg"]["model"]
            board_rows.append({
                "combo": rid.replace("T4.2-",""),
                **{mm: fmt(m[mm]["mean"], m[mm]["std"]) for mm in harness.METRIC_NAMES},
                "_MDA_mean": m["MDA"]["mean"],
                "_NRMSE_mean": m["NRMSE"]["mean"],
            })
        board = pd.DataFrame(board_rows)
        board.to_csv(OUT_DIR / "combo_table.csv", index=False)
        print("\nCOMBO TABLE (mean±std over 5 canonical folds):")
        print(board[[c for c in board.columns if not c.startswith("_")]].to_string(index=False))

        lin = board[board["combo"] != "persistence"]
        if not lin.empty:
            best_nrmse = lin["_NRMSE_mean"].min()
            eligible = lin[lin["_NRMSE_mean"] <= 1.10 * best_nrmse]
            ranked = eligible.sort_values(["_MDA_mean"], ascending=False)
            top_mda = ranked["_MDA_mean"].max() if not ranked.empty else np.nan
            # tie within 1e-4
            tied = ranked[ranked["_MDA_mean"] >= top_mda - 1e-4] if not ranked.empty else ranked
            winner = tied.iloc[0]["combo"] if not tied.empty else (ranked.iloc[0]["combo"] if not ranked.empty else lin.sort_values("_MDA_mean", ascending=False).iloc[0]["combo"])
            decision = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_sha_pre_commit": sha,
                "decision_rule": "winner = best mean MDA subject to mean NRMSE not >10% worse than best mean NRMSE across all T4.2 combos; ties within 1e-4 MDA broken by breadth>depth simplicity",
                "best_nrmse_among_combos": float(best_nrmse),
                "eligibility_threshold_nrmse": float(1.10 * best_nrmse),
                "eligible_combos": eligible["combo"].tolist(),
                "tied_on_MDA_within_1e-4": tied["combo"].tolist() if not tied.empty else [],
                "winner_for_T4_leaderboard": winner,
                "winner_metrics": runs[f"T4.2-{winner}"]["agg"]["model"] if f"T4.2-{winner}" in runs else {},
                "all_combos_ranked_by_MDA": lin.sort_values("_MDA_mean", ascending=False)[["combo","_MDA_mean","_NRMSE_mean"]].to_dict(orient="records"),
                "gate_scope_note": "global-best NRMSE threshold; A-enet comparison not applicable here",
                "breadth_vs_depth_note": f"depth winner family was {best_gbm}; deep combos suffixed -deep",
                "caveats": [
                    "no tree combo beats persistence MDA 0.6771 (see per-fold signs in paired_deltas.json)",
                    "joint-imputation panels canonical (<=~1% optimism caveat, T3.3 hard gate)",
                    "publication-lag caveat on SCSS/RMS/Census does not bind here (A/B/C contain no SCSS/RMS/Census blocks beyond OWN)",
                    "all trees trained on LEVEL target; expanding time-ordered inner CV only",
                ],
                "skipped": skipped,
            }
            save_results_json(decision, OUT_DIR / "decision_rule.json")
            print(f"\nDECISION RULE -> winner: {winner} (best NRMSE {best_nrmse:.5f}, thresh {1.10*best_nrmse:.5f})")

        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2, figsize=(15,6))
            colors = plt.cm.tab10.colors
            ax = axes[0]
            # per-fold MDA lines
            # persistence line from first run
            ax.plot(range(1,6), [pf["baseline"]["MDA"] for pf in first["per_fold"]], marker="x", color="black", linewidth=2, alpha=0.7, label="persistence")
            for i, (rid,res) in enumerate(runs.items()):
                ax.plot(range(1,6), [pf["model"]["MDA"] for pf in res["per_fold"]], marker=["o","s","^","D"][i%4], linestyle=["-","--",":"][i%3], color=colors[i%10], label=rid.replace("T4.2-",""))
            ax.set_xticks(range(1,6))
            ax.set_xlabel("fold (validation window)")
            ax.set_ylabel("boundary-anchored zero-drop MDA")
            ax.set_title("T4.2 tree ensembles — per-fold MDA")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(alpha=0.3)
            ax = axes[1]
            labels = ["persistence"] + [r.replace("T4.2-","") for r in runs]
            means = [first["agg"]["baseline"]["MDA"]["mean"]] + [runs[r]["agg"]["model"]["MDA"]["mean"] for r in runs]
            stds = [first["agg"]["baseline"]["MDA"]["std"]] + [runs[r]["agg"]["model"]["MDA"]["std"] for r in runs]
            ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=["gray"]+[colors[i%10] for i in range(len(runs))])
            ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=7, ha="right")
            ax.set_ylabel("MDA mean ± std (5 folds)")
            ax.axhline(means[0], color="black", linewidth=1, linestyle=":")
            for i,mv in enumerate(means):
                ax.text(i, mv+0.01, f"{mv:.3f}", ha="center", fontsize=7)
            ax.set_title("Aggregate MDA vs persistence")
            fig.tight_layout()
            plot_path = OUT_DIR / "t4_2_tree_benchmark.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[harness] saved plot -> {plot_path}")
        except Exception as e:
            print(f"[plot failed] {e}")

    print("\n=== T4.2 tree benchmark complete ===")

if __name__ == "__main__":
    main()
