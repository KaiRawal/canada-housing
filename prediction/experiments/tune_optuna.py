#!/usr/bin/env python3
"""
T5.1 — Optuna search under the same CV folds; every trial logged to runs.jsonl (pruned scope: A/B only)

Mini-goal: squeeze validated performance out of shortlisted B-enet (and A-enet as adjudicated runner-up)
via Optuna on same 5 expanding folds.

Search space (pruned, not heavy): ElasticNet alpha loguniform 1e-4..1.0,
l1_ratio ∈ {0.1,0.3,0.5,0.7,0.9,0.95}  (selection fixed to 'cyclic' for determinism).

CV: SAME 5 outer folds via harness (2008-09 … 2016-17, expanding, grouped by CMA).
We use DIRECT outer CV Optuna (simpler, compute-bounded) — each trial is scored on the
5 outer folds through harness.evaluate (identical folds/masks as T1–T4). This is documented
explicitly: outer CV tuning is optimistic relative to a nested inner CV (tuning sees the
same outer folds it is scored on), but the ranking and gate-eligibility relative to the
codified 1.10× global-best threshold remains apples-to-apples because every T4 combo was
also scored on the same outer folds. An inner expanding 4×12m scheme would be the strictly
leakage-free nested alternative; we state this caveat and note that the headroom estimate
is therefore a conditional upper bound.

Trials: ~70 for B_base_lag1_OWN-subset + ~30 for A_base_lag1 (total 100, bounded).
Every trial appends to runs.jsonl with config:{trial_id, params, cv:outer, feature_set},
cv_metrics per fold + agg, baseline_metrics alongside. Fixed seed 0.

Winner selection: best mean MDA subject to ≤1.10× global-best NRMSE gate
(global-best = min across T4 + T5 trials) — plus unconstrained winner reported.

No test-split file is read anywhere.
"""
import argparse
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
from harness import ARTIFACTS_DIR, TARGET, build_folds, current_git_sha, load_train_data, save_results_json  # noqa: E402

import optuna  # noqa: E402
import sklearn  # noqa: E402

OUT_DIR = ARTIFACTS_DIR / "T5.1"

# Search space
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
ALPHA_LOW = 1e-4
ALPHA_HIGH = 1.0

# Canonical constants (from leaderboard)
PERSIST_MDA = 0.6771279882839918
PERSIST_NRMSE = 0.0011305311297628712
CODIFIED_GLOBAL_BEST_NRMSE = 0.0010752020172069055
C_ENET_NRMSE = 0.0010727975947863121
GATE_RATIO = 1.10

OWN_SUBSET_COLS = ["tot_d12", "tot_lag12", "tot_lag24", "tot_dlog1"]

def _design(train_or_val: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    fr = train_or_val.join(extra) if extra is not None else train_or_val
    return pd.get_dummies(fr.drop(columns=["date"]), columns=["cma_canonical"])


def make_fixed_enet_fn(extra: pd.DataFrame | None, alpha: float, l1_ratio: float):
    """Return a harness model_fn with FIXED ElasticNet hyperparams."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    extra_local = extra

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        X_tr = _design(train_features, extra_local)
        if X_tr.isna().any().any() or not np.isfinite(X_tr.to_numpy(float)).all():
            raise ValueError("NaN/non-finite in training features")
        inner = Pipeline([("scaler", StandardScaler()),
                          ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000))])
        est = TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
        est.fit(X_tr.to_numpy(float), train_target.to_numpy())

        X_cols = X_tr.columns

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            X_va = _design(val_features, extra_local).reindex(columns=X_cols, fill_value=0.0)
            preds = pd.Series(est.predict(X_va.to_numpy(float)), index=val_features.index)
            return preds

        return predict_fn

    return model_fn


def evaluate_fixed(alpha, l1_ratio, extra, df, folds):
    fn = make_fixed_enet_fn(extra, alpha, l1_ratio)
    return harness.evaluate(fn, df=df, folds=folds, target=TARGET)


def run_optuna_for_set(set_name: str, extra: pd.DataFrame | None, df, folds, n_trials: int, seed: int, trial_log: list, study_name: str):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)

    sha = current_git_sha()
    versions = {
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
    }
    try:
        import lightgbm
        versions["lightgbm"] = lightgbm.__version__
    except Exception:
        versions["lightgbm"] = "not_installed"

    def objective(trial: optuna.Trial):
        alpha = trial.suggest_float("alpha", ALPHA_LOW, ALPHA_HIGH, log=True)
        l1_ratio = trial.suggest_categorical("l1_ratio", L1_RATIOS)
        # Fixed selection cyclic for determinism
        res = evaluate_fixed(alpha, l1_ratio, extra, df, folds)
        # Log every trial to runs.jsonl immediately
        agg = res["agg"]
        run_id = f"T5.1-trial-{set_name}-{trial.number:03d}"
        cfg = {
            "git_sha_pre_commit": sha,
            "trial_id": trial.number,
            "trial_params": {"alpha": float(alpha), "l1_ratio": float(l1_ratio)},
            "params": {"alpha": float(alpha), "l1_ratio": float(l1_ratio)},
            "feature_set": set_name,
            "feature_set_name": set_name,
            "n_base_features": 176,
            "n_engineered_extra": 0 if extra is None else int(extra.shape[1]),
            "engineered_columns": [] if extra is None else list(extra.columns),
            "cv": "outer",
            "inner_or_outer": "outer",
            "study": study_name,
            "versions": versions,
            "n_folds": 5,
            "target": TARGET,
            "anchor_year": 2017,
            "search_space": {"alpha": f"loguniform {ALPHA_LOW}..{ALPHA_HIGH}", "l1_ratio": L1_RATIOS},
        }
        # Append to runs.jsonl via harness
        harness.log_run(
            run_id=run_id, task="T5", sub="T5.1",
            description=f"Optuna trial {trial.number} for {set_name}: ElasticNet alpha={alpha:.6g} l1_ratio={l1_ratio} (outer CV, pruned scope)",
            config=cfg,
            results=res,
        )
        # Store for artifacts
        trial_log.append({
            "trial_number": trial.number,
            "feature_set": set_name,
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "MDA_mean": agg["model"]["MDA"]["mean"],
            "MDA_std": agg["model"]["MDA"]["std"],
            "NRMSE_mean": agg["model"]["NRMSE"]["mean"],
            "NRMSE_std": agg["model"]["NRMSE"]["std"],
            "NMSE_mean": agg["model"]["NMSE"]["mean"],
            "NMAE_mean": agg["model"]["NMAE"]["mean"],
            "baseline_MDA_mean": agg["baseline"]["MDA"]["mean"],
            "baseline_NRMSE_mean": agg["baseline"]["NRMSE"]["mean"],
            "per_fold": res["per_fold"],
            "run_id": run_id,
        })
        # Objective: minimize mean NRMSE (primary goal is point-error headroom)
        # Use NRMSE mean
        return float(agg["model"]["NRMSE"]["mean"])

    study.optimize(objective, n_trials=n_trials)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials-B", type=int, default=70, help="Optuna trials for B subset")
    parser.add_argument("--n-trials-A", type=int, default=30, help="Optuna trials for A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-log", action="store_true", help="skip runs.jsonl logging (debug)")
    args = parser.parse_args()

    print("=== T5.1 Optuna tuning (pruned A/B, outer CV) ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print(harness.describe_folds(folds))
    sha = current_git_sha()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build OWN block and pruned subset
    own_full, own_stats = tf.build_own_block(df)
    # Validate subset columns exist
    missing = set(OWN_SUBSET_COLS) - set(own_full.columns)
    if missing:
        raise ValueError(f"OWN subset missing cols {missing}, available {list(own_full.columns)}")
    own_subset = own_full[OWN_SUBSET_COLS].copy()
    print(f"[T5.1] OWN full {own_full.shape[1]} cols -> pruned subset {own_subset.shape[1]} cols {OWN_SUBSET_COLS}")

    # Feature set mapping
    feature_sets = {
        "A_base_lag1": None,
        "B_base_lag1_OWN-subset": own_subset,
    }
    # Save feature_sets provenance
    save_results_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "feature_sets": {
            "A_base_lag1": {"n_engineered_extra": 0, "engineered_columns": []},
            "B_base_lag1_OWN-subset": {"n_engineered_extra": len(OWN_SUBSET_COLS), "engineered_columns": OWN_SUBSET_COLS},
        },
        "source": "tf.build_own_block pruned to tot_d12/lag12/24/dlog1 (4 cols) per 2026-08-28 amendment",
        "search_space": {"alpha": f"loguniform {ALPHA_LOW}..{ALPHA_HIGH}", "l1_ratio": L1_RATIOS, "selection": "cyclic (fixed)"},
        "cv": "outer 5 expanding folds 2008-09…2016-17 (same as T1–T4); inner 4×12m would be nested alternative (caveated)",
        "trials": {"B": args.n_trials_B, "A": args.n_trials_A, "total": args.n_trials_B + args.n_trials_A, "seed": args.seed},
    }, OUT_DIR / "feature_sets.json")

    # Monkey-patch harness.log_run to no-op if --no-log
    if args.no_log:
        orig_log = harness.log_run
        def _nolog(*a, **kw):
            print(f"[no-log] would log {kw.get('run_id', a[0] if a else '?')}")
        harness.log_run = _nolog

    all_trials = []

    print(f"\n--- Optuna study B ({args.n_trials_B} trials, seed {args.seed}) ---")
    study_B = run_optuna_for_set("B_base_lag1_OWN-subset", own_subset, df, folds, args.n_trials_B, seed=args.seed, trial_log=all_trials, study_name="T5.1-B-enet")
    print(f"B study best trial #{study_B.best_trial.number}: value={study_B.best_value:.6g} params={study_B.best_params}")

    print(f"\n--- Optuna study A ({args.n_trials_A} trials, seed {args.seed+1}) ---")
    study_A = run_optuna_for_set("A_base_lag1", None, df, folds, args.n_trials_A, seed=args.seed+1, trial_log=all_trials, study_name="T5.1-A-enet")
    print(f"A study best trial #{study_A.best_trial.number}: value={study_A.best_value:.6g} params={study_A.best_params}")

    # Restore if patched
    if args.no_log:
        harness.log_run = orig_log

    # Build trial_table.csv
    df_trials = pd.DataFrame([{
        "trial_id": f"T5.1-trial-{r['feature_set']}-{r['trial_number']:03d}",
        "run_id": r["run_id"],
        "feature_set": r["feature_set"],
        "trial_number": r["trial_number"],
        "alpha": r["alpha"],
        "l1_ratio": r["l1_ratio"],
        "MDA_mean": r["MDA_mean"],
        "MDA_std": r["MDA_std"],
        "NRMSE_mean": r["NRMSE_mean"],
        "NRMSE_std": r["NRMSE_std"],
        "NMSE_mean": r["NMSE_mean"],
        "NMAE_mean": r["NMAE_mean"],
        "baseline_MDA_mean": r["baseline_MDA_mean"],
        "baseline_NRMSE_mean": r["baseline_NRMSE_mean"],
        "delta_MDA_vs_persistence": r["MDA_mean"] - PERSIST_MDA,
        "NRMSE_ratio_vs_codified_best": r["NRMSE_mean"] / CODIFIED_GLOBAL_BEST_NRMSE,
        "NRMSE_ratio_vs_persistence": r["NRMSE_mean"] / PERSIST_NRMSE,
    } for r in all_trials])
    df_trials.to_csv(OUT_DIR / "trial_table.csv", index=False)
    print(f"[T5.1] saved trial_table.csv with {len(df_trials)} trials")

    # Determine global-best NRMSE across T4 + T5
    # T4 global best known constants; but compute actual min across trials too
    t5_best_nrmse = df_trials["NRMSE_mean"].min()
    # Codified global best from T4.1 B-enet
    global_best_t4 = min(CODIFIED_GLOBAL_BEST_NRMSE, C_ENET_NRMSE)
    global_best_all = float(min(global_best_t4, t5_best_nrmse))
    # Find which id holds global best
    if t5_best_nrmse < global_best_t4:
        best_row = df_trials.loc[df_trials["NRMSE_mean"].idxmin()]
        global_best_id = best_row["run_id"]
    else:
        # Use C-enet if it is smaller than codified else B-enet
        global_best_id = "T4.1-C-enet" if C_ENET_NRMSE < CODIFIED_GLOBAL_BEST_NRMSE else "T4.1-B-enet"
    gate_threshold = global_best_all * GATE_RATIO

    # Winner selection: best MDA subject to ≤1.10× global-best NRMSE
    eligible = df_trials[df_trials["NRMSE_mean"] <= gate_threshold].copy()
    # Unconstrained best MDA
    unconstrained_best = df_trials.loc[df_trials["MDA_mean"].idxmax()]
    if not eligible.empty:
        eligible_sorted = eligible.sort_values("MDA_mean", ascending=False)
        gated_best = eligible_sorted.iloc[0]
        gated_best_id = gated_best["run_id"]
    else:
        gated_best = None
        gated_best_id = None

    # For B subset specifically, find its gated best
    b_trials = df_trials[df_trials["feature_set"] == "B_base_lag1_OWN-subset"]
    b_eligible = b_trials[b_trials["NRMSE_mean"] <= gate_threshold]
    if not b_eligible.empty:
        b_gated_best = b_eligible.loc[b_eligible["MDA_mean"].idxmax()]
    else:
        b_gated_best = None
    b_unconstrained = b_trials.loc[b_trials["MDA_mean"].idxmax()] if not b_trials.empty else None
    b_nrmse_best = b_trials.loc[b_trials["NRMSE_mean"].idxmin()] if not b_trials.empty else None

    # A gated best
    a_trials = df_trials[df_trials["feature_set"] == "A_base_lag1"]
    a_eligible = a_trials[a_trials["NRMSE_mean"] <= gate_threshold]
    a_gated_best = a_eligible.loc[a_eligible["MDA_mean"].idxmax()] if not a_eligible.empty else None

    print(f"\nGlobal best NRMSE: {global_best_all:.6g} ({global_best_id}) threshold {gate_threshold:.6g}")
    print(f"Unconstrained best MDA: {unconstrained_best['run_id']} MDA={unconstrained_best['MDA_mean']:.4f} NRMSE={unconstrained_best['NRMSE_mean']:.6g} ratio={unconstrained_best['NRMSE_ratio_vs_codified_best']:.3f}")
    if gated_best is not None:
        print(f"Gated best (≤1.10× global-best): {gated_best_id} MDA={gated_best['MDA_mean']:.4f} NRMSE={gated_best['NRMSE_mean']:.6g}")
    else:
        print("No gated eligible trials (≤1.10× global-best) — gate yields empty")
    if b_gated_best is not None:
        print(f"B gated best: {b_gated_best['run_id']} MDA={b_gated_best['MDA_mean']:.4f} NRMSE={b_gated_best['NRMSE_mean']:.6g}")
    if b_nrmse_best is not None:
        print(f"B NRMSE best: {b_nrmse_best['run_id']} NRMSE={b_nrmse_best['NRMSE_mean']:.6g} alpha={b_nrmse_best['alpha']:.6g} l1={b_nrmse_best['l1_ratio']}")

    # Headroom to 20% NRMSE goal (0.00113 -> 0.00090)
    persistence_nrmse = PERSIST_NRMSE
    goal_nrmse = 0.00090
    b_current_best_nrmse = b_nrmse_best["NRMSE_mean"] if b_nrmse_best is not None else np.nan
    headroom_closed = (persistence_nrmse - b_current_best_nrmse) / (persistence_nrmse - goal_nrmse) if not np.isnan(b_current_best_nrmse) else np.nan
    print(f"Headroom to 0.00090 goal: persistence {persistence_nrmse:.6g} -> best B {b_current_best_nrmse:.6g} (goal {goal_nrmse:.6g}) closed {headroom_closed*100:.1f}% of 20% gap; need {0.00107:.6g} ->0.00090 is -15.9% more")

    # Log winner as T5.1-B-enet-optuna-best (and A if exists)
    # Choose B gated best as primary winner if exists else B NRMSE best, else unconstrained
    if b_gated_best is not None:
        winner_row = b_gated_best
        winner_feature_set = "B_base_lag1_OWN-subset"
        winner_alpha = float(winner_row["alpha"])
        winner_l1 = float(winner_row["l1_ratio"])
    elif b_nrmse_best is not None:
        winner_row = b_nrmse_best
        winner_feature_set = "B_base_lag1_OWN-subset"
        winner_alpha = float(winner_row["alpha"])
        winner_l1 = float(winner_row["l1_ratio"])
    else:
        winner_row = unconstrained_best
        winner_feature_set = winner_row["feature_set"]
        winner_alpha = float(winner_row["alpha"])
        winner_l1 = float(winner_row["l1_ratio"])

    # Re-evaluate winner to get full per_fold for winner log (reuse stored)
    # Find the trial record
    winner_trial_num = int(winner_row["trial_number"])
    # Need to map trial_number to stored per_fold — for B, trial numbers 0..B-1, for A offset?
    # Instead look up in all_trials
    winner_trial_rec = next(r for r in all_trials if r["run_id"] == winner_row["run_id"])
    # Build results dict for winner log
    # We need to reconstruct harness results structure for winner
    # We have per_fold already, but need agg. Recompute from per_fold
    per_fold_winner = winner_trial_rec["per_fold"]
    # Compute agg via harness._aggregate
    agg_winner = harness._aggregate(per_fold_winner)
    results_winner = {
        "folds": [{"fold_id": f["fold_id"]} for f in folds],
        "per_fold": per_fold_winner,
        "agg": agg_winner,
        "config": {"n_folds": 5, "target": TARGET, "anchor_year": 2017},
    }
    extra_winner = own_subset if winner_feature_set == "B_base_lag1_OWN-subset" else None
    versions_winner = {"sklearn": sklearn.__version__, "optuna": optuna.__version__}
    try:
        import lightgbm
        versions_winner["lightgbm"] = lightgbm.__version__
    except Exception:
        versions_winner["lightgbm"] = "not_installed"

    winner_run_id = "T5.1-B-enet-optuna-best"
    # If winner is from A set, use A label — but primary is B, so keep B label even if A wins? Plan says B-enet-optuna-best
    # We'll log B winner always, and additionally A winner if distinct
    if not args.no_log:
        harness.log_run(
            run_id=winner_run_id, task="T5", sub="T5.1",
            description=f"T5.1 winner B-enet Optuna: alpha={winner_alpha:.6g} l1_ratio={winner_l1} on {winner_feature_set}; gated best MDA={winner_row['MDA_mean']:.4f} NRMSE={winner_row['NRMSE_mean']:.6g} (outer CV)",
            config={
                "git_sha_pre_commit": sha,
                "model": "elasticnet_optuna",
                "feature_set": winner_feature_set,
                "feature_set_name": winner_feature_set,
                "n_base_features": 176,
                "n_engineered_extra": 0 if extra_winner is None else int(extra_winner.shape[1]),
                "engineered_columns": OWN_SUBSET_COLS if winner_feature_set.startswith("B") else [],
                "best_params": {"alpha": winner_alpha, "l1_ratio": winner_l1},
                "params": {"alpha": winner_alpha, "l1_ratio": winner_l1},
                "winner_selection": "best mean MDA subject to ≤1.10× global-best NRMSE gate" + (f" (gated {winner_row['run_id']})" if b_gated_best is not None else " (no gated eligible, fell back to NRMSE best)"),
                "global_best_NRMSE": global_best_all,
                "gate_threshold": gate_threshold,
                "unconstrained_best_id": unconstrained_best["run_id"],
                "gated_best_id": gated_best_id,
                "versions": versions_winner,
                "n_folds": 5,
                "target": TARGET,
                "anchor_year": 2017,
                "search_space": {"alpha": f"loguniform {ALPHA_LOW}..{ALPHA_HIGH}", "l1_ratio": L1_RATIOS},
                "cv": "outer",
                "inner_or_outer": "outer",
            },
            results=results_winner,
        )
        # Also log A winner if B winner was not from A and A has a distinct gated best
        if a_gated_best is not None and a_gated_best["run_id"] != winner_row["run_id"]:
            a_rec = next(r for r in all_trials if r["run_id"] == a_gated_best["run_id"])
            agg_a = harness._aggregate(a_rec["per_fold"])
            results_a = {"folds": [{"fold_id": f["fold_id"]} for f in folds], "per_fold": a_rec["per_fold"], "agg": agg_a, "config": {"n_folds": 5, "target": TARGET, "anchor_year": 2017}}
            harness.log_run(
                run_id="T5.1-A-enet-optuna-best", task="T5", sub="T5.1",
                description=f"T5.1 winner A-enet Optuna: alpha={a_gated_best['alpha']:.6g} l1_ratio={a_gated_best['l1_ratio']} (gated best for A)",
                config={
                    "git_sha_pre_commit": sha,
                    "model": "elasticnet_optuna",
                    "feature_set": "A_base_lag1",
                    "feature_set_name": "A_base_lag1",
                    "n_base_features": 176,
                    "n_engineered_extra": 0,
                    "engineered_columns": [],
                    "best_params": {"alpha": float(a_gated_best["alpha"]), "l1_ratio": float(a_gated_best["l1_ratio"])},
                    "params": {"alpha": float(a_gated_best["alpha"]), "l1_ratio": float(a_gated_best["l1_ratio"])},
                    "winner_selection": "best mean MDA subject to ≤1.10× global-best NRMSE gate (A subset)",
                    "global_best_NRMSE": global_best_all,
                    "gate_threshold": gate_threshold,
                    "versions": versions_winner,
                },
                results=results_a,
            )

    # Artifacts
    # optuna_study.json
    def study_to_dict(study, set_name):
        return {
            "study_name": study.study_name,
            "feature_set": set_name,
            "n_trials": len(study.trials),
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_value,
                "params": study.best_params,
            } if study.best_trial else None,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in study.trials
            ],
        }

    optuna_study = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "cv": "outer 5 expanding folds (direct outer Optuna; inner 4x12m nested would be leakage-free alternative, caveated)",
        "search_space": {"alpha": f"loguniform {ALPHA_LOW}..{ALPHA_HIGH}", "l1_ratio": L1_RATIOS},
        "seed": args.seed,
        "studies": {
            "B_base_lag1_OWN-subset": study_to_dict(study_B, "B_base_lag1_OWN-subset"),
            "A_base_lag1": study_to_dict(study_A, "A_base_lag1"),
        },
        "versions": versions_winner,
    }
    save_results_json(optuna_study, OUT_DIR / "optuna_study.json")

    # best_params.json
    best_params = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "global_best_NRMSE": global_best_all,
        "global_best_id": global_best_id,
        "gate_threshold": gate_threshold,
        "gate_rule": "winner = best mean MDA subject to mean NRMSE ≤1.10× global-best NRMSE (global-best = min across T4 + T5 trials)",
        "winner_B": {
            "run_id": winner_row["run_id"],
            "feature_set": winner_feature_set,
            "params": {"alpha": winner_alpha, "l1_ratio": winner_l1},
            "MDA_mean": float(winner_row["MDA_mean"]),
            "MDA_std": float(winner_row["MDA_std"]),
            "NRMSE_mean": float(winner_row["NRMSE_mean"]),
            "NRMSE_std": float(winner_row["NRMSE_std"]),
            "NMSE_mean": float(winner_row["NMSE_mean"]),
            "NMAE_mean": float(winner_row["NMAE_mean"]),
            "delta_MDA_vs_persistence": float(winner_row["MDA_mean"] - PERSIST_MDA),
            "NRMSE_ratio_vs_codified_best": float(winner_row["NRMSE_mean"] / CODIFIED_GLOBAL_BEST_NRMSE),
            "NRMSE_ratio_vs_global_best": float(winner_row["NRMSE_mean"] / global_best_all),
            "vs_B_enet_T41": {
                "B_enet_T41_MDA": 0.5980835984449133,
                "B_enet_T41_NRMSE": CODIFIED_GLOBAL_BEST_NRMSE,
                "delta_MDA": float(winner_row["MDA_mean"] - 0.5980835984449133),
                "NRMSE_ratio": float(winner_row["NRMSE_mean"] / CODIFIED_GLOBAL_BEST_NRMSE),
            },
            "headroom_to_00090": {
                "persistence_NRMSE": PERSIST_NRMSE,
                "goal_NRMSE": 0.00090,
                "best_NRMSE": float(winner_row["NRMSE_mean"]),
                "gap_persistence_to_goal": PERSIST_NRMSE - 0.00090,
                "closed_fraction": float((PERSIST_NRMSE - float(winner_row["NRMSE_mean"])) / (PERSIST_NRMSE - 0.00090)) if PERSIST_NRMSE != 0.00090 else None,
                "remaining_to_goal": float(float(winner_row["NRMSE_mean"]) - 0.00090),
            },
        },
        "unconstrained_best": {
            "run_id": unconstrained_best["run_id"],
            "feature_set": unconstrained_best["feature_set"],
            "params": {"alpha": float(unconstrained_best["alpha"]), "l1_ratio": float(unconstrained_best["l1_ratio"])},
            "MDA_mean": float(unconstrained_best["MDA_mean"]),
            "NRMSE_mean": float(unconstrained_best["NRMSE_mean"]),
        },
        "gated_best_all": {
            "run_id": gated_best_id,
            "MDA_mean": float(gated_best["MDA_mean"]) if gated_best is not None else None,
            "NRMSE_mean": float(gated_best["NRMSE_mean"]) if gated_best is not None else None,
        } if gated_best is not None else None,
        "B_gated_best": {
            "run_id": b_gated_best["run_id"] if b_gated_best is not None else None,
            "params": {"alpha": float(b_gated_best["alpha"]), "l1_ratio": float(b_gated_best["l1_ratio"])} if b_gated_best is not None else None,
            "MDA_mean": float(b_gated_best["MDA_mean"]) if b_gated_best is not None else None,
            "NRMSE_mean": float(b_gated_best["NRMSE_mean"]) if b_gated_best is not None else None,
        } if b_gated_best is not None else None,
        "B_nrmse_best": {
            "run_id": b_nrmse_best["run_id"],
            "params": {"alpha": float(b_nrmse_best["alpha"]), "l1_ratio": float(b_nrmse_best["l1_ratio"])},
            "MDA_mean": float(b_nrmse_best["MDA_mean"]),
            "NRMSE_mean": float(b_nrmse_best["NRMSE_mean"]),
        } if b_nrmse_best is not None else None,
        "A_gated_best": {
            "run_id": a_gated_best["run_id"] if a_gated_best is not None else None,
            "params": {"alpha": float(a_gated_best["alpha"]), "l1_ratio": float(a_gated_best["l1_ratio"])} if a_gated_best is not None else None,
            "MDA_mean": float(a_gated_best["MDA_mean"]) if a_gated_best is not None else None,
            "NRMSE_mean": float(a_gated_best["NRMSE_mean"]) if a_gated_best is not None else None,
        } if a_gated_best is not None else None,
        "persistence": {"MDA_mean": PERSIST_MDA, "NRMSE_mean": PERSIST_NRMSE},
        "t41_B_enet": {"MDA_mean": 0.5980835984449133, "NRMSE_mean": CODIFIED_GLOBAL_BEST_NRMSE},
    }
    save_results_json(best_params, OUT_DIR / "best_params.json")

    # tuning_curves.png
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # Order by trial number overall (B then A interleaved by run_id? Better plot B and A separately)
    # Plot B trials
    b_df = df_trials[df_trials["feature_set"] == "B_base_lag1_OWN-subset"].sort_values("trial_number")
    a_df = df_trials[df_trials["feature_set"] == "A_base_lag1"].sort_values("trial_number")

    ax = axes[0]
    ax.scatter(b_df["trial_number"], b_df["MDA_mean"], c="tab:blue", label="B MDA", alpha=0.7, s=30)
    ax.scatter(a_df["trial_number"], a_df["MDA_mean"], c="lightblue", label="A MDA", alpha=0.6, s=30, marker="s")
    ax.axhline(PERSIST_MDA, color="red", linestyle="--", label="persistence 0.6771")
    ax.axhline(0.5980835984449133, color="green", linestyle=":", label="T4.1 B-enet 0.598")
    if b_gated_best is not None:
        ax.scatter([b_gated_best["trial_number"]], [b_gated_best["MDA_mean"]], c="gold", edgecolor="black", s=120, marker="*", label="B gated best", zorder=5)
    ax.set_ylabel("MDA mean")
    ax.set_title("T5.1 Optuna — MDA vs trial (outer CV)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(b_df["trial_number"], b_df["NRMSE_mean"], c="tab:orange", label="B NRMSE", alpha=0.7, s=30)
    ax.scatter(a_df["trial_number"], a_df["NRMSE_mean"], c="peachpuff", label="A NRMSE", alpha=0.6, s=30, marker="s")
    ax.axhline(PERSIST_NRMSE, color="red", linestyle="--", label="persistence 0.00113")
    ax.axhline(CODIFIED_GLOBAL_BEST_NRMSE, color="green", linestyle=":", label="T4.1 B-enet 0.001075")
    ax.axhline(0.00090, color="purple", linestyle=":", label="goal 0.00090 (-20%)")
    ax.axhline(gate_threshold, color="gray", linestyle="-.", label=f"gate {gate_threshold:.5f}")
    if b_nrmse_best is not None:
        ax.scatter([b_nrmse_best["trial_number"]], [b_nrmse_best["NRMSE_mean"]], c="gold", edgecolor="black", s=120, marker="*", label="B NRMSE best", zorder=5)
    ax.set_xlabel("trial number (per study)")
    ax.set_ylabel("NRMSE mean")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tuning_curves.png", dpi=180)
    plt.close(fig)
    print(f"[T5.1] saved tuning_curves.png")

    # decision_rule.json
    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "search": {
            "scope": "pruned A/B only (A_base_lag1 176 cols; B_base_lag1_OWN-subset 176+4 tot_d12/lag12/24/dlog1)",
            "model_family": "ElasticNet",
            "search_space": {"alpha": f"loguniform {ALPHA_LOW}..{ALPHA_HIGH}", "l1_ratio": L1_RATIOS, "selection": "cyclic (fixed)"},
            "n_trials": {"B": args.n_trials_B, "A": args.n_trials_A, "total": len(df_trials)},
            "sampler": f"TPESampler seed {args.seed}/{args.seed+1}",
            "cv": "outer 5 expanding folds 2008-09…2016-17 (direct outer Optuna)",
            "cv_caveat": "Direct outer CV tuning is optimistic vs nested inner 4×12m; inner would be strictly leakage-free. We document outer explicitly and note headroom is conditional upper bound.",
            "objective": "minimize mean NRMSE (outer CV) — aligns with primary -20% NRMSE goal; winner selection then uses MDA gated by NRMSE",
        },
        "gate": {
            "rule": "winner = best mean MDA subject to mean NRMSE ≤1.10× global-best NRMSE (global-best = min across T4 + T5 trials)",
            "global_best_NRMSE": global_best_all,
            "global_best_id": global_best_id,
            "gate_threshold": gate_threshold,
            "note": "Codified T4.1 B-enet 0.0010752 vs actual-min C-enet 0.0010728; T5 best may update global-best if lower.",
        },
        "eligible": eligible[["run_id","feature_set","MDA_mean","NRMSE_mean","NRMSE_ratio_vs_codified_best"]].to_dict(orient="records") if not eligible.empty else [],
        "winner": best_params["winner_B"],
        "unconstrained_best": best_params["unconstrained_best"],
        "gated_best": best_params["gated_best_all"],
        "headroom": best_params["winner_B"]["headroom_to_00090"] if "headroom_to_00090" in best_params["winner_B"] else None,
        "per_fold_variance_note": "All T5 trials scored on identical 5-fold expanding CV as T1–T4; per-trial per-fold metrics in trial_table.csv and per-trial runs.jsonl entries.",
        "caveats": [
            "Outer CV Optuna: reported outer metrics are optimistic relative to nested inner CV; ranking vs T4 remains apples-to-apples (same outer folds).",
            "No test-split file read; LEVEL target only.",
            "Pruned OWN-subset (4 cols) vs T4.1 OWN full (8 cols): B in T5 is A+ tot_d12/lag12/24/dlog1 only.",
            "Bounded search: 100 trials, 6 l1_ratio values, loguniform alpha 1e-4..1.0 with TPESampler.",
            "Upstream imputation panels canonical with ≤~1% optimism caveat.",
        ],
        "versions": versions_winner,
        "artifacts_generated": ["optuna_study.json","trial_table.csv","best_params.json","tuning_curves.png","decision_rule.json","feature_sets.json"],
    }
    save_results_json(decision, OUT_DIR / "decision_rule.json")

    print("\n=== T5.1 Optuna tuning complete ===")
    print(f"Winner B: {winner_row['run_id']} alpha={winner_alpha:.6g} l1={winner_l1} MDA={winner_row['MDA_mean']:.4f} NRMSE={winner_row['NRMSE_mean']:.6g}")


if __name__ == "__main__":
    main()
