"""
target_formulation.py — T2.2: level vs Δlog vs YoY% target formulation study.

Question: does reformulating the regression target fix ridge-on-lags' WORSE-
than-persistence directional accuracy (T2.1 finding: pooled ridge predicts
levels with point errors in persistence's class — NRMSE 0.00169 vs 0.00113 —
but MDA 0.371 vs 0.4625)?

Three target forms computed per CMA by BACKWARD-LOOKING transforms only:

    level : y_t
    dlog  : log(y_t) − log(y_{t−1})
    yoy   : 100 · (y_t / y_{t−12} − 1)

Test vehicle: the T2.1 pooled Ridge(alpha=1.0)-on-lags + CMA dummies
(StandardScaler inside a Pipeline fitted on the TRAIN WINDOW ONLY), reused
verbatim from baselines_suite.ridge_lags_model_fn. Each variant trains on its
transformed target and has its predictions INVERTED back to LEVEL space before
scoring, so all three compete against persistence on IDENTICAL level-space
metrics (MDA / NMSE / NRMSE / NMAE via harness wrappers).

Inversion / information-set rules (no future info beyond what persistence uses):
    level : identity.
    dlog  : ŷ_t = exp(log(y_{t−1}) + Δlog_pred_t)   — anchored on ACTUAL y_{t−1}
            (one-step-refreshed; exactly the information persistence conditions
            on, and the lag features already give ridge).
    yoy   : ŷ_t = y_{t−12} · (1 + yoy_pred_t/100)   — anchored on ACTUAL y_{t−12}.
Anchors come from a within-CMA shift(1)/shift(12) over the full train split:
purely backward-looking by construction. Rows whose anchor is missing (e.g. a
late starter's very first in-window month under yoy/dlog) yield NaN predictions
and drop out via the harness common mask, identically for model and baseline.

Sanity checks baked in:
  1. The `level` variant must reproduce T2.1-ridge-lags (identical setup).
  2. Persistence-in-level-space IS the zero-Δlog / zero-YoY forecast, so
     "persistence on transformed targets" adds no separate model — noted in
     analyses/T2.2.md rather than re-run.

No test-split file is read anywhere in this script.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parent.parent
for p in (str(EXPERIMENTS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness  # noqa: E402
from baselines_suite import RIDGE_ALPHA, ridge_lags_model_fn  # noqa: E402
from harness import (  # noqa: E402
    ARTIFACTS_DIR,
    TARGET,
    build_folds,
    current_git_sha,
    describe_folds,
    load_train_data,
    log_run,
    save_results_json,
)

OUT_DIR = ARTIFACTS_DIR / "T2.2"


# ---------------------------------------------------------------------------
# Target-form transforms (backward-looking only)
# ---------------------------------------------------------------------------

def prepare_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-CMA backward-shift anchors (_lag1, _lag12) and transformed
    targets (_dlog, _yoy). shift() inside groupby guarantees only PAST
    observations feed any transform — no leakage by construction.
    """
    out = df.sort_values(harness.ID_COLS).copy()
    g = out.groupby("cma_canonical")[TARGET]
    out["_lag1"] = g.shift(1)
    out["_lag12"] = g.shift(12)
    if (out[TARGET] <= 0).any():
        raise ValueError("non-positive NHPI levels — log transform invalid")
    out["_dlog"] = np.log(out[TARGET]) - np.log(out["_lag1"])
    out["_yoy"] = 100.0 * (out[TARGET] / out["_lag12"] - 1.0)
    return out


# ---------------------------------------------------------------------------
# One CV pass for a given target form
# ---------------------------------------------------------------------------

def evaluate_target_form(df_t: pd.DataFrame, folds: list[dict],
                         form: str, return_preds: bool = False) -> dict:
    """
    Mirrors harness.evaluate() exactly (same folds, same _score_fold path,
    same persistence baseline), except the ridge test vehicle is trained on
    the TRANSFORMED target and its validation predictions are inverted to
    LEVEL space before scoring.

    Per-fold entries record `n_train_rows_used` = rows whose transform is
    defined and therefore actually fed to the ridge fit (T2.3 critic fix #5),
    which is < the fold's nominal train size for dlog/yoy. With
    return_preds=True the RAW (un-inverted) validation predictions are also
    returned under "raw_val_preds" — used by the T2.3 direct-sign diagnostic.
    """
    transform_col = {"level": TARGET, "dlog": "_dlog", "yoy": "_yoy"}[form]

    def invert(val_df: pd.DataFrame, raw: pd.Series) -> pd.Series:
        if form == "level":
            return raw.astype(float)
        if form == "dlog":
            # ŷ_t = y_{t-1}(actual) * exp(pred); anchor NaN -> pred NaN
            return val_df["_lag1"] * np.exp(raw.astype(float))
        if form == "yoy":
            # ŷ_t = y_{t-12}(actual) * (1 + pred/100)
            return val_df["_lag12"] * (1.0 + raw.astype(float) / 100.0)
        raise ValueError(form)

    per_fold = []
    for f in folds:
        train_df = df_t.loc[f["train_index"]]
        val_df = df_t.loc[f["val_index"]]

        # Train ONLY on rows whose transform is defined (drop NaN targets);
        # identical feature matrix + pipeline as T2.1's ridge-on-lags.
        # Helper columns (_lag1/_lag12/_dlog/_yoy) are anchors/labels only —
        # never features — so they are stripped before fitting.
        helper_cols = [c for c in train_df.columns if c.startswith("_")]
        fit_mask = train_df[transform_col].notna()
        predict_fn = ridge_lags_model_fn(
            train_df.loc[fit_mask].drop(columns=[TARGET] + helper_cols),
            train_df.loc[fit_mask, transform_col],
        )
        raw = predict_fn(val_df.drop(columns=[TARGET] + helper_cols))
        pred_levels = pd.Series(invert(val_df, raw), index=val_df.index)

        baseline_pred = harness.persistence_predict(train_df, val_df, TARGET)

        # Identical scoring path as harness._score_fold: common-mask rows,
        # boundary-anchored MDA, evaluation.py point metrics.
        scores = harness._score_fold(val_df, train_df, pred_levels,
                                     baseline_pred, TARGET)
        n_nan_pred = int(len(val_df) - pred_levels.notna().sum())
        entry = {"fold_id": f["fold_id"],
                 "n_train_rows_used": int(fit_mask.sum()),  # critic fix #5
                 "n_nan_inverted_preds": n_nan_pred,
                 **scores}
        if return_preds:
            entry["raw_val_preds"] = {
                "index": [int(i) for i in val_df.index],
                "raw": [None if not np.isfinite(v) else float(v)
                        for v in raw],
            }
        per_fold.append(entry)

    out = {
        "folds": [
            {k: fl[k] for k in ("fold_id", "val_start_year", "val_end_year",
                                "n_train_rows", "n_val_rows",
                                "n_train_cmas", "n_val_cmas")}
            for fl in folds
        ],
        "per_fold": per_fold,
        "agg": harness._aggregate(per_fold),
        "config": {"n_folds": len(folds), "target": TARGET,
                   "anchor_year": folds[0]["anchor_year"],
                   "target_form": form},
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(d) -> str:
    return "  ".join(
        f"{m}={d[m]['mean']:.4f}±{d[m]['std']:.4f}"
        if abs(d[m]["mean"]) > 1e-3 else f"{m}={d[m]['mean']:.3e}±{d[m]['std']:.3e}"
        for m in harness.METRIC_NAMES
    )


def main() -> None:
    print("=== T2.2 target formulation study ===")
    df = load_train_data(TARGET)
    print(f"train split: {len(df)} rows, {df['cma_canonical'].nunique()} CMAs, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")

    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("\nfold design:\n" + describe_folds(folds))

    df_t = prepare_transforms(df)

    specs = [
        ("T2.2-level", "level",
         "Pooled Ridge(α=1)-on-lags predicting the LEVEL directly "
         "(reproduces T2.1-ridge-lags; sanity anchor for this study)."),
        ("T2.2-dlog", "dlog",
         "Same ridge predicting monthly log-difference Δlog; predictions "
         "inverted to levels via ACTUAL y_{t-1} (one-step-refreshed — the "
         "same information persistence conditions on)."),
        ("T2.2-yoy", "yoy",
         "Same ridge predicting YoY % change (y_t/y_{t-12} − 1); predictions "
         "inverted to levels via ACTUAL y_{t-12}."),
    ]

    runs = {}
    for run_id, form, desc in specs:
        print(f"\n--- running {run_id} ---")
        res = evaluate_target_form(df_t, folds, form)
        runs[run_id] = res

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        log_run(
            run_id=run_id, task="T2", sub="T2.2", description=desc,
            config={
                "git_sha_pre_commit": current_git_sha(),
                "model": "ridge_on_lags", "alpha": RIDGE_ALPHA,
                "target_form": form,
                "inversion_anchor": {"level": "none", "dlog": "actual y_{t-1}",
                                     "yoy": "actual y_{t-12}"}[form],
                "features": "all 176 lag-1 columns + CMA dummies",
                "scaling": "StandardScaler fit on train window only",
                "decision_rule": "primary: MDA improvement over persistence "
                                 "without sacrificing point-error class; "
                                 "secondary: stability across folds",
            },
            results=res,
        )
        agg = res["agg"]
        print(f"  model      : {fmt(agg['model'])}")
        print(f"  persistence: {fmt(agg['baseline'])}")

    # -- Sanity check 1: level variant reproduces T2.1-ridge-lags --------------
    # T2.3 critic fix #3: the guard covers the FULL per_fold dicts — every
    # metric, both blocks, every fold — not just the aggregate mean MDA.
    t21_path = ARTIFACTS_DIR / "T2.1" / "T2.1-ridge-lags_results.json"
    if t21_path.exists():
        import json
        t21 = json.loads(t21_path.read_text())
        pf_here = runs["T2.2-level"]["per_fold"]
        pf_ref = t21["per_fold"]
        assert len(pf_here) == len(pf_ref), \
            f"fold count mismatch: {len(pf_here)} vs {len(pf_ref)}"
        for a, b in zip(pf_here, pf_ref):
            assert a["fold_id"] == b["fold_id"]
            assert a["n_eval_rows"] == b["n_eval_rows"], \
                f"fold {a['fold_id']} eval-row count mismatch"
            for block in ("model", "baseline"):
                for m in harness.METRIC_NAMES:
                    assert abs(a[block][m] - b[block][m]) < 1e-12, \
                        (f"fold {a['fold_id']} {block} {m}: "
                         f"{a[block][m]} != {b[block][m]}")
        print(f"\n[sanity] T2.2-level reproduces T2.1-ridge-lags on ALL folds, "
              f"BOTH blocks, ALL metrics (full per-fold dict equality)")

    # -- Head-to-head table ----------------------------------------------------
    # T2.3 critic fix #4: cells formatted via fmt() so sub-1e-3 means
    # (NMSE!) render in scientific notation instead of being unreadable.
    def fmt_cell(mean: float, std: float) -> str:
        return (f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3
                else f"{mean:.3e}±{std:.3e}")

    rows = []
    for run_id, _, desc in specs:
        res = runs[run_id]
        agg_m = res["agg"]["model"]
        agg_b = res["agg"]["baseline"]
        row = {
            "predictor": run_id,
            **{m: fmt_cell(agg_m[m]["mean"], agg_m[m]["std"])
               for m in harness.METRIC_NAMES},
            **{f"{m}_vs_persist_delta": fmt_cell(agg_m[m]["mean"]
                                                 - agg_b[m]["mean"],
                                                 agg_m[m]["std"])
               for m in harness.METRIC_NAMES},
        }
        rows.append(row)
    board = pd.DataFrame(rows)
    board.to_csv(OUT_DIR / "head_to_head.csv", index=False)
    print("\nHEAD-TO-HEAD (mean±std across 5 canonical folds; deltas vs the "
          "paired same-mask persistence):\n" + board.to_string(index=False))

    # -- Plot -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors

    ax = axes[0]
    for i, (run_id, _, _) in enumerate(specs):
        vals = [pf["model"]["MDA"] for pf in runs[run_id]["per_fold"]]
        ax.plot(range(1, 6), vals, marker="o", color=colors[i],
                label=run_id.replace("T2.2-", ""))
    base_vals = [pf["baseline"]["MDA"] for pf in runs["T2.2-dlog"]["per_fold"]]
    ax.plot(range(1, 6), base_vals, marker="s", color="black",
            linestyle="--", label="persistence (paired)")
    ax.set_xticks(range(1, 6),
                  [f"F{k}\n({2008 + 2*(k-1)}-{2009 + 2*(k-1)})" for k in range(1, 6)])
    ax.set_ylabel("MDA"); ax.set_ylim(0, 1)
    ax.set_title("Per-fold directional accuracy by target form")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [r.replace("T2.2-", "") for r, _, _ in specs] + ["persistence"]
    means = ([runs[r]["agg"]["model"]["MDA"]["mean"] for r, _, _ in specs]
             + [np.mean(base_vals)])
    stds = ([runs[r]["agg"]["model"]["MDA"]["std"] for r, _, _ in specs]
            + [np.std(base_vals)])
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=[colors[i % 10] for i in range(len(labels))])
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylabel("MDA mean ± std (5 folds)"); ax.set_ylim(0, 1)
    ax.set_title("Aggregate MDA vs persistence")
    for i, mv in enumerate(means):
        ax.text(i, mv + 0.02, f"{mv:.4f}", ha="center", fontsize=8)
    fig.suptitle("T2.2 — level vs Δlog vs YoY% (ridge-on-lags test vehicle, "
                 "level-space scoring)")
    fig.tight_layout()
    plot_path = OUT_DIR / "t2_2_head_to_head.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[harness] saved plot -> {plot_path}")

    print("\n=== T2.2 target formulation study complete ===")


if __name__ == "__main__":
    main()
