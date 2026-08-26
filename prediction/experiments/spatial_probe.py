"""
spatial_probe.py — T2.3: leave-one-CMA-out (LOCO) probe of spatial pooling.

Question: does POOLING across CMAs help predict a held-out CMA at all? This
gates the T3 investment in spatial features (kNN neighbour lags, region
encodings, centroids): if cross-CMA information cannot beat what a CMA's own
history provides, spatial-lag features are unlikely to add signal either.

DECISION RULE (stated up front, before results were seen):
  Spatial pooling is deemed PROMISING iff pooled-LOCO beats BOTH
    (a) the persistence baseline AND
    (b) a self-fit ridge trained on the held-out CMA's OWN train history
  on a majority (>50%) of evaluable held-out CMAs (aggregate MDA per CMA over
  the 5 canonical folds), without landing in a hopeless point-error class
  (NRMSE within ~3x the winner's).
  Otherwise -> temporal-only path recommended for T3/T4.

Design (bounded compute: <= 24 CMAs x 5 folds = 120 pooled fits + <= 120
self-fits, all trivial):
  * Vehicle: the T2.1/T2.2 pooled Ridge(alpha=1)-on-lags + CMA dummies,
    reused VERBATIM from baselines_suite.ridge_lags_model_fn (StandardScaler
    inside the pipeline, fitted on the fold's train window only). For the LOCO
    variant the fit EXCLUDES every train-window row of the held-out CMA; its
    dummy is then unseen at fit time and is handled by the existing
    reindex(columns=train_columns, fill_value=0.0) — i.e. the held-out CMA is
    predicted from numeric lag features alone (graceful unseen-CMA handling).
  * Self-fit variant: the SAME model function fitted on ONLY the held-out
    CMA's train rows (single constant dummy column — absorbed by the scaler,
    documented). Feasibility rule: >= 24 monthly train rows (2y); otherwise
    the CMA/fold is EXCLUDED and counted, never fudged (the six late-starter
    CMAs starting ~2016-12 have ZERO train rows in every canonical fold, whose
    train windows end before 2017; Charlottetown starts ~1995 and is fine).
  * Scoring: harness._score_fold per (fold, held-out CMA) cell on exactly that
    CMA's validation rows — common-mask pairing with the PERSISTENCE baseline
    computed by the harness itself on the identical rows; boundary-anchored
    zero-direction-drop MDA identical to every other stage. The train->val
    MDA anchor legitimately uses the held-out CMA's last TRAIN observation
    (past information only).
  * Aggregation: per-held-out-CMA mean metrics across folds; deltas of LOCO
    vs (a) persistence and (b) self-fit; structure check — does the pooling
    benefit correlate with the CMA's train-history length (small/new CMAs
    benefit more)?

No test-split file is read anywhere in this script.
"""

import json
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

OUT_DIR = ARTIFACTS_DIR / "T2.3"
MIN_SELF_FIT_ROWS = 24  # >= 2 years of monthly train history


def score_cells(df: pd.DataFrame, folds: list[dict],
                mode: str) -> tuple[list[dict], list[dict]]:
    """
    Run one scored cell per (fold, held-out CMA present in the val window).

    mode="loco"    : pooled ridge fitted on the train window EXCLUDING the
                     held-out CMA.
    mode="selffit" : ridge fitted on ONLY the held-out CMA's train rows;
                     cells with < MIN_SELF_FIT_ROWS train rows are skipped
                     (recorded as exclusions, not silently dropped).

    Returns (cells, exclusions): each cell is one _score_fold output dict plus
    ids; exclusions records why a self-fit cell was not run.
    """
    assert mode in ("loco", "selffit")
    cells, exclusions = [], []
    for f in folds:
        train_df = df.loc[f["train_index"]]
        val_df = df.loc[f["val_index"]]
        for cma, g_val in val_df.groupby("cma_canonical", sort=False):
            tr_cma = train_df[train_df["cma_canonical"] == cma]
            n_hist = len(tr_cma)
            if mode == "loco":
                fit_df = train_df[train_df["cma_canonical"] != cma]
            else:
                if n_hist < MIN_SELF_FIT_ROWS:
                    exclusions.append({
                        "fold_id": f["fold_id"], "cma": cma,
                        "n_train_rows": int(n_hist),
                        "reason": f"< {MIN_SELF_FIT_ROWS} train rows",
                    })
                    continue
                fit_df = tr_cma
            if len(g_val) == 0:  # cannot happen (groupby over val rows)
                continue

            # Fit sees ONLY the (restricted) train window; the date/id columns
            # ride along because ridge_lags_model_fn expects the harness
            # feature-frame contract (ids included, date dropped inside).
            predict_fn = ridge_lags_model_fn(
                fit_df.drop(columns=[TARGET]), fit_df[TARGET])
            pred = pd.Series(
                predict_fn(g_val.drop(columns=[TARGET])), index=g_val.index)

            base = harness.persistence_predict(train_df, g_val, TARGET)
            scores = harness._score_fold(g_val, train_df, pred, base, TARGET)
            first_date = df[df["cma_canonical"] == cma]["date"].min()
            cells.append({
                "fold_id": f["fold_id"], "cma": cma,
                "n_train_rows_held_out_cma": int(n_hist),
                "n_pool_train_rows": int(len(fit_df)),
                "cma_first_obs": str(first_date.date()),
                **scores,
            })
    return cells, exclusions


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

METRICS = harness.METRIC_NAMES


def per_cma_table(cells: list[dict]) -> pd.DataFrame:
    """Mean metric per held-out CMA across the folds where it was evaluated."""
    rows = []
    for cma, g in pd.DataFrame(cells).groupby("cma"):
        row = {"cma": cma, "n_folds_evaluated": len(g),
               "n_train_rows_first_fold": int(g["n_train_rows_held_out_cma"].min()),
               "cma_first_obs": g["cma_first_obs"].iloc[0]}
        for block in ("model", "baseline"):
            for m in METRICS:
                vals = g[block].map(lambda d: d[m]).dropna()
                vals = vals[np.isfinite(vals.astype(float))]
                row[f"{block}_{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean: float, std: float) -> str:
    return (f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3
            else f"{mean:.3e}±{std:.3e}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true",
                        help="recompute artifacts but skip runs.jsonl rows "
                             "(used to re-enter main() after a partial run "
                             "already appended this stage's rows)")
    args = parser.parse_args()

    print("=== T2.3 leave-one-CMA-out spatial-pooling probe ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()

    print("\n--- LOCO pooled ridge (held-out CMA excluded from fit) ---")
    loco_cells, loco_excl = score_cells(df, folds, "loco")
    print(f"{len(loco_cells)} cells scored")

    print("\n--- self-fit ridge (held-out CMA's OWN train history only) ---")
    self_cells, self_excl = score_cells(df, folds, "selffit")
    print(f"{len(self_cells)} cells scored "
          f"({len(self_excl)} exclusions for < {MIN_SELF_FIT_ROWS} train rows)")

    loco_res = {
        "folds": [{"fold_id": fl["fold_id"],
                   "val_start_year": fl["val_start_year"],
                   "val_end_year": fl["val_end_year"]} for fl in folds],
        "per_cell": loco_cells,
        "config": {"n_folds": len(folds), "target": TARGET,
                   "anchor_year": folds[0]["anchor_year"], "mode": "loco"},
    }
    self_res = {
        "folds": loco_res["folds"],
        "per_cell": self_cells,
        "exclusions": self_excl,
        "config": {**loco_res["config"], "mode": "selffit"},
    }
    save_results_json(loco_res, OUT_DIR / "T2.3-loco-pooled_results.json")
    save_results_json(self_res, OUT_DIR / "T2.3-selffit-ridge_results.json")

    # -- Registry rows ----------------------------------------------------------
    def agg_over_cells(cells: list[dict]) -> dict:
        pf = [{"fold_id": c["fold_id"], "n_eval_rows": c["n_eval_rows"],
               "model": c["model"], "baseline": c["baseline"]}
              for c in cells]
        return pf, harness._aggregate(pf)

    for rid, res, desc in [
        ("T2.3-loco-pooled", loco_res,
         "LOCO probe: pooled Ridge(α=1)-on-lags + CMA dummies fitted on the "
         "train window EXCLUDING the held-out CMA, predicting that CMA over "
         "each canonical fold's val window; one scored cell per (fold, CMA); "
         "unseen-CMA dummy handled by reindex(fill=0); persistence baseline "
         "paired on identical masks."),
        ("T2.3-selffit-ridge", self_res,
         "Self-fit comparator: the same ridge fitted on ONLY the held-out "
         "CMA's own train history (>= 24 monthly rows required; late-starter "
         "CMAs with none are excluded and counted, never fudged); isolates "
         "pooling benefit from no-model baselines."),
    ]:
        pf, agg = agg_over_cells(res["per_cell"])
        if not args.no_log:
            log_run(
                run_id=rid, task="T2", sub="T2.3", description=desc,
                config={"git_sha_pre_commit": sha, "model": "ridge_on_lags",
                        "alpha": RIDGE_ALPHA,
                        "unit_of_analysis": "(fold, held-out CMA) cell",
                        "n_cells": len(res["per_cell"]),
                        "min_self_fit_rows": MIN_SELF_FIT_ROWS,
                        **res["config"]},
                results={"folds": res["folds"], "per_fold": pf, "agg": agg,
                         "config": res["config"]},
            )
        m = agg["model"]["MDA"]; b = agg["baseline"]["MDA"]
        print(f"\n{rid}: MDA {fmt(m['mean'], m['std'])}  vs persistence "
              f"{fmt(b['mean'], b['std'])}  "
              f"(NRMSE {agg['model']['NRMSE']['mean']:.5f} vs "
              f"{agg['baseline']['NRMSE']['mean']:.5f})")

    # -- Per-CMA table + deltas ---------------------------------------------------
    loco_tab = per_cma_table(loco_cells).rename(
        columns={f"{b}_{m}_mean": f"loco_{b}_{m}" for b in ("model", "baseline")
                 for m in METRICS})
    self_tab = per_cma_table(self_cells).rename(
        columns={f"{b}_{m}_mean": f"self_{b}_{m}" for b in ("model", "baseline")
                 for m in METRICS})
    tab = loco_tab.merge(self_tab.drop(columns=["cma_first_obs",
                                                "n_folds_evaluated",
                                                "n_train_rows_first_fold"]),
                         on="cma", how="left")
    tab["delta_loco_minus_persist_MDA"] = (tab["loco_model_MDA"]
                                           - tab["loco_baseline_MDA"])
    tab["delta_loco_minus_self_MDA"] = (tab["loco_model_MDA"]
                                        - tab["self_model_MDA"])
    tab = tab.sort_values("delta_loco_minus_persist_MDA")

    out_csv = OUT_DIR / "per_cma_loco_summary.csv"
    tab.to_csv(out_csv, index=False)
    print(f"\n[harness] saved per-CMA table -> {out_csv}")

    cols = ["cma", "n_folds_evaluated", "n_train_rows_first_fold",
            "loco_model_MDA", "loco_baseline_MDA", "self_model_MDA",
            "delta_loco_minus_persist_MDA", "delta_loco_minus_self_MDA",
            "loco_model_NRMSE", "self_model_NRMSE", "loco_baseline_NRMSE"]
    pretty = tab[cols].copy()
    for c in cols:
        if pretty[c].dtype.kind == "f":
            pretty[c] = pretty[c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "-")
    print("\nPER-CMA LOCO RESULTS (sorted by pooling benefit vs persistence):\n"
          + pretty.to_string(index=False))

    # -- Verdict against the pre-stated decision rule ------------------------------
    evaluable = tab.dropna(subset=["delta_loco_minus_persist_MDA",
                                   "delta_loco_minus_self_MDA"])
    beats_persist = int((evaluable["delta_loco_minus_persist_MDA"] > 0).sum())
    beats_self = int((evaluable["delta_loco_minus_self_MDA"] > 0).sum())
    beats_both = int(((evaluable["delta_loco_minus_persist_MDA"] > 0)
                      & (evaluable["delta_loco_minus_self_MDA"] > 0)).sum())
    n_ev = len(evaluable)

    # Structure check: does pooling benefit track history length?
    hist_corr = float(evaluable["n_train_rows_first_fold"].corr(
        evaluable["delta_loco_minus_persist_MDA"])) if n_ev > 2 else float("nan")

    verdict = {
        "decision_rule": "PROMISING iff LOCO beats BOTH persistence and "
                         "self-fit on >50% of evaluable held-out CMAs "
                         "(aggregate MDA) without a hopeless NRMSE class",
        "n_evaluable_cmas": n_ev,
        "beats_persistence": beats_persist,
        "beats_self_fit": beats_self,
        "beats_both": beats_both,
        "persistence_beats_loco_count":
            int((evaluable["delta_loco_minus_persist_MDA"] < 0).sum()),
        "hist_len_vs_benefit_corr": hist_corr,
        "verdict": ("PROMISING" if n_ev and beats_both > n_ev / 2
                    else "NOT PROMISING -> temporal-only path"),
    }
    save_results_json({"generated_at": pd.Timestamp.utcnow().isoformat(),
                       "git_sha_pre_commit": sha, **verdict},
                      OUT_DIR / "verdict.json")
    print("\nVERDICT: " + json.dumps(verdict, indent=2))

    # -- Plots ----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    ax = axes[0]
    ev = evaluable.sort_values("delta_loco_minus_persist_MDA")
    y = np.arange(len(ev))
    ax.barh(y - 0.2, ev["delta_loco_minus_persist_MDA"], height=0.4,
            color=colors[0], label="LOCO − persistence")
    ax.barh(y + 0.2, ev["delta_loco_minus_self_MDA"], height=0.4,
            color=colors[1], label="LOCO − self-fit")
    ax.set_yticks(y, ev["cma"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔMDA (pooled-LOCO advantage)")
    ax.set_title("Per-CMA pooling benefit (MDA)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    ax.scatter(tab["n_train_rows_first_fold"],
               tab["delta_loco_minus_persist_MDA"], color=colors[3])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("held-out CMA train rows in fold 1 (history depth)")
    ax.set_ylabel("LOCO − persistence ΔMDA")
    ax.set_title(f"Does pooling help short-history CMAs more?\n"
                 f"corr = {hist_corr:.2f}")
    ax.grid(alpha=0.3)
    fig.suptitle("T2.3 — leave-one-CMA-out spatial-pooling probe")
    fig.tight_layout()
    plot_path = OUT_DIR / "t2_3_loco_probe.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[harness] saved plot -> {plot_path}")

    print("\n=== T2.3 LOCO probe complete ===")


if __name__ == "__main__":
    main()
