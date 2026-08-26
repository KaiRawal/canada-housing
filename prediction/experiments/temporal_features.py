"""
temporal_features.py — T3.2: temporal features beyond lag-1, by source family.

Mini-goal: give models temporal signal beyond lag-1 — multi-year lags, diffs,
rolling stats of key CMHC groups (SCSS starts/completions, RMS rents, Census)
— and quantify which families carry signal for the final-model feature set.

FEATURE CONSTRUCTION (STRICTLY CAUSAL — information-set argument):
  Every candidate feature for row (CMA c, month t) is a pure backward-looking
  transform of the train-split columns, ending at t-1:
    * own-target level lags    y_{t-L},            L in {3, 6, 12, 24}
    * own-target diffs         y_{t-1}-y_{t-2}  and  y_{t-1}-y_{t-13}
      (groupby diff(1)/diff(12) then shift(1))
    * own-target Δlog          log(y_{t-1})-log(y_{t-2})
    * momentum sign            sign(y_{t-1}-y_{t-2})
    * family rolling stats     covariate_{t-1}.rolling(w, min_periods=6).mean/std,
                               w in {12, 36}   (window ENDS at t-1)
    * family 12-month diffs    covariate_{t-1} - covariate_{t-13}
  No transform ever references a value at or after t. All features are
  computed ONCE from the train-split file (X_train_full_total.csv covers
  1981-2022; the CV folds carve validation windows out of it, so each row's
  backward shifts resolve against rows strictly earlier in calendar time —
  the identical convention the upstream `total_lag_1` column already uses).
  Residual NaNs (each CMA's first months, before lags/windows exist) are
  filled causally: per-CMA forward-fill first, then same-month cross-
  sectional median (mirrors the T3.2 critic fix #5 neighbour-median fill),
  then overall column median. Fill counts are printed and saved.

FAMILIES / BLOCKS (key covariates chosen a priori from the column naming +
dropped_constant_features_total.txt; all 176 input columns carry `_lag_1`):
    OWN    : total_lag_1-derived (8 features)
    SCSS   : scss_starts_dwelling_type_all, scss_completions_dwelling_type_all
             -> rolling mean/std {12,36} + diff12 per covariate (10 features)
    RMS    : rms_average_rent_bedroom_type_total,
             rms_vacancy_rate_bedroom_type_total        -> same recipe (10)
    CENSUS : census_dwelling_value_average_total,
             census_income_..._median_household_income_before_taxes
             -> diff12 + rolling mean {12,36} per covariate (6; annual-ish
                source, std of a step function adds nothing)

VEHICLE: the T2.x pooled Ridge(alpha=1) + CMA dummies reused VERBATIM
(baselines_suite.ridge_lags_model_fn; StandardScaler fitted inside each
fold's train window ONLY), compared incrementally:
    lag1-only -> +OWN -> +SCSS -> +RMS -> +CENSUS (all blocks)
on identical folds/masks via harness.evaluate; persistence reported alongside
by the harness itself. Paired intersection-mask ΔMDA/NRMSE deltas of each
increment vs the lag1-only ridge are computed with
harness.paired_intersection_scores (anchor-bug-fixed since T3.2 critic fix
#1) on the same common masks as the headline scoring.

DECISION RULE (stated up front): a block earns T6 candidacy iff mean paired
ΔMDA (intersection-masked, increment minus lag1-only) > 0 when added AND the
full-mask aggregate NRMSE does NOT degrade >10% (ratio <= 1.10); per-fold
signs reported honestly. A directional edge alone never constitutes "better"
(KNOWLEDGE.md standing decision).

No test-split file is read anywhere in this script.
"""

import argparse
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

OUT_DIR = ARTIFACTS_DIR / "T3.2"

ROLL_WINDOWS = (12, 36)
ROLL_MIN_PERIODS = 6

# key covariates per family (df columns carry the `_lag_1` suffix)
FAMILY_KEYS = {
    "scss": [
        "scss_starts_dwelling_type_all_lag_1",
        "scss_completions_dwelling_type_all_lag_1",
    ],
    "rms": [
        "rms_average_rent_bedroom_type_total_lag_1",
        "rms_vacancy_rate_bedroom_type_total_lag_1",
    ],
    "census": [
        "census_dwelling_value_average_total_lag_1",
        "census_income_average_and_median_median_household_income_before_taxes_lag_1",
    ],
}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _fill_causal(values: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Causal NaN fill for engineered features: per-CMA forward-fill (past values
    of the SAME feature only) -> same-month cross-sectional median -> overall
    column median. Returns a Series aligned to df.index.
    """
    assert df[["cma_canonical", "date"]].equals(
        df[["cma_canonical", "date"]].sort_values(
            harness.ID_COLS).reset_index(drop=True)), \
        "df must be sorted by (cma, date) with a reset index"
    tmp = pd.DataFrame({"cma": df["cma_canonical"].to_numpy(),
                        "date": df["date"].to_numpy(),
                        "v": values.to_numpy()})
    tmp["v"] = tmp.groupby("cma")["v"].ffill()
    month = pd.to_datetime(tmp["date"]).dt.to_period("M")
    tmp["v"] = tmp["v"].fillna(
        tmp.assign(_m=month).groupby("_m")["v"].transform("median"))
    tmp["v"] = tmp["v"].fillna(tmp["v"].median())
    return pd.Series(tmp["v"].to_numpy(), index=df.index)


def build_own_block(df: pd.DataFrame) -> pd.DataFrame:
    """Own-target temporal features (see module docstring). 8 columns."""
    srt = df.sort_values(harness.ID_COLS)
    g = srt.groupby("cma_canonical")[TARGET]
    y = srt[TARGET]
    feats = pd.DataFrame(index=df.index)
    specs = {}
    for L in (3, 6, 12, 24):
        specs[f"tot_lag{L}"] = g.shift(L)
    d1 = g.diff(1).shift(1)          # y_{t-1} - y_{t-2}
    d12 = g.diff(12).shift(1)        # y_{t-1} - y_{t-13}
    dlog1 = (np.log(y) - np.log(y).groupby(srt["cma_canonical"])
             .shift(1)).shift(1)
    specs["tot_d1"] = d1
    specs["tot_d12"] = d12
    specs["tot_dlog1"] = dlog1
    specs["tot_mom_sign"] = np.sign(d1)
    n_filled = 0
    for name, raw in specs.items():
        filled = _fill_causal(raw, df)
        n_filled += int(raw.isna().sum())
        feats[name] = filled
    print(f"[temporal] OWN block: {feats.shape[1]} features, "
          f"{n_filled} NaN cells causally filled "
          f"(ffill -> same-month x-sectional median -> column median)")
    return feats


def build_family_block(df: pd.DataFrame, family: str,
                       with_std: bool = True) -> pd.DataFrame:
    """
    Rolling mean/std over ROLL_WINDOWS (ending at t-1) + 12-month diffs of the
    family's key covariates. CENSUS gets no rolling std (annual step series).
    """
    srt = df.sort_values(harness.ID_COLS)
    feats = pd.DataFrame(index=df.index)
    n_filled = 0
    for col in FAMILY_KEYS[family]:
        s = srt[col]
        lag1 = srt.groupby("cma_canonical")[col].shift(1)
        specs = {f"{col}_d12": lag1.groupby(srt["cma_canonical"]).diff(12)}
        for w in ROLL_WINDOWS:
            grp = srt.groupby("cma_canonical")[col]
            specs[f"{col}_roll{w}_mean"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=ROLL_MIN_PERIODS)
                .mean()).shift(1)
            if with_std:
                specs[f"{col}_roll{w}_std"] = grp.transform(
                    lambda x, w=w: x.rolling(w, min_periods=ROLL_MIN_PERIODS)
                    .std()).shift(1)
        for name, raw in specs.items():
            filled = _fill_causal(raw, df)
            n_filled += int(raw.isna().sum())
            feats[name] = filled
    print(f"[temporal] {family.upper()} block: {feats.shape[1]} features, "
          f"{n_filled} NaN cells causally filled")
    return feats


# ---------------------------------------------------------------------------
# Vehicle wrappers
# ---------------------------------------------------------------------------

def make_augmented_ridge_model_fn(*blocks: pd.DataFrame):
    """ridge_lags_model_fn verbatim after appending precomputed block cols."""
    extra = pd.concat(blocks, axis=1) if blocks else None

    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        tr = train_features.join(extra) if extra is not None else train_features
        base = ridge_lags_model_fn(tr, train_target)

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            va = (val_features.join(extra) if extra is not None
                  else val_features)
            return base(va)

        return predict_fn

    return model_fn


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3 else f"{mean:.3e}±{std:.3e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true",
                        help="recompute artifacts but skip runs.jsonl rows")
    args = parser.parse_args()

    print("=== T3.2 temporal-feature family ablation ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Build blocks ------------------------------------------------------
    own = build_own_block(df)
    scss = build_family_block(df, "scss", with_std=True)
    rms = build_family_block(df, "rms", with_std=True)
    census = build_family_block(df, "census", with_std=False)

    all_feats = pd.concat([own, scss, rms, census], axis=1)
    assert not all_feats.isna().any().any(), "NaN survived the causal fill"
    assert np.isfinite(all_feats.to_numpy(float)).all()

    # provenance dump: block membership + fill audit
    save_results_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "construction": "strictly backward-looking shifts/rollings computed "
                        "once from the train split (window ends t-1); causal "
                        "NaN fill: per-CMA ffill -> same-month cross-sectional "
                        "median -> column median",
        "roll_windows_months": list(ROLL_WINDOWS),
        "roll_min_periods": ROLL_MIN_PERIODS,
        "blocks": {
            "own": list(own.columns),
            "scss": {"key_covariates": FAMILY_KEYS["scss"],
                     "features": list(scss.columns)},
            "rms": {"key_covariates": FAMILY_KEYS["rms"],
                    "features": list(rms.columns)},
            "census": {"key_covariates": FAMILY_KEYS["census"],
                       "features": list(census.columns)},
        },
        "n_features_total": int(all_feats.shape[1]),
    }, OUT_DIR / "feature_construction.json")

    # ---- Incremental CV runs ------------------------------------------------
    steps = [
        ("T3.2-temporal-lag1only", (), "lag1-only"),
        ("T3.2-temporal-add-own", (own,), "lag1-only+OWN"),
        ("T3.2-temporal-add-scss", (own, scss), "lag1-only+OWN+SCSS"),
        ("T3.2-temporal-add-rms", (own, scss, rms), "lag1-only+OWN+SCSS+RMS"),
        ("T3.2-temporal-all", (own, scss, rms, census), "ALL blocks"),
    ]
    runs = {}
    for run_id, blocks, tag in steps:
        print(f"\n--- running {run_id} ({tag}) ---")
        res = evaluate(make_augmented_ridge_model_fn(*blocks),
                       df=df, folds=folds, target=TARGET)
        runs[tag] = res
        agg = res["agg"]
        print("  model      : " +
              "  ".join(f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}"
                        for m in harness.METRIC_NAMES))
        print("  persistence: " +
              "  ".join(f"{m}={fmt(agg['baseline'][m]['mean'], agg['baseline'][m]['std'])}"
                        for m in harness.METRIC_NAMES))
        cfg_extra = {
            "model": "ridge_on_lags+temporal" if blocks else "ridge_on_lags",
            "alpha": RIDGE_ALPHA,
            "blocks_added": [b for b in
                             ("own", "scss", "rms", "census")[:len(blocks)]],
            "n_engineered_features": sum(b.shape[1] for b in blocks),
            "n_base_features": 176,
        }
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        if not args.no_log:
            log_run(
                run_id=run_id, task="T3", sub="T3.2",
                description=(
                    f"Incremental temporal-feature step '{tag}': pooled "
                    f"Ridge({RIDGE_ALPHA:g})-on-lags + CMA dummies "
                    f"{'+' + '+'.join(cfg_extra['blocks_added']) + ' engineered temporal blocks' if blocks else 'with NO engineered additions (reference)'}; "
                    "identical canonical folds; persistence scored alongside "
                    "by the harness."),
                config={"git_sha_pre_commit": sha, **cfg_extra},
                results=res,
            )
        else:
            print("  [--no-log] registry row skipped")

    # ---- Paired intersection-mask deltas vs lag1-only ------------------------
    print("\n--- paired intersection-mask deltas: increments vs lag1-only ---")
    ref_tag = "lag1-only"
    delta_rows, paired_summary = [], {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha_pre_commit": sha,
        "decision_rule": "KEEP block for T6 iff mean paired ΔMDA(intersection,"
                         " increment - lag1-only) > 0 AND full-mask aggregate "
                         "NRMSE <= 1.10 x lag1-only",
        "increments": {},
    }
    for run_id, blocks, tag in steps[1:]:
        added = tag.replace("lag1-only+", "")
        per_fold, detail = [], []
        for f in folds:
            train_df = df.loc[f["train_index"]]
            val_df = df.loc[f["val_index"]]
            ref_pred_fn = ridge_lags_model_fn(train_df.drop(columns=[TARGET]),
                                              train_df[TARGET])
            inc_pred_fn = make_augmented_ridge_model_fn(*blocks)(
                train_df.drop(columns=[TARGET]), train_df[TARGET])
            ref_pred = pd.Series(ref_pred_fn(val_df.drop(columns=[TARGET])),
                                 index=val_df.index)
            inc_pred = pd.Series(inc_pred_fn(val_df.drop(columns=[TARGET])),
                                 index=val_df.index)
            base_pred = harness.persistence_predict(train_df, val_df, TARGET)

            common = (val_df[TARGET].notna() & inc_pred.notna()
                      & ref_pred.notna() & base_pred.notna())
            ve = val_df.loc[val_df.index[common]]
            paired = harness.paired_intersection_scores(
                train_df, ve,
                {"increment": inc_pred.loc[ve.index],
                 "lag1_only": ref_pred.loc[ve.index]}, TARGET)

            d_mda = (paired["per_predictor"]["increment"]["MDA"]
                     - paired["per_predictor"]["lag1_only"]["MDA"])
            per_fold.append({
                "fold_id": f["fold_id"],
                "n_eval_rows": int(common.sum()),
                "model": paired["per_predictor"]["increment"],
                "baseline": paired["per_predictor"]["lag1_only"],
            })
            detail.append({
                "fold_id": f["fold_id"],
                "val_window": f"{f['val_start_year']}-{f['val_end_year']}",
                "n_directions_intersection":
                    paired["counts"]["n_scored_directions_intersection"],
                "increment_MDA": paired["per_predictor"]["increment"]["MDA"],
                "lag1_only_MDA": paired["per_predictor"]["lag1_only"]["MDA"],
                "delta_MDA_increment_minus_lag1only": d_mda,
                "increment_NRMSE":
                    paired["per_predictor"]["increment"]["NRMSE"],
                "lag1_only_NRMSE":
                    paired["per_predictor"]["lag1_only"]["NRMSE"],
                "persistence_MDA_paired":
                    harness.paired_intersection_scores(
                        train_df, ve,
                        {"p": base_pred.loc[ve.index],
                         "q": inc_pred.loc[ve.index]},
                        TARGET)["per_predictor"]["p"]["MDA"],
            })
            print(f"  {added:>9} fold {f['fold_id']}: ΔMDA = {d_mda:+.4f}")

        agg = harness._aggregate(per_fold)
        deltas = [d["delta_MDA_increment_minus_lag1only"] for d in detail]
        n_pos = int(np.sum(np.asarray(deltas) > 0))
        mean_delta = float(np.mean(deltas))
        nrmse_ratio = (runs[tag]["agg"]["model"]["NRMSE"]["mean"]
                       / runs[ref_tag]["agg"]["model"]["NRMSE"]["mean"])
        keep = bool(mean_delta > 0 and nrmse_ratio <= 1.10)
        paired_summary["increments"][added] = {
            "agg": agg, "detail": detail,
            "folds_with_positive_delta_MDA": f"{n_pos}/{len(detail)}",
            "mean_paired_delta_MDA": mean_delta,
            "aggregate_full_mask_NRMSE_ratio_vs_lag1only":
                round(nrmse_ratio, 4),
            "verdict_keep_for_T6": keep,
        }
        print(f"  {added}: ΔMDA>0 on {n_pos}/{len(detail)} folds "
              f"(mean {mean_delta:+.4f}); full-mask NRMSE ratio "
              f"{nrmse_ratio:.3f} -> {'KEEP' if keep else 'DROP'}")
        delta_rows.extend([{**d, "block_step": added} for d in detail])

        if not args.no_log:
            log_run(
                run_id=(f"T3.2-intersect-{'-'.join(added.lower().split('+'))}"
                        "-vs-lag1only".replace(" ", "-")),
                task="T3", sub="T3.2",
                description=(
                    f"PAIRED intersection-mask comparison for the T3.2 "
                    f"decision rule: {tag} vs lag1-only ridge on one shared "
                    "nonzero-direction set per fold; persistence MDA on the "
                    "same mask recorded alongside."),
                config={"git_sha_pre_commit": sha,
                        "scoring": "paired intersection-mask",
                        "comparison": f"{tag} vs lag1-only ridge",
                        "alpha": RIDGE_ALPHA,
                        "blocks_added": added},
                results={"folds": [{"fold_id": fl["fold_id"]} for fl in folds],
                         "per_fold": per_fold, "agg": agg,
                         "config": {"scoring": "paired intersection-mask"}},
            )

    delta_tab = pd.DataFrame(delta_rows)[
        ["block_step", "fold_id", "val_window", "n_directions_intersection",
         "increment_MDA", "lag1_only_MDA",
         "delta_MDA_increment_minus_lag1only", "increment_NRMSE",
         "lag1_only_NRMSE", "persistence_MDA_paired"]]
    delta_tab.to_csv(OUT_DIR / "paired_delta_table.csv", index=False)
    save_results_json(paired_summary, OUT_DIR / "decision_rule.json")

    # ---- Incremental block table (headline aggregates) -----------------------
    board_rows = []
    for run_id, blocks, tag in steps:
        m = runs[tag]["agg"]["model"]
        b = runs[tag]["agg"]["baseline"]
        board_rows.append({
            "step": tag,
            **{f"model_{k}": fmt(m[k]["mean"], m[k]["std"])
               for k in harness.METRIC_NAMES},
            **{f"persist_{k}": fmt(b[k]["mean"], b[k]["std"])
               for k in harness.METRIC_NAMES},
        })
    board = pd.DataFrame(board_rows)
    board.to_csv(OUT_DIR / "incremental_block_table.csv", index=False)
    print("\nINCREMENTAL BLOCK TABLE (mean±std over the 5 canonical folds):")
    print(board.to_string(index=False))

    # ---- Plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "lag1-only": ("tab:blue", "--", "o"),
        "lag1-only+OWN": ("tab:green", "-", "o"),
        "lag1-only+OWN+SCSS": ("tab:orange", "-", "s"),
        "lag1-only+OWN+SCSS+RMS": ("tab:red", "-", "D"),
        "ALL blocks": ("tab:purple", "-", "^"),
    }
    ax.plot(range(1, 6),
            [pf["baseline"]["MDA"] for pf in runs[ref_tag]["per_fold"]],
            marker="x", color="black", linewidth=2, alpha=0.6,
            label="persistence (same mask)")
    for tag, (color, ls, mk) in styles.items():
        ax.plot(range(1, 6),
                [pf["model"]["MDA"] for pf in runs[tag]["per_fold"]],
                marker=mk, linestyle=ls, color=color, label=tag)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("fold (validation window)")
    ax.set_ylabel("boundary-anchored zero-drop MDA")
    ax.set_title("T3.2 — incremental temporal-feature blocks (canonical folds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "t3_2_temporal_blocks.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[harness] saved plot -> {plot_path}")

    print("\n=== T3.2 temporal-feature ablation complete ===")


if __name__ == "__main__":
    main()
