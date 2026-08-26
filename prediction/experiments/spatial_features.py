"""
spatial_features.py — T3.1: cheap ablation of kNN neighbour-lag features.

Context (T2.3): LOCO showed pooled lag-models do NOT transfer across CMAs, so
spatial features were DEMOTED to a cheap ablation. This script quantifies
whether neighbour-lag features add anything OVER temporal-only lags for a
within-CMA pooled ridge.

Features (STRICTLY backward-looking — info-set argument):
  For row (CMA c, month t) with neighbours N_k(c) = k nearest CMAs by
  centroid great-circle distance (k=3 and k=5 variants):
    nbr{k}_lvl_mean   = mean_{j in N_k(c)} y_{j, t-1}
    nbr{k}_dlt_mean   = mean_{j in N_k(c)} (y_{j, t-1} - y_{j, t-2})
    nbr{k}_lvl_wmean  = distance-weighted variant, w_j ∝ 1/d_cj
    nbr{k}_dlt_wmean  =        "           on the Δ feature
  Neighbours' y_{t-1} / y_{t-2} are OBSERVED actuals at forecast time t under
  exactly the same assumption the whole pipeline already makes for the CMA's
  OWN lag features (persistence baseline, ridge-on-lags, recursive variants
  all condition on actual y_{t-1}; see baselines_suite._past_lag1 docstring).
  Only backward shifts are used — no contemporaneous or future neighbour
  values — so the information set is identical to the temporal-only model's,
  plus neighbour pasts. Late-starter CMAs (~2016-12) have no own history in
  folds 1–4 but their NEIGHBOURS do; the features therefore exist for every
  late-starter validation row (only each CMA's own first two months anywhere
  in the panel lack lags -> filled 0.0, documented).

Vehicle: the T2.x pooled Ridge(alpha=1)-on-lags + CMA dummies reused VERBATIM
(baselines_suite.ridge_lags_model_fn; StandardScaler inside fold), run twice:
  * temporal-only (all original lag columns)
  * + the four k-neighbour columns above
on identical folds/masks; persistence reported alongside by the harness.

DECISION RULE (stated up front): neighbour-lag features earn a place in T6
ablations iff paired ΔMDA (intersection-masked, spatial minus temporal-only)
> 0 on a MAJORITY of the 5 folds AND point metrics do not degrade materially
(spatial aggregate NRMSE <= 1.10 x temporal-only).

Optional LightGBM probe: SKIPPED — lightgbm is not installed in this
environment and installing heavy deps is not warranted for an ablation slot.

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

OUT_DIR = ARTIFACTS_DIR / "T3.1"
SHAPEFILE = ROOT / "data_cleaning" / "data" / "cleaned" / "cma_boundary" / \
    "lcma000b21a_e.shp"

# cma_canonical (imputation output) -> Census 2021 shapefile CMANAME (CMATYPE=B)
CMA_TO_SHAPE = {
    "St. John's, Newfoundland and Labrador": "St. John's",
    "Halifax, Nova Scotia": "Halifax",
    "Montréal, Quebec": "Montréal",
    "Ottawa-Gatineau, Ontario part, Ontario/Quebec":
        "Ottawa - Gatineau (Ontario part / partie de l'Ontario)",
    "Ottawa-Gatineau, Quebec part, Ontario/Quebec":
        "Ottawa - Gatineau (partie du Québec / Quebec part)",
    "Québec, Quebec": "Québec",
    "Sherbrooke, Quebec": "Sherbrooke",
    "Trois-Rivières, Quebec": "Trois-Rivières",
    "Greater Sudbury, Ontario": "Greater Sudbury / Grand Sudbury",
    "Hamilton, Ontario": "Hamilton",
    "St. Catharines-Niagara, Ontario": "St. Catharines - Niagara",
    "Kitchener-Cambridge-Waterloo, Ontario": "Kitchener - Cambridge - Waterloo",
    "Guelph, Ontario": "Guelph",
    "London, Ontario": "London",
    "Oshawa, Ontario": "Oshawa",
    "Windsor, Ontario": "Windsor",
    "Winnipeg, Manitoba": "Winnipeg",
    "Regina, Saskatchewan": "Regina",
    "Saskatoon, Saskatchewan": "Saskatoon",
    "Kelowna, British Columbia": "Kelowna",
    "Vancouver, British Columbia": "Vancouver",
    "Victoria, British Columbia": "Victoria",
    "Edmonton, Alberta": "Edmonton",
    "Charlottetown, Prince Edward Island": "Charlottetown",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_centroids() -> pd.DataFrame:
    """
    Representative-point centroids (lat/lon) of the 24 study CMAs from the
    Census 2021 boundary shapefile used upstream. CAVEAT (documented): the
    2021 boundaries are applied to the full 1981-2022 panel; centroid
    geography moves negligibly over the sample, so this is benign for kNN
    weights.
    """
    import geopandas as gpd

    g = gpd.read_file(SHAPEFILE)
    g["lat"] = g.geometry.representative_point().y
    g["lon"] = g.geometry.representative_point().x
    rows = {}
    for canonical, shape_name in CMA_TO_SHAPE.items():
        hit = g[g["CMANAME"] == shape_name]
        assert len(hit) >= 1, f"shapefile match not found: {shape_name}"
        rows[canonical] = {"shape_cmaname": shape_name,
                           "lat": float(hit["lat"].iloc[0]),
                           "lon": float(hit["lon"].iloc[0])}
    return pd.DataFrame(rows).T.astype({"lat": float, "lon": float})


def knn_table(centroids: pd.DataFrame, k: int) -> pd.DataFrame:
    """Long table: cma, neighbor, dist_km for the k nearest neighbours."""
    recs = []
    cmalist = list(centroids.index)
    lats = centroids["lat"].to_numpy()
    lons = centroids["lon"].to_numpy()
    for i, c in enumerate(cmalist):
        d = haversine_km(lats[i], lons[i], lats, lons)
        others = [(o, float(d[j])) for j, o in enumerate(cmalist) if o != c]
        others.sort(key=lambda t: t[1])
        for rank, (nbr, dist) in enumerate(others[:k], start=1):
            recs.append({"cma": c, "neighbor": nbr, "dist_km": dist,
                         "rank": rank})
    return pd.DataFrame(recs)


def _backward_panels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wide (date x CMA) panels of the backward-shifted own-history features:
      lvl1(c, t) = y_{c, t-1},   dlt1(c, t) = y_{c, t-1} - y_{c, t-2}
    Pure shift(1)/diff(1).shift(1) per CMA — no future information.
    """
    srt = df.sort_values(harness.ID_COLS)
    lvl1 = srt.groupby("cma_canonical")[TARGET].shift(1)
    dlt1 = srt.groupby("cma_canonical")[TARGET].diff(1).shift(1)
    base = srt[["cma_canonical", "date"]]
    wide_lvl = base.assign(_v=lvl1.to_numpy()).pivot(
        index="date", columns="cma_canonical", values="_v")
    wide_dlt = base.assign(_v=dlt1.to_numpy()).pivot(
        index="date", columns="cma_canonical", values="_v")
    return wide_lvl, wide_dlt


def build_spatial_features_strict(df: pd.DataFrame, knn: pd.DataFrame,
                                  prefix: str) -> pd.DataFrame:
    """
    Four backward-looking neighbour-lag columns aligned to df's index.
    For each CMA c and month t:
      {pfx}_lvl_mean  = mean_{j in N(c)} y_{j, t-1}
      {pfx}_dlt_mean  = mean_{j in N(c)} dlt1(j, t)
      {pfx}_lvl_wmean = Σ_j w_cj * lvl1(j,t) / Σ_j w_cj,  w_cj = 1/dist_km(c,j)
      {pfx}_dlt_wmean =        "          on the Δ feature
    Aggregation goes through the date-pivoted wide panels so neighbour rows
    align by calendar month. NaN cells (each CMA's own first two panel months,
    plus nothing else — late starters' neighbours have history) -> 0.0,
    documented.
    """
    wide_lvl, wide_dlt = _backward_panels(df)
    panels = {"lvl": wide_lvl, "dlt": wide_dlt}
    feats = pd.DataFrame(index=df.index)
    for c, grp in knn.groupby("cma"):
        nbrs = grp["neighbor"].tolist()
        w = grp["dist_km"].to_numpy()
        w = w / w.sum()
        idx_c = df.index[df["cma_canonical"] == c]
        dates_c = df.loc[idx_c, "date"]
        for feat, wide in panels.items():
            mat = wide.reindex(columns=nbrs).loc[dates_c].to_numpy()
            obs = ~np.isnan(mat)
            with np.errstate(invalid="ignore", divide="ignore"):
                feats.loc[idx_c, f"{prefix}_{feat}_mean"] = \
                    np.nansum(np.where(obs, mat, 0.0), axis=1) / obs.sum(axis=1)
                feats.loc[idx_c, f"{prefix}_{feat}_wmean"] = \
                    np.nansum(np.where(obs, mat * w, 0.0), axis=1) \
                    / (obs * w).sum(axis=1)
    n_nan = int(feats.isna().sum().sum())
    print(f"[spatial] {feats.shape[1]} features ({prefix}), NaN cells filled "
          f"with 0.0: {n_nan} ({n_nan / feats.size:.3%} of cells)")
    return feats.fillna(0.0)


def make_spatial_ridge_model_fn(feats: pd.DataFrame):
    """
    Wrap ridge_lags_model_fn verbatim after appending the precomputed
    neighbour-lag columns (joined by row index; both train and val frames
    carry the parent df index because harness.evaluate slices df directly).
    """
    def model_fn(train_features: pd.DataFrame, train_target: pd.Series):
        tr = train_features.join(feats)
        base = ridge_lags_model_fn(tr, train_target)

        def predict_fn(val_features: pd.DataFrame) -> pd.Series:
            return base(val_features.join(feats))

        return predict_fn

    return model_fn


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}" if abs(mean) > 1e-3 else f"{mean:.3e}±{std:.3e}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true",
                        help="recompute artifacts but skip runs.jsonl rows")
    args = parser.parse_args()

    print("=== T3.1 kNN neighbour-lag feature ablation ===")
    df = load_train_data(TARGET)
    folds = build_folds(df, n_folds=5, val_window_years=2)
    print("fold design:\n" + describe_folds(folds))
    sha = current_git_sha()

    # ---- Geography -------------------------------------------------------
    centroids = load_centroids()
    knns = {k: knn_table(centroids, k) for k in (3, 5)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo_dump = {
        "source_shapefile": str(SHAPEFILE.relative_to(ROOT)),
        "centroid_method": "representative_point (EPSG:3347 -> WGS84 lon/lat)",
        "distance": "great-circle km",
        "caveat": "Census 2021 boundaries applied to the 1981-2022 panel "
                  "(benign for kNN weights; centroid drift negligible)",
        "centroids": centroids.to_dict(orient="index"),
        "knn": {str(k): tbl.to_dict(orient="records")
                for k, tbl in knns.items()},
    }
    save_results_json(geo_dump, OUT_DIR / "knn_weights.json")

    # ---- Feature frames ---------------------------------------------------
    feats_by_k = {}
    for k, tbl in knns.items():
        prefix = f"nbr{k}"
        feats = build_spatial_features_strict(df, tbl, prefix)
        feats_by_k[k] = feats
        # collinearity diagnostic vs own lag-1 level
        own_lag1 = df.sort_values(harness.ID_COLS).groupby(
            "cma_canonical")[TARGET].shift(1).reindex(df.index)
        corr = {c: round(float(feats[c].corr(own_lag1)), 4)
                for c in feats.columns}
        print(f"  k={k}: corr(feature, own lag-1 level) = {corr}")
        feats.to_csv(OUT_DIR / f"spatial_features_k{k}.csv", index=True)

    # ---- CV runs -----------------------------------------------------------
    runs = {}
    specs = [
        ("T3.1-temporal-ridge", ridge_lags_model_fn, None,
         "Temporal-only reference block for T3.1: the T2.1 pooled "
         "Ridge(α=1)-on-lags + CMA dummies rerun unchanged so the spatial "
         "delta is computed on freshly paired identical folds/masks."),
    ]
    for k in (3, 5):
        specs.append((
            f"T3.1-spatial-k{k}", make_spatial_ridge_model_fn(feats_by_k[k]),
            k,
            f"Pooled Ridge(α=1)-on-lags + CMA dummies PLUS four k={k} "
            "neighbour-lag features (nbr lvl/dlt mean + distance-weighted "
            "variants; strictly backward shifts of neighbour actuals). "
            "Cheap-ablation test of whether kNN spatial-lag signal adds "
            "anything over temporal-only lags."))

    for run_id, fn, k, desc in specs:
        print(f"\n--- running {run_id} ---")
        res = evaluate(fn, df=df, folds=folds, target=TARGET)
        runs[run_id] = res
        agg = res["agg"]
        print("  model      : " +
              "  ".join(f"{m}={fmt(agg['model'][m]['mean'], agg['model'][m]['std'])}"
                        for m in harness.METRIC_NAMES))
        print("  persistence: " +
              "  ".join(f"{m}={fmt(agg['baseline'][m]['mean'], agg['baseline'][m]['std'])}"
                        for m in harness.METRIC_NAMES))
        cfg_extra = {"model": "ridge_on_lags+spatial" if k else
                     "ridge_on_lags", "alpha": RIDGE_ALPHA,
                     "spatial_k": k,
                     "spatial_features": (list(feats_by_k[k].columns)
                                          if k else [])}
        save_results_json(res, OUT_DIR / f"{run_id}_results.json")
        if not args.no_log:
            log_run(run_id=run_id, task="T3", sub="T3.1", description=desc,
                    config={"git_sha_pre_commit": sha, **cfg_extra},
                    results=res)

    # ---- Paired intersection-mask deltas (DECISION RULE inputs) -------------
    print("\n--- paired intersection-mask deltas: spatial vs temporal-only "
          "---")
    delta_rows = []
    paired_summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
                      "git_sha_pre_commit": sha,
                      "decision_rule": "KEEP iff paired ΔMDA(intersection) > 0 "
                                       "on a majority of folds AND mean paired "
                                       "ΔMDA > 0 AND full-mask aggregate NRMSE "
                                       "<= 1.10 x temporal-only",
                      "per_k": {}}
    for k in (3, 5):
        per_fold = []
        detail = []
        for f in folds:
            train_df = df.loc[f["train_index"]]
            val_df = df.loc[f["val_index"]]
            tr_pred_fn = ridge_lags_model_fn(train_df.drop(columns=[TARGET]),
                                             train_df[TARGET])
            sp_model_fn = make_spatial_ridge_model_fn(feats_by_k[k])
            sp_pred_fn = sp_model_fn(train_df.drop(columns=[TARGET]),
                                     train_df[TARGET])
            tr_pred = pd.Series(tr_pred_fn(val_df.drop(columns=[TARGET])),
                                index=val_df.index)
            sp_pred = pd.Series(sp_pred_fn(val_df.drop(columns=[TARGET])),
                                index=val_df.index)
            base_pred = harness.persistence_predict(train_df, val_df, TARGET)

            common = (val_df[TARGET].notna() & tr_pred.notna()
                      & sp_pred.notna() & base_pred.notna())
            ve = val_df.loc[val_df.index[common]]
            paired_st = harness.paired_intersection_scores(
                train_df, ve, {"spatial": sp_pred.loc[ve.index],
                               "temporal_only": tr_pred.loc[ve.index]},
                TARGET)
            paired_sp = harness.paired_intersection_scores(
                train_df, ve,
                {"spatial_vs_persistence_model": sp_pred.loc[ve.index],
                 "persistence": base_pred.loc[ve.index]}, TARGET)

            d_mda = (paired_st["per_predictor"]["spatial"]["MDA"]
                     - paired_st["per_predictor"]["temporal_only"]["MDA"])
            per_fold.append({
                "fold_id": f["fold_id"],
                "n_eval_rows": int(common.sum()),
                "model": paired_st["per_predictor"]["spatial"],
                "baseline": paired_st["per_predictor"]["temporal_only"],
            })
            detail.append({
                "fold_id": f["fold_id"],
                "n_directions_intersection":
                    paired_st["counts"]["n_scored_directions_intersection"],
                "spatial_MDA": paired_st["per_predictor"]["spatial"]["MDA"],
                "temporal_MDA":
                    paired_st["per_predictor"]["temporal_only"]["MDA"],
                "delta_MDA_spatial_minus_temporal": d_mda,
                "spatial_NRMSE": paired_st["per_predictor"]["spatial"]["NRMSE"],
                "temporal_NRMSE":
                    paired_st["per_predictor"]["temporal_only"]["NRMSE"],
                "spatial_vs_persistence_delta_MDA":
                    paired_sp["per_predictor"]["spatial_vs_persistence_model"]["MDA"]
                    - paired_sp["per_predictor"]["persistence"]["MDA"],
            })
            print(f"  k={k} fold {f['fold_id']}: ΔMDA = {d_mda:+.4f}")

        agg = harness._aggregate(per_fold)
        n_pos = sum(1 for d in detail
                    if d["delta_MDA_spatial_minus_temporal"] > 0)
        # like-for-like point-error ratio: FULL-mask CV aggregates of the two
        # evaluate() runs (identical folds AND identical row masks)
        nrmse_ratio = (
            runs[f"T3.1-spatial-k{k}"]["agg"]["model"]["NRMSE"]["mean"]
            / runs["T3.1-temporal-ridge"]["agg"]["model"]["NRMSE"]["mean"])
        mean_delta_mda = float(np.mean(
            [d["delta_MDA_spatial_minus_temporal"] for d in detail]))
        keep = bool(n_pos > len(detail) / 2 and nrmse_ratio <= 1.10
                    and mean_delta_mda > 0)
        paired_summary["per_k"][k] = {
            "agg": agg, "detail": detail,
            "folds_with_positive_delta_MDA": n_pos,
            "mean_paired_delta_MDA": mean_delta_mda,
            "aggregate_full_mask_NRMSE_ratio_vs_temporal":
                round(nrmse_ratio, 4),
            "verdict_keep_for_T6": keep,
        }
        print(f"  k={k}: ΔMDA>0 on {n_pos}/{len(detail)} folds "
              f"(mean {mean_delta_mda:+.4f}); full-mask NRMSE ratio "
              f"{nrmse_ratio:.3f} -> {'KEEP' if keep else 'DROP'}")
        log_run(
            run_id=f"T3.1-intersect-k{k}-vs-temporal", task="T3", sub="T3.1",
            description=f"PAIRED intersection-mask comparison for the T3.1 "
                        f"decision rule: ridge-on-lags WITH k={k} "
                        "neighbour-lag features vs temporal-only ridge on one "
                        "shared nonzero-direction set per fold; also records "
                        "spatial-vs-persistence paired ΔMDA.",
            config={"git_sha_pre_commit": sha,
                    "scoring": "paired intersection-mask",
                    "comparison": f"spatial-k{k} vs temporal-only ridge",
                    "alpha": RIDGE_ALPHA,                     "spatial_k": k},
            results={"folds": [{"fold_id": fl["fold_id"]} for fl in folds],
                     "per_fold": per_fold, "agg": agg,
                     "config": {"scoring": "paired intersection-mask"}},
        ) if not args.no_log else print("  [--no-log] intersect row skipped")
        delta_rows.extend([{**d, "k": k} for d in detail])

    delta_tab = pd.DataFrame(delta_rows)[
        ["k", "fold_id", "n_directions_intersection", "spatial_MDA",
         "temporal_MDA", "delta_MDA_spatial_minus_temporal", "spatial_NRMSE",
         "temporal_NRMSE", "spatial_vs_persistence_delta_MDA"]]
    delta_tab.to_csv(OUT_DIR / "paired_delta_table.csv", index=False)
    save_results_json(paired_summary, OUT_DIR / "decision_rule.json")

    # ---- Plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, 6),
            [pf["baseline"]["MDA"] for pf in runs["T3.1-temporal-ridge"]["per_fold"]],
            marker="o", label="persistence (same mask)", color="tab:gray")
    ax.plot(range(1, 6),
            [pf["model"]["MDA"] for pf in runs["T3.1-temporal-ridge"]["per_fold"]],
            marker="o", label="temporal-only ridge", color="tab:blue")
    colors = {"3": "tab:green", "5": "tab:red"}
    for k in (3, 5):
        ax.plot(range(1, 6),
                [pf["model"]["MDA"]
                 for pf in runs[f"T3.1-spatial-k{k}"]["per_fold"]],
                marker="o", label=f"+ neighbour-lags k={k}",
                color=colors[str(k)])
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("fold (val window)")
    ax.set_ylabel("boundary-anchored zero-drop MDA")
    ax.set_title("T3.1 — kNN neighbour-lag ablation (canonical folds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "t3_1_spatial_ablation.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[harness] saved plot -> {plot_path}")

    print("\n=== T3.1 spatial-feature ablation complete ===")


if __name__ == "__main__":
    main()
