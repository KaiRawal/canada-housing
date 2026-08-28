#!/usr/bin/env python3
"""
T4.4 Leaderboard builder — consolidates T4.1–T4.3 on identical 5-fold expanding CV.
- Dedupes runs.jsonl by latest (id, date) and handles differing-ID trap T4.2-rf-A vs T4.2-RF-A.
- Produces unified leaderboard CSV/JSON with gate eligibility, per-fold variance, shortlist.
- No model fits, no test-set access.
"""
import json
import pathlib
import collections
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "runs.jsonl"
ART_T44 = ROOT / "artifacts" / "T4.4"
ART_T41 = ROOT / "artifacts" / "T4.1"
ART_T42 = ROOT / "artifacts" / "T4.2"
ART_T43 = ROOT / "artifacts" / "T4.3"

# Canonical persistence
PERSIST_MDA = 0.6771279882839918
PERSIST_NRMSE = 0.0011305311297628712
B_ENET_MDA = 0.5980835984449133
B_ENET_NRMSE = 0.0010752020172069055
C_ENET_NRMSE = 0.0010727975947863121  # actual min among all T4 combos

CODIFIED_GLOBAL_BEST_NRMSE = B_ENET_NRMSE  # per T4.4 plan codified 2026-08-28
GATE_RATIO = 1.10
GATE_THRESHOLD = CODIFIED_GLOBAL_BEST_NRMSE * GATE_RATIO
# For adjudication also compute persistence-relative threshold
PERSIST_THRESHOLD = PERSIST_NRMSE * GATE_RATIO

def load_runs_deduped():
    rows = []
    for line in open(RUNS):
        line=line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    # dedupe by id keeping latest date
    latest = {}
    for r in rows:
        rid = r.get("id")
        date = r.get("date","")
        # also consider timestamp if present
        ts = r.get("timestamp", date)
        # use date string comparison (ISO)
        key = rid
        prev = latest.get(key)
        if prev is None or ts >= prev.get("timestamp", prev.get("date","")):
            latest[key]=r
    # handle differing-ID trap: T4.2-rf-A vs T4.2-RF-A => keep only uppercase latest
    if "T4.2-rf-A" in latest and "T4.2-RF-A" in latest:
        # lowercase is draft; exclude from authoritative
        # record trap but exclude
        pass
    return latest

def is_authoritative_combo(rid):
    if rid in ("T4.1-criticfix-evidence", "T4.1-B-enet-niter-proof"):
        return False
    if rid == "T4.2-rf-A":
        return False  # differing-ID draft
    # include only T4.1, T4.2, T4.3 combos that are model runs (have metrics_agg)
    if not rid.startswith("T4."):
        return False
    # filter sub T4.1–T4.3
    if rid.startswith("T4.1-") or rid.startswith("T4.2-") or rid.startswith("T4.3-"):
        return True
    return False

def build_leaderboard():
    latest = load_runs_deduped()
    # collect authoritative T4 combos
    combos = []
    for rid, r in latest.items():
        if not is_authoritative_combo(rid):
            continue
        # skip if no metrics_agg
        if "metrics_agg" not in r:
            continue
        m = r["metrics_agg"]
        # some rows have MDA mean etc nested; older rows may have different structure
        # metrics_agg is dict with keys MDA, NMSE, NRMSE, NMAE each with mean/std
        try:
            mda_mean = m["MDA"]["mean"]
            mda_std = m["MDA"]["std"]
            nrmse_mean = m["NRMSE"]["mean"]
            nrmse_std = m["NRMSE"]["std"]
            nmse_mean = m["NMSE"]["mean"] if "NMSE" in m else np.nan
            nmae_mean = m["NMAE"]["mean"] if "NMAE" in m else np.nan
        except Exception as e:
            continue
        # feature_set from config
        cfg = r.get("config",{})
        feat = cfg.get("feature_set") or cfg.get("feature_set_name") or ""
        # fallback: parse from id
        if not feat:
            if "-A" in rid and "B" not in rid and "C" not in rid:
                feat = "A_base_lag1"
            elif "-B" in rid:
                feat = "B_base_lag1_OWN-subset"
            elif "-C" in rid:
                feat = "C_base_lag1_OWN_nbr5"
        combos.append({
            "model_id": rid,
            "task": r.get("task"),
            "sub": r.get("sub"),
            "date": r.get("date"),
            "timestamp": r.get("timestamp"),
            "feature_set": feat,
            "MDA_mean": mda_mean,
            "MDA_std": mda_std,
            "NRMSE_mean": nrmse_mean,
            "NRMSE_std": nrmse_std,
            "NMSE_mean": nmse_mean,
            "NMAE_mean": nmae_mean,
            "config": cfg,
        })
    df = pd.DataFrame(combos)
    # compute derived columns
    df["delta_MDA_vs_persistence"] = df["MDA_mean"] - PERSIST_MDA
    df["NRMSE_ratio_vs_codified_best"] = df["NRMSE_mean"] / CODIFIED_GLOBAL_BEST_NRMSE
    df["NRMSE_ratio_vs_actual_min"] = df["NRMSE_mean"] / C_ENET_NRMSE
    df["NRMSE_ratio_vs_persistence"] = df["NRMSE_mean"] / PERSIST_NRMSE
    df["gate_eligible_codified"] = df["NRMSE_mean"] <= GATE_THRESHOLD
    df["gate_eligible_actual_min"] = df["NRMSE_mean"] <= (C_ENET_NRMSE*GATE_RATIO)
    df["gate_eligible_persistence"] = df["NRMSE_mean"] <= PERSIST_THRESHOLD
    # overall rank by NRMSE then MDA? Plan says rank within eligible and overall; compute per-fold variance (std, folds-positive counts already in artifacts) — no new model fits needed
    # Rank overall by MDA descending? But plan says "Rank within eligible and overall" - overall rank by MDA for table presentation
    df = df.sort_values(by=["gate_eligible_codified","MDA_mean","NRMSE_mean"], ascending=[False, False, True], na_position="last")
    # Actually for leaderboard display we want overall ranked by MDA descending but flag eligibility
    df_overall = df.sort_values(by="MDA_mean", ascending=False).reset_index(drop=True)
    df_overall["rank_overall_MDA"] = np.arange(1, len(df_overall)+1)
    df_overall["rank_overall_NRMSE"] = df_overall["NRMSE_mean"].rank(method="min").astype(int)
    # rank within eligible by MDA
    eligible = df_overall[df_overall["gate_eligible_codified"]].copy()
    eligible = eligible.sort_values(by="MDA_mean", ascending=False).reset_index(drop=True)
    eligible["rank_eligible_MDA"] = np.arange(1, len(eligible)+1)
    # map back
    df_overall["rank_eligible_MDA"] = df_overall["model_id"].map(dict(zip(eligible["model_id"], eligible["rank_eligible_MDA"])))
    # shortlist: top 2 within pruned A/B scope and gate-eligible
    # Pruned scope: A/B only (exclude C). Also exclude classical? classical A but with different denominator - keep but flag.
    # Plan says shortlist top 2: by rule T4.1 B-enet remains frontrunner; second is best eligible runner-up (by MDA within eligible — likely RF-A 0.615 but its NRMSE 0.00135 fails gate, so next eligible is maybe recursive-LGBM-B 0.594 or A-enet if gate relaxed — apply rule strictly and document). If no second eligible, shortlist is single + "none".
    # So filter to A/B scope eligible:
    pruned_ids = df_overall[df_overall["feature_set"].str.contains("A_base|B_base", na=False) | df_overall["model_id"].str.contains("SARIMAX|VAR")]
    # But for shortlist we consider only A/B linear/tree/recursive (exclude classical? classical is A but with different denominator — keep as not comparable)
    # We'll consider only non-classical A/B combos for shortlist
    shortlist_candidates = df_overall[
        df_overall["gate_eligible_codified"] &
        df_overall["feature_set"].str.contains("A_base|B_base", na=False) &
        ~df_overall["model_id"].isin(["T4.3-SARIMAX-110-A","T4.3-VAR-1-A"])
    ].sort_values(by="MDA_mean", ascending=False)
    # If C-enet is eligible but not in pruned scope, it should not be shortlisted for T5 (pruned scope A/B only)
    # So shortlist will be B-enet first, then next eligible in pruned scope (if any)
    shortlist = shortlist_candidates["model_id"].tolist()
    # B-enet should be among them (it is not top MDA but is eligible). Sorting by MDA puts A-enet first but A-enet is ineligible under codified gate. So shortlist_candidates currently sorted by MDA would have B-enet as maybe 2nd? Let's check:
    # Among eligible + A/B, only B-enet (0.598) and maybe? Actually no other A/B combo passes 0.00118. So shortlist_candidates will contain only B-enet (and C-enet excluded). So length 1 -> shortlist is ["T4.1-B-enet", "none"]
    if len(shortlist) >= 2:
        top2 = shortlist[:2]
    elif len(shortlist)==1:
        top2 = [shortlist[0], "none"]
    else:
        top2 = ["none","none"]

    return df_overall, eligible, shortlist_candidates, top2, latest

def main():
    ART_T44.mkdir(parents=True, exist_ok=True)
    df_overall, eligible, shortlist_candidates, top2, latest = build_leaderboard()

    # Save leaderboard.csv
    cols = ["rank_overall_MDA","rank_eligible_MDA","model_id","feature_set","MDA_mean","MDA_std","NRMSE_mean","NRMSE_std","NMSE_mean","NMAE_mean",
            "delta_MDA_vs_persistence","NRMSE_ratio_vs_codified_best","NRMSE_ratio_vs_persistence","gate_eligible_codified","gate_eligible_persistence"]
    out = df_overall[cols].copy()
    out.to_csv(ART_T44 / "leaderboard.csv", index=False)

    # Load paired deltas for per-fold variance enrichment
    per_fold_info = {}
    # Try to load paired_deltas for T4.1/T4.2 and per_fold JSONs for T4.3
    for f in [ART_T41/"paired_deltas.json", ART_T42/"paired_deltas.json"]:
        if f.exists():
            d=json.load(open(f))
            for combo, vals in d.get("per_combo",{}).items():
                per_fold_info[combo] = vals

    # Build leaderboard.json with full detail
    leaderboard_json = {
        "generated_at": datetime.utcnow().isoformat()+"+00:00",
        "canonical_persistence": {"MDA_mean": PERSIST_MDA, "NRMSE_mean": PERSIST_NRMSE},
        "codified_gate": {
            "global_best_id": "T4.1-B-enet",
            "global_best_NRMSE": CODIFIED_GLOBAL_BEST_NRMSE,
            "actual_min_NRMSE": C_ENET_NRMSE,
            "actual_min_id": "T4.1-C-enet",
            "threshold_codified": GATE_THRESHOLD,
            "threshold_actual_min": C_ENET_NRMSE*GATE_RATIO,
            "threshold_persistence": PERSIST_THRESHOLD,
            "ratio_note": "Codified threshold uses B-enet 0.0010752; actual min C-enet 0.0010728 gives threshold 0.0011801 — both yield same eligible set (B-enet, C-enet)."
        },
        "adjudication_A_enet": {
            "model_id": "T4.1-A-enet",
            "MDA": 0.6490319330205947,
            "NRMSE": 0.0011896614990103708,
            "ratio_vs_codified": 0.00118966/CODIFIED_GLOBAL_BEST_NRMSE,
            "ratio_vs_actual_min": 0.00118966/C_ENET_NRMSE,
            "ratio_vs_persistence": 0.00118966/PERSIST_NRMSE,
            "eligible_codified": 0.00118966 <= GATE_THRESHOLD,
            "eligible_persistence": 0.00118966 <= PERSIST_THRESHOLD,
            "decision": "Directional best (MDA 0.649, Δ vs persistence -0.0017, 1/5 folds) but INELIGIBLE under codified global-best gate (1.106× codified, 1.109× actual_min); WOULD be eligible under persistence-relative 10% gate (1.052× vs 0.0011305, threshold 0.0012436). Flagged for researcher adjudication; not shortlisted for T5 under strict rule."
        },
        "eligible_combos_codified": eligible["model_id"].tolist(),
        "shortlist_for_T5_pruned_AB_scope": top2,
        "rows": df_overall.to_dict(orient="records"),
    }
    with open(ART_T44/"leaderboard.json","w") as f:
        json.dump(leaderboard_json, f, indent=2)

    # per_fold_variance.json
    variance = {}
    for _, row in df_overall.iterrows():
        rid=row["model_id"]
        variance[rid]={
            "MDA_mean": row["MDA_mean"],
            "MDA_std": row["MDA_std"],
            "NRMSE_mean": row["NRMSE_mean"],
            "NRMSE_std": row["NRMSE_std"],
            "delta_MDA_vs_persistence": row["delta_MDA_vs_persistence"],
            "NRMSE_ratio_vs_codified_best": row["NRMSE_ratio_vs_codified_best"],
            "gate_eligible_codified": bool(row["gate_eligible_codified"]),
        }
        # add paired folds-positive if available
        if rid in per_fold_info:
            variance[rid]["paired_folds_positive_vs_persistence"] = per_fold_info[rid].get("folds_positive_vs_persistence")
            variance[rid]["paired_mean_delta_MDA_vs_persistence"] = per_fold_info[rid].get("mean_delta_MDA_vs_persistence")
        # for T4.3, infer from per_fold JSONs if not in paired
        if rid.startswith("T4.3-"):
            # load corresponding result JSON
            jpath = ART_T43 / f"{rid}_results.json"
            if jpath.exists():
                d=json.load(open(jpath))
                per_fold = d.get("per_fold",[])
                # compute per-fold MDA deltas vs baseline (paired on same mask for direct/recursive)
                deltas = [pf["model"]["MDA"] - pf["baseline"]["MDA"] for pf in per_fold]
                wins = sum(1 for dd in deltas if dd>0)
                variance[rid]["per_fold_MDA_deltas_vs_own_baseline"] = deltas
                variance[rid]["folds_positive_vs_own_baseline"] = f"{wins}/5"
                variance[rid]["mean_delta_vs_own_baseline"] = float(np.mean(deltas))
                # for classical, also compute vs global persistence (agg-only, not paired)
                if rid in ("T4.3-SARIMAX-110-A","T4.3-VAR-1-A"):
                    variance[rid]["note"] = "Classical scored on 245 rows in fold5 (6 late-starter CMAs excluded) vs 317 for direct/recursive; baseline reported is own-baseline 0.67205±0.0395 on 245-row denominator, not global 0.67712±0.0474 on 317-row denominator. Δ vs global persistence is agg-only, not paired common-mask."
                    # fold5 global pers MDA = 0.7545 on 317 rows
                    global_pers_per_fold = [0.7040816326530612,0.6607142857142857,0.6196319018404908,0.6466666666666666,0.7545454545454545]
                    deltas_global = [pf["model"]["MDA"] - g for pf,g in zip(per_fold, global_pers_per_fold)]
                    # but for fold5, n_eval differs, so flag as not paired
                    variance[rid]["per_fold_MDA_deltas_vs_global_persistence_agg_only"] = deltas_global
                    variance[rid]["folds_positive_vs_global_persistence_agg_only"] = f"{sum(1 for dd in deltas_global if dd>0)}/5"
                # for recursive-LGBM-B specifically
                if rid=="T4.3-recursive-LGBM-B":
                    # also compute vs direct-LGBM-B
                    direct_path = ART_T43 / "T4.3-direct-LGBM-B_results.json"
                    if direct_path.exists():
                        dd=json.load(open(direct_path))
                        deltas_vs_direct=[rr["model"]["MDA"]-dr["model"]["MDA"] for rr,dr in zip(per_fold, dd.get("per_fold",[]))]
                        variance[rid]["per_fold_MDA_deltas_vs_direct"] = deltas_vs_direct
                        variance[rid]["folds_positive_vs_direct"] = f"{sum(1 for d in deltas_vs_direct if d>0)}/5"
                        variance[rid]["mean_delta_vs_direct"] = float(np.mean(deltas_vs_direct))

    with open(ART_T44/"per_fold_variance.json","w") as f:
        json.dump(variance, f, indent=2)

    # Plot leaderboard
    plt.figure(figsize=(10,6))
    # Sort by MDA descending for plot
    plot_df = df_overall.sort_values(by="MDA_mean", ascending=True)  # horizontal bar
    y = np.arange(len(plot_df))
    colors = ["#2ca02c" if e else "#1f77b4" for e in plot_df["gate_eligible_codified"]]
    plt.barh(y, plot_df["MDA_mean"], xerr=plot_df["MDA_std"], color=colors, ecolor="black", capsize=3, alpha=0.7)
    plt.yticks(y, plot_df["model_id"], fontsize=7)
    plt.axvline(PERSIST_MDA, color="red", linestyle="--", label="persistence 0.6771")
    plt.xlabel("MDA mean ± std (5 folds)")
    plt.title("T4 Leaderboard: MDA vs persistence (green = gate-eligible NRMSE ≤1.10× B-enet 0.001075)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART_T44/"t4_leaderboard.png", dpi=180)
    plt.close()

    # Second plot: NRMSE
    plt.figure(figsize=(10,6))
    plot_df2 = df_overall.sort_values(by="NRMSE_mean", ascending=True)
    y2 = np.arange(len(plot_df2))
    colors2 = ["#2ca02c" if e else "#1f77b4" for e in plot_df2["gate_eligible_codified"]]
    plt.barh(y2, plot_df2["NRMSE_mean"], xerr=plot_df2["NRMSE_std"], color=colors2, ecolor="black", capsize=3, alpha=0.7)
    plt.yticks(y2, plot_df2["model_id"], fontsize=7)
    plt.axvline(PERSIST_NRMSE, color="red", linestyle="--", label="persistence 0.00113")
    plt.axvline(GATE_THRESHOLD, color="orange", linestyle=":", label="gate 0.00118 (1.10× B-enet)")
    plt.xlabel("NRMSE mean ± std")
    plt.title("T4 Leaderboard: NRMSE (green = eligible)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART_T44/"per_fold_variance.png", dpi=180)
    plt.close()

    # decision_rule.json for T4.4
    decision = {
        "generated_at": datetime.utcnow().isoformat()+"+00:00",
        "gate": {
            "rule": "winner = best mean MDA subject to mean NRMSE ≤1.10× global-best NRMSE",
            "global_best_id": "T4.1-B-enet",
            "global_best_NRMSE": CODIFIED_GLOBAL_BEST_NRMSE,
            "threshold": GATE_THRESHOLD,
            "actual_min_id": "T4.1-C-enet",
            "actual_min_NRMSE": C_ENET_NRMSE,
            "actual_min_threshold": C_ENET_NRMSE*GATE_RATIO,
            "note": "Both codified (B-enet) and actual-min (C-enet) thresholds give same eligible set: {B-enet, C-enet}. A-enet 0.649/1.109× excluded under global gate but 1.052× vs persistence — would be eligible under persistence-relative gate (0.0012436). Adjudicated as directional best but ineligible strict."
        },
        "eligible_combos": eligible["model_id"].tolist(),
        "eligible_detail": eligible[["model_id","MDA_mean","NRMSE_mean","NRMSE_ratio_vs_codified_best"]].to_dict(orient="records") if not eligible.empty else [],
        "shortlist_for_T5": {
            "scope": "pruned A/B only (B = A + OWN-subset tot_d12/lag12/24/dlog1); C/k5 excluded per T4.1, SCSS/RMS dropped",
            "top2": top2,
            "rationale": "Only B-enet passes codified gate in A/B pruned scope; no second eligible exists, so shortlist is single + none. If gate relaxed to persistence-relative, A-enet would be runner-up (MDA 0.649). Among T4.2/4.3 A/B combos none pass gate; best ineligible runner-up by MDA is RF-A 0.6151 (1.26×), then recursive-LGBM-B 0.5942 (1.30×).",
            "margins": {
                "B_enet": {"MDA": B_ENET_MDA, "NRMSE": B_ENET_NRMSE, "vs_persistence_MDA": B_ENET_MDA - PERSIST_MDA, "vs_persistence_NRMSE_ratio": B_ENET_NRMSE/PERSIST_NRMSE},
                "second_eligible": None if top2[1]=="none" else top2[1]
            }
        },
        "per_fold_variance_note": "All deltas are on identical 5-fold expanding CV (2008-09…2016-17) grouped by CMA; classical SARIMAX/VAR scored on 245 rows in fold5 (6 late-starters excluded) vs 317 for direct/recursive — their baseline is own-baseline 0.67205±0.0395 on 245 denominator vs global 0.67712±0.0474 on 317 denominator; Δ vs global pers is agg-only, not paired common-mask. Recursive-LGBM-B 4/5 wins vs direct (+0.0371/+0.0188/+0.0231/+0.0283/-0.0225) but still -0.0829 vs persistence and -0.0039 vs B-enet.",
        "caveats": [
            "No model beats persistence MDA 0.6771±0.0474 on paired 5-fold CV; multi-metric bar (direction + point-error class) unmet.",
            "Partial recursion: 176 lag-1 covariate columns use observed val row lag-1 values; only OWN-subset (tot_d12/lag12/24/dlog1) recomputed from predicted history — isolates target-memory recursion effect; future covariate forecasts beyond horizon not needed.",
            "Recursive-Ridge-B harm: NRMSE 0.00373→0.00750 (2× worse, fold5 0.01347→0.03198); LGBM-B tie (0.00139→0.00140) shows GBM-safe but no gain.",
            "Classical denominators: SARIMAX/VAR n_eval_rows fold5 =245 vs direct/recursive 317; 323 is n_val_rows pre-common-mask (24 CMAs × partial histories). Reported baselines differ accordingly.",
            "n_iter_ range for T4.1 B-enet proof is 62–3467 (chosen_per_fold [3467,588,62,148,137]), not 588–3467; Ridge n_iter_ is None as expected (closed-form).",
            "Bounded-search caveat: 'no tree/recursion beats linear ceiling' conditional on modest breadth (RF 2 combos 30 trees, GBMs 4 combos each, LGBM-deep 4 combos, LightGBM recursion 4 combos) and small para grid — wider search not explored.",
            "Versions caveat: sklearn 1.9.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, statsmodels 0.14.6, skforecast not_installed_fallback_manual_recursion.",
            "Bounded-search/versions caveats from T4.3 fixes carried forward.",
            "Upstream imputation panels stay canonical with ≤~1% optimism caveat (T3.3 hard gate); A/B only pruned scope.",
            "BC: All T4.1–T4.3 used LEVEL target only, same harness folds, same harness.evaluate path; test split never accessed."
        ],
        "versions": {"sklearn": "1.9.0", "lightgbm": "4.7.0", "xgboost": "3.4.1", "catboost": "1.2.10", "statsmodels": "0.14.6", "skforecast": "not_installed_fallback_manual_recursion"},
        "artifacts_generated": ["leaderboard.csv","leaderboard.json","per_fold_variance.json","per_fold_variance.png","t4_leaderboard.png","decision_rule.json"]
    }
    with open(ART_T44/"decision_rule.json","w") as f:
        json.dump(decision, f, indent=2)

    print("Leaderboard built:")
    print(df_overall[["model_id","MDA_mean","NRMSE_mean","gate_eligible_codified"]].to_string())
    print("Eligible:", eligible["model_id"].tolist())
    print("Top2:", top2)

if __name__=="__main__":
    main()
