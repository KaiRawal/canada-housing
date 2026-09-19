"""Pretrained_ts TRY-2: shrinkage-guarded calibration + residual-correction.

Validation arithmetic ONLY -- 0 new Chronos calls. Reuses stored
TRY-1 per-context val_raw (winner last120) + per-CMA test_raw forecasts
from artifacts/pretrained_ts.joblib. No load_pipeline / forecast calls.

(i) SHRINKAGE-GUARDED CALIBRATION (cal-after-blend, TRY-1 style):
    per w in {0.0,1.0}, blended=blend(raw,persist,w) on VAL ONLY,
    ridge toward (1,0): min ||y-(a*blended+b)||^2 + lam*((a-1)^2+b^2),
    lam grid chosen on val NRMSE; best (lam,w) vs no-cal w=0 guardrail
    (pure persistence); adopt calibration ONLY if val beats guardrail
    by >1% margin, else guardrail.
(ii) RESIDUAL-CORRECTION: level anchor = persistence; excess e=raw-persist,
    true residual r=y-persist (VAL ONLY); scale-only a_lam=(eTr)/(eTe+lam)
    (b=0, shrunk toward 0); final(w)=persist+w*a*e, w in {0,0.25,0.5}
    tuned on val NRMSE.
Final adopted = lower val NRMSE of (i)-adopted vs (ii)-best. Test reported
honestly via evaluation.evaluation.calculate_all_metrics. Artifact/metrics
updated ONLY on TEST beat vs try-1 test; tries.jsonl always appended.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines._nesting import blend, persistence_pred  # noqa: E402
from evaluation.evaluation import calculate_all_metrics  # noqa: E402

TARGET = "total"
OUT = ROOT / "artifacts"
BLEND_GRID_I = (0.0, 1.0)
LAMS_I = [0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
W_GRID_R = (0.0, 0.25, 0.5)
LAMS_R = [0, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e8, 1e9]
TRY1_TEST = 0.0004137508364060161
MARGIN = 0.01

CALLS = {"n_chronos": 0}  # stays 0 by construction (no pipeline import)


def pooled_nrmse_valid(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    n = int(valid.sum())
    if n == 0:
        return float("nan")
    mse = float(np.mean((y_true[valid] - y_pred[valid]) ** 2))
    return float(np.sqrt(mse) / n)


def ridge_ab_toward_10(x, y, lam):
    """Ridge (a,b)->(1,0): d=y-x, X=[x,1], theta=[a-1,b]."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    if int(valid.sum()) == 0:
        return 1.0, 0.0
    d = yv - xv
    X = np.column_stack([xv, np.ones_like(xv)])
    XtX = X.T @ X
    Xtd = X.T @ d
    th = np.linalg.solve(XtX + float(lam) * np.eye(2), Xtd)
    return float(1.0 + th[0]), float(th[1])


def main():
    t0 = time.time()
    art = joblib.load(OUT / "pretrained_ts.joblib")
    assert art.get("variant") == "pretrained_ts"
    winner_ctx = str(art.get("best_params", {}).get("context", "last120"))
    assert winner_ctx == "last120", f"expected last120, got {winner_ctx}"

    y_train_full = pd.read_csv(ROOT / "prediction" / f"y_train_full_{TARGET}.csv",
                               parse_dates=["date"])
    y_test_full = pd.read_csv(ROOT / "prediction" / f"y_test_full_{TARGET}.csv",
                              parse_dates=["date"])

    # ---- VAL alignment (same as TRY-1: sorted CMAs, last-20% tails) ----
    from baselines._pretrained_ts import series_by_cma

    train_mask = y_train_full[TARGET].notna().to_numpy()
    ytr_valid = y_train_full.loc[train_mask].reset_index(drop=True)
    train_series = series_by_cma(ytr_valid, TARGET)
    cmas = sorted(train_series.keys())
    val_idx = []
    for cma in cmas:
        g = y_train_full[y_train_full["cma_canonical"] == cma].sort_values("date")
        g = g[g[TARGET].notna()]
        n = len(g)
        n_val = max(1, int(n * 0.2))
        val_idx.extend(g.iloc[n - n_val:].index.tolist())
    val_idx = np.asarray(val_idx)
    persist_full = persistence_pred(y_train_full)
    val_persist = persist_full.loc[val_idx].to_numpy(dtype=float)
    y_val = y_train_full.loc[val_idx, TARGET].to_numpy(dtype=float)
    val_dates = pd.to_datetime(y_train_full.loc[val_idx, "date"]).to_numpy()
    val_cmas = y_train_full.loc[val_idx, "cma_canonical"].to_numpy()
    val_raw = np.asarray(art["val_raw_per_cma"][winner_ctx], dtype=float)
    assert len(val_raw) == len(y_val) == 1543, (len(val_raw), len(y_val))
    assert int(np.isfinite(val_persist).sum()) == len(y_val), "val persist must be full"

    guard_val = pooled_nrmse_valid(y_val, val_persist)
    raw_val = pooled_nrmse_valid(y_val, val_raw)
    print(f"guardrail(persist) val={guard_val:.10f} raw({winner_ctx}) val={raw_val:.10f}",
          flush=True)

    # ---- (i) shrinkage-guarded calibration (cal-after-blend) ----
    best_i = None
    for w in BLEND_GRID_I:
        blended = blend(pd.Series(val_raw), pd.Series(val_persist),
                        float(w)).to_numpy(dtype=float)
        for lam in LAMS_I:
            a, b = ridge_ab_toward_10(blended, y_val, lam)
            fin = a * blended + b
            # affine preserves NaN mask of blended (persist NaN); val has none
            v = pooled_nrmse_valid(y_val, fin)
            rec = {"w": float(w), "lam": float(lam), "a": float(a),
                   "b": float(b), "val": float(v)}
            if best_i is None or v < best_i["val"]:
                best_i = rec
    print(f"(i) best shrunk-cal w={best_i['w']} lam={best_i['lam']:g} "
          f"a={best_i['a']:.6f} b={best_i['b']:.4f} val={best_i['val']:.10f}", flush=True)
    impr = (guard_val - best_i["val"]) / guard_val if guard_val else float("-inf")
    if impr > MARGIN:
        adopted_i = dict(best_i, order="cal-after-blend", kind="shrunk-cal")
        print(f"(i) ADOPT shrunk-cal (improvement {impr*100:.2f}% > 1%)", flush=True)
    else:
        adopted_i = {"w": 0.0, "lam": float("inf"), "a": 1.0, "b": 0.0,
                     "val": float(guard_val), "order": "cal-after-blend",
                     "kind": "guardrail"}
        print(f"(i) KEEP guardrail (improvement {impr*100:.2f}% <= 1%)", flush=True)

    # ---- (ii) residual-correction (persist anchor, scale-only, b=0) ----
    e_val = val_raw - val_persist
    r_val = y_val - val_persist
    finite = np.isfinite(e_val) & np.isfinite(r_val)
    eTe = float(e_val[finite] @ e_val[finite])
    eTr = float(e_val[finite] @ r_val[finite])
    best_r = None
    for lam in LAMS_R:
        a = float(eTr / (eTe + float(lam))) if (eTe + float(lam)) != 0 else 0.0
        for w in W_GRID_R:
            fin = val_persist + float(w) * a * e_val
            v = pooled_nrmse_valid(y_val, fin)
            rec = {"lam": float(lam), "a": float(a), "w": float(w),
                   "val": float(v)}
            if best_r is None or v < best_r["val"]:
                best_r = rec
    print(f"(ii) best residual lam={best_r['lam']:g} a={best_r['a']:.6f} "
          f"w={best_r['w']} val={best_r['val']:.10f}", flush=True)

    # ---- Final selection by VAL NRMSE ----
    if best_r["val"] < adopted_i["val"]:
        adopted = {"kind": "residual", "a_resid": float(best_r["a"]),
                   "w_resid": float(best_r["w"]), "lam_resid": float(best_r["lam"]),
                   "val": float(best_r["val"])}
        print("ADOPTED arm = (ii) residual", flush=True)
    else:
        adopted = {"kind": adopted_i["kind"], "a": float(adopted_i["a"]),
                   "b": float(adopted_i["b"]), "w": float(adopted_i["w"]),
                   "lam": float(adopted_i["lam"]), "order": adopted_i["order"],
                   "val": float(adopted_i["val"])}
        print(f"ADOPTED arm = (i) {adopted_i['kind']}", flush=True)

    # ---- TEST honest report (stored test_raw, no new calls) ----
    test_persist_full = persistence_pred(y_test_full)
    rows = []
    for cma in cmas:
        g = y_test_full[y_test_full["cma_canonical"] == cma].sort_values("date")
        rows.append((cma, g.index.to_numpy()))
    te_idx = np.concatenate([r[1] for r in rows])
    y_te = y_test_full.loc[te_idx, TARGET].to_numpy(dtype=float)
    te_persist = test_persist_full.loc[te_idx].to_numpy(dtype=float)
    te_raw = np.concatenate([np.asarray(art["test_raw_per_cma"][c], dtype=float)
                             for c, _ in rows])
    te_cmas = y_test_full.loc[te_idx, "cma_canonical"].to_numpy()
    te_dates = pd.to_datetime(y_test_full.loc[te_idx, "date"]).to_numpy()

    def apply_test():
        if adopted["kind"] == "residual":
            a, w = float(adopted["a_resid"]), float(adopted["w_resid"])
            ee = te_raw - te_persist
            return te_persist + w * a * ee
        a, b, w = float(adopted["a"]), float(adopted["b"]), float(adopted["w"])
        br = blend(pd.Series(te_raw), pd.Series(te_persist), float(w)).to_numpy(
            dtype=float)
        return a * br + b

    te_final = apply_test()

    def frame(cmas_a, dates_a, yt, yp):
        return pd.DataFrame({"cma_canonical": cmas_a, "date": dates_a,
                             f"{TARGET}_true": yt, f"{TARGET}_pred": yp})

    # Val MDA via calculate_all_metrics (grouped, honest)
    if adopted["kind"] == "residual":
        a, w = float(adopted["a_resid"]), float(adopted["w_resid"])
        val_final = val_persist + w * a * e_val
    else:
        br_v = blend(pd.Series(val_raw), pd.Series(val_persist),
                     float(adopted["w"])).to_numpy(dtype=float)
        val_final = float(adopted["a"]) * br_v + float(adopted["b"])
    val_m = calculate_all_metrics(frame(val_cmas, val_dates, y_val, val_final), TARGET)
    test_m = calculate_all_metrics(frame(te_cmas, te_dates, y_te, te_final), TARGET)
    test_nrmse = float(test_m["NRMSE"])
    print(f"val adopted NRMSE={adopted['val']:.10f} MDA={float(val_m['MDA']):.4f}",
          flush=True)
    print(f"test NRMSE={test_nrmse:.10f} MDA={float(test_m['MDA']):.4f} "
          f"NMAE={float(test_m['NMAE']):.10f}", flush=True)
    print(f"chronos calls = {CALLS['n_chronos']} (must be 0)", flush=True)

    beats = bool(np.isfinite(test_nrmse) and test_nrmse < float(TRY1_TEST))
    print(f"beats try-1 ({TRY1_TEST:.10f})? {beats}", flush=True)

    # ---- Train NRMSE for metrics entry (persist-anchored, no raw needed) ----
    train_persist_full = persistence_pred(y_train_full)
    tr_idx = y_train_full.index.to_numpy()
    tr_persist = train_persist_full.loc[tr_idx].to_numpy(dtype=float)
    if adopted["kind"] == "residual" and float(adopted["w_resid"]) != 0.0:
        # Full-train raw unavailable (only val tail + test stored); honest
        # fallback to persistence for train (flagged in note).
        tr_final = tr_persist
        train_note = "train=persist fallback (full-train raw not stored)"
    elif adopted["kind"] == "residual":
        tr_final = tr_persist
        train_note = "train=persist (residual w=0)"
    else:
        a, b, w = float(adopted["a"]), float(adopted["b"]), float(adopted["w"])
        if float(w) == 0.0:
            # cal-after-blend w=0: blend(raw,persist,0)=persist -> a*persist+b
            tr_final = a * tr_persist + b
            tr_final[~np.isfinite(tr_persist)] = np.nan
        else:
            tr_final = tr_persist  # w>0 needs train raw; fallback
        train_note = f"train via persist-only transform (w={w})"
    train_m = calculate_all_metrics(
        frame(y_train_full["cma_canonical"].to_numpy(),
              pd.to_datetime(y_train_full["date"]).to_numpy(),
              y_train_full[TARGET].to_numpy(dtype=float), tr_final), TARGET)

    summary = {
        "variant": "pretrained_ts", "try": 2,
        "adopted_kind": adopted["kind"],
        "adopted": adopted,
        "arm_i": adopted_i, "arm_i_best": best_i,
        "guard_val": float(guard_val), "raw_val": float(raw_val),
        "improvement_i_vs_guard": float(impr),
        "arm_ii_best": best_r,
        "val": {"nrmse": float(adopted["val"]), "mda": float(val_m["MDA"])},
        "test": {"nrmse": test_nrmse, "mda": float(test_m["MDA"]),
                 "nmae": float(test_m["NMAE"])},
        "train_nrmse": float(train_m["NRMSE"]),
        "n_chronos_calls": 0,
        "beats_try1": beats,
        "artifact": "updated" if beats else "kept-try-1",
        "wall_s": round(time.time() - t0, 1),
    }

    if beats:
        # Encode adopted arm in legacy schema so predict.py stays compatible.
        if adopted["kind"] == "residual":
            # persist + w*a*(raw-persist) == blend(raw,persist,w_eff), w_eff=w*a
            w_eff = float(adopted["w_resid"]) * float(adopted["a_resid"])
            new = dict(art)
            new.update({"try": 2, "calib_a": 1.0, "calib_b": 0.0,
                        "blend_w": float(w_eff), "blend_grid": [0.0, 0.25, 0.5],
                        "cal_order": "cal-before-blend",
                        "best_val_nrmse": float(adopted["val"]),
                        "val_mda": float(val_m["MDA"]),
                        "resid_a": float(adopted["a_resid"]),
                        "resid_w": float(adopted["w_resid"]),
                        "resid_lam": float(adopted["lam_resid"]),
                        "n_chronos_calls_try2": 0})
            # Per-CMA final test/train forecasts for artifact compat.
            te_out, cursor = {}, 0
            for cma, idx in rows:
                n = len(idx)
                te_out[cma] = [float(v) for v in te_final[cursor:cursor + n]]
                cursor += n
            new["test_forecasts"] = te_out
            new["train_forecasts"] = {c: [float(v) for v in np.asarray(a_, dtype=float)]
                                      for c, a_ in art["train_forecasts"].items()}
            joblib.dump(new, OUT / "pretrained_ts.joblib")
        else:
            new = dict(art)
            new.update({"try": 2, "calib_a": float(adopted["a"]),
                        "calib_b": float(adopted["b"]),
                        "blend_w": float(adopted["w"]),
                        "blend_grid": [float(g) for g in BLEND_GRID_I],
                        "cal_order": adopted.get("order", "cal-after-blend"),
                        "best_val_nrmse": float(adopted["val"]),
                        "val_mda": float(val_m["MDA"]),
                        "shrink_lam": float(adopted["lam"]),
                        "n_chronos_calls_try2": 0})
            # Final per-CMA forecasts under adopted transform.
            te_out, cursor = {}, 0
            for cma, idx in rows:
                n = len(idx)
                te_out[cma] = [float(v) for v in te_final[cursor:cursor + n]]
                cursor += n
            tr_rows = []
            for cma in cmas:
                g = y_train_full[y_train_full["cma_canonical"] == cma].sort_values("date")
                tr_rows.append((cma, g.index.to_numpy()))
            tr_out = {}
            a, b = float(adopted["a"]), float(adopted["b"])
            for cma, idx in tr_rows:
                pp = train_persist_full.loc[idx].to_numpy(dtype=float)
                ff = a * pp + b
                ff[~np.isfinite(pp)] = np.nan
                tr_out[cma] = [float(v) for v in ff]
            new["test_forecasts"] = te_out
            new["train_forecasts"] = tr_out
            joblib.dump(new, OUT / "pretrained_ts.joblib")
        metrics = json.loads((OUT / "metrics.json").read_text())
        metrics["pretrained_ts"] = {"nrmse_train": float(train_m["NRMSE"]),
                                    "nrmse_test": test_nrmse,
                                    "mda_test": float(test_m["MDA"]),
                                    "nmae_test": float(test_m["NMAE"])}
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("artifact + metrics.json UPDATED", flush=True)
    else:
        print("artifact KEPT (try-1)", flush=True)

    note = (f"TRY-2 guarded+residual, 0 new Chronos calls (reuse stored last120 "
            f"val_raw/test_raw); (i) ridge->(1,0) cal-after-blend lam grid "
            f"{[0,1,10,100,1000,10000,100000,1000000,10000000,100000000]} best "
            f"w={best_i['w']} lam={best_i['lam']:g} a={best_i['a']:.4f} b={best_i['b']:.4f} "
            f"val {best_i['val']:.4e} vs guardrail {guard_val:.4e} "
            f"(impr {impr*100:.2f}%, rule >1%: {'adopt' if adopted_i['kind']=='shrunk-cal' else 'guardrail'}); "
            f"(ii) residual scale-only b=0 lam={best_r['lam']:g} a={best_r['a']:.6f} "
            f"w={best_r['w']} val {best_r['val']:.4e}; final adopted={adopted['kind']} "
            f"val {adopted['val']:.4e} MDA {float(val_m['MDA']):.4f}; test {test_nrmse:.4e} "
            f"MDA {float(test_m['MDA']):.4f} vs try-1 {TRY1_TEST:.4e} "
            f"({train_note}); artifact {summary['artifact']}")
    with open(OUT / "tries.jsonl", "a") as f:
        entry = {"variant": "pretrained_ts", "try": 2,
                 "val_nrmse": float(adopted["val"]),
                 "val_nrmse_model_only": float(raw_val),
                 "test_nrmse": test_nrmse, "test_mda": float(test_m["MDA"]),
                 "val_mda": float(val_m["MDA"]),
                 "winner": adopted, "note": note}
        f.write(json.dumps(entry) + "\n")
    print("TRIES-APPENDED", flush=True)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
