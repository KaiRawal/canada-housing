"""Pretrained_ts TRY-1: strict out-of-sample val backtest + global affine
val-calibration + persistence blend (guardrail w=0), context re-check.

Chronos budget: 24 CMAs x 3 max = 72 forward calls
  (val: 2 contexts x 24; test: winner context x 24).
Val arithmetic (calibration + blend + order selection): 0 extra calls.
Train entry: reuse run-1 stored raw last120 backtest when winner==last120
(0 calls); else run winner-context backtest.

Writes (only on beat vs run-1 test NRMSE, else keeps + says so):
  artifacts/pretrained_ts.joblib, artifacts/metrics.json
Always appends: artifacts/tries.jsonl
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

from baselines._nesting import blend, persistence_pred, tune_blend  # noqa: E402
from baselines._pretrained_ts import (  # noqa: E402
    backtest_train_per_cma,
    forecast_median,
    series_by_cma,
    split_fit_val,
    truncate_context,
)
from evaluation.evaluation import calculate_all_metrics  # noqa: E402

TARGET = "total"
OUT = ROOT / "artifacts"
BLEND_GRID = (0.0, 1.0)
CONTEXTS = ["last120", "full"]
RUN1_TEST = 0.00903866301222721

CALLS = {"n": 0}


def fc(pipe, ctx, horizon):
    CALLS["n"] += 1
    return forecast_median(pipe, ctx, horizon)


def pooled_nrmse_valid(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    n = int(valid.sum())
    if n == 0:
        return float("nan")
    mse = float(np.mean((y_true[valid] - y_pred[valid]) ** 2))
    return float(np.sqrt(mse) / n)


def ls_cal(p, y):
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = ~(np.isnan(p) | np.isnan(y))
    A = np.column_stack([p[valid], np.ones(int(valid.sum()))])
    (a, b), _, _, _ = np.linalg.lstsq(A, y[valid], rcond=None)
    return float(a), float(b)


def main():
    t0 = time.time()
    y_train_full = pd.read_csv(ROOT / "prediction" / f"y_train_full_{TARGET}.csv",
                               parse_dates=["date"])
    y_test_full = pd.read_csv(ROOT / "prediction" / f"y_test_full_{TARGET}.csv",
                              parse_dates=["date"])
    train_mask = y_train_full[TARGET].notna().to_numpy()
    ytr = y_train_full.loc[train_mask].reset_index(drop=True)
    test_mask = y_test_full[TARGET].notna().to_numpy()
    yte = y_test_full.loc[test_mask].reset_index(drop=True)
    train_series = series_by_cma(ytr, TARGET)
    test_series = series_by_cma(yte, TARGET)
    cmas = sorted(train_series.keys())
    print(f"cmas={len(cmas)} val_horizons=" +
          ",".join(str(len(split_fit_val(train_series[c])[1])) for c in cmas[:5]) + "...",
          flush=True)

    from baselines._pretrained_ts import load_pipeline
    device = None
    pipe = None
    for dev in ("mps", "cpu"):
        try:
            pipe = load_pipeline(dev)
            device = dev
            break
        except Exception as exc:  # noqa: BLE001
            print(f"load failed on {dev}: {exc}", flush=True)
    assert pipe is not None, "no device worked"
    print(f"pipeline on {device}", flush=True)

    # ---- VAL + TEST raw forecasts (72-call cap) ----
    val_true, val_raw, val_fit_last = {}, {}, {}
    for choice in CONTEXTS:
        vt, vr = [], []
        for cma in cmas:
            vals = train_series[cma]
            fit, val = split_fit_val(vals)
            ctx = truncate_context(fit, choice)
            if len(ctx) >= 3 and len(val) > 0:
                fcst = fc(pipe, ctx, len(val))
            else:
                fcst = np.full(len(val), fit[-1] if len(fit) else np.nan)
            vt.append(np.asarray(val, dtype=float))
            vr.append(np.asarray(fcst, dtype=float))
        val_true[choice] = np.concatenate(vt)
        val_raw[choice] = np.concatenate(vr)
    print(f"val raw done, calls={CALLS['n']} "
          f"raw_nrmse={ {c: pooled_nrmse_valid(val_true[c], val_raw[c]) for c in CONTEXTS} }",
          flush=True)

    # Val row index map (date-ordered tails) for one-step persistence + MDA frame.
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
    persist_val_nrmse = pooled_nrmse_valid(y_val, val_persist)
    print(f"persist val_nrmse={persist_val_nrmse:.10f}", flush=True)

    # ---- Val arithmetic per context: cal + blend order ----
    ctx_results = {}
    for choice in CONTEXTS:
        p = np.asarray(val_raw[choice], dtype=float)
        raw_n = pooled_nrmse_valid(y_val, p)
        a_pre, b_pre = ls_cal(p, y_val)
        cal = a_pre * p + b_pre
        cal_n = pooled_nrmse_valid(y_val, cal)
        w_pre, best_pre = tune_blend(y_val, cal, val_persist, grid=BLEND_GRID)
        # cal-after-blend: per-w LS on blended raw.
        post_curve = []
        for g in BLEND_GRID:
            br = blend(pd.Series(p), pd.Series(val_persist),
                       float(g)).to_numpy(dtype=float)
            v = ~(np.isnan(y_val) | np.isnan(br))
            if int(v.sum()) == 0:
                post_curve.append({"w": float(g), "a": float("nan"),
                                   "b": float("nan"), "val_nrmse": float("inf")})
                continue
            Ab = np.column_stack([br[v], np.ones(int(v.sum()))])
            (aw, bw), _, _, _ = np.linalg.lstsq(Ab, y_val[v], rcond=None)
            fin = float(aw) * br + float(bw)
            post_curve.append({"w": float(g), "a": float(aw), "b": float(bw),
                               "val_nrmse": pooled_nrmse_valid(y_val, fin)})
        post_best = min(post_curve, key=lambda d: d["val_nrmse"])
        if float(post_best["val_nrmse"]) < float(best_pre):
            order, a, b, w, best = ("cal-after-blend", float(post_best["a"]),
                                    float(post_best["b"]), float(post_best["w"]),
                                    float(post_best["val_nrmse"]))
        else:
            order, a, b, w, best = ("cal-before-blend", a_pre, b_pre,
                                    float(w_pre), float(best_pre))
        ctx_results[choice] = {"raw": raw_n, "cal_n": cal_n, "a_pre": a_pre,
                               "b_pre": b_pre, "w_pre": float(w_pre),
                               "best_pre": float(best_pre),
                               "post_curve": post_curve, "order": order,
                               "a": a, "b": b, "w": w, "best": best}
        print(f"ctx={choice} raw={raw_n:.10f} cal=({a_pre:.6f},{b_pre:.6f})->{cal_n:.10f} "
              f"pre[w={w_pre}->{best_pre:.10f}] post={[(d['w'], round(d['val_nrmse'], 10)) for d in post_curve]} "
              f"=> {order} w={w} val={best:.10f}", flush=True)

    winner_ctx = min(ctx_results, key=lambda c: ctx_results[c]["best"])
    W = ctx_results[winner_ctx]
    print(f"WINNER ctx={winner_ctx} order={W['order']} a={W['a']} b={W['b']} w={W['w']} "
          f"val={W['best']:.10f}", flush=True)

    # ---- TEST raw for winner only (24 calls) ----
    test_raw = {}
    for cma in cmas:
        vals = train_series[cma]
        horizon = len(test_series.get(cma, np.array([], dtype=float)))
        ctx = truncate_context(vals, winner_ctx)
        if len(ctx) >= 3 and horizon > 0:
            test_raw[cma] = fc(pipe, ctx, horizon)
        else:
            last = float(vals[-1]) if len(vals) else float("nan")
            test_raw[cma] = np.full(horizon, last)
    print(f"test raw done, calls={CALLS['n']}", flush=True)

    # ---- Apply winner transform ----
    def apply_final(raw_arr, persist_arr, W):
        raw = np.asarray(raw_arr, dtype=float)
        ps = np.asarray(persist_arr, dtype=float)
        if W["order"] == "cal-after-blend":
            br = blend(pd.Series(raw), pd.Series(ps),
                       float(W["w"])).to_numpy(dtype=float)
            return float(W["a"]) * br + float(W["b"])
        cal = float(W["a"]) * raw + float(W["b"])
        return blend(pd.Series(cal), pd.Series(ps),
                     float(W["w"])).to_numpy(dtype=float)

    # Val MDA on winner.
    val_final = apply_final(val_raw[winner_ctx], val_persist, W)
    val_frame = pd.DataFrame({"cma_canonical": val_cmas, "date": val_dates,
                              f"{TARGET}_true": y_val, f"{TARGET}_pred": val_final})
    val_m = calculate_all_metrics(val_frame, TARGET)

    # Test frames: raw-honest + final + persistence.
    test_persist_full = persistence_pred(y_test_full)
    rows = []
    for cma in cmas:
        g = y_test_full[y_test_full["cma_canonical"] == cma].sort_values("date")
        idx = g.index.to_numpy()
        rows.append((cma, idx))
    te_idx = np.concatenate([r[1] for r in rows])
    y_te = y_test_full.loc[te_idx, TARGET].to_numpy(dtype=float)
    te_persist = test_persist_full.loc[te_idx].to_numpy(dtype=float)
    te_raw_concat = np.concatenate([np.asarray(test_raw[c], dtype=float)
                                    for c, _ in rows])
    te_final = apply_final(te_raw_concat, te_persist, W)

    def frame(pred):
        return pd.DataFrame({"cma_canonical": y_test_full.loc[te_idx, "cma_canonical"].to_numpy(),
                             "date": pd.to_datetime(y_test_full.loc[te_idx, "date"]).to_numpy(),
                             f"{TARGET}_true": y_te, f"{TARGET}_pred": pred})
    m_raw = calculate_all_metrics(frame(te_raw_concat), TARGET)
    m_final = calculate_all_metrics(frame(te_final), TARGET)
    m_persist = calculate_all_metrics(frame(te_persist), TARGET)
    print(f"test raw_nrmse={m_raw['NRMSE']:.10f} mda={m_raw['MDA']:.4f}", flush=True)
    print(f"test final_nrmse={m_final['NRMSE']:.10f} mda={m_final['MDA']:.4f} "
          f"nmae={m_final['NMAE']:.10f}", flush=True)
    print(f"test persist_nrmse={m_persist['NRMSE']:.10f} mda={m_persist['MDA']:.4f}",
          flush=True)

    # ---- Train entry: reuse stored raw backtest if winner==last120 ----
    old = joblib.load(OUT / "pretrained_ts.joblib")
    if winner_ctx == str(old.get("best_params", {}).get("context", "")):
        train_fc = {c: np.asarray(a, dtype=float)
                    for c, a in old["train_forecasts"].items()}
        print("train: reused stored raw backtest (0 calls)", flush=True)
    else:
        train_fc = backtest_train_per_cma(pipe, train_series, winner_ctx)
        for cma in train_series:
            if cma not in train_fc:
                train_fc[cma] = np.full(len(train_series[cma]), np.nan)
        print(f"train: fresh backtest, calls={CALLS['n']}", flush=True)
    train_persist_full = persistence_pred(y_train_full)
    tr_rows = []
    for cma in cmas:
        g = y_train_full[y_train_full["cma_canonical"] == cma].sort_values("date")
        tr_rows.append((cma, g.index.to_numpy()))
    tr_idx = np.concatenate([r[1] for r in tr_rows])
    tr_raw_concat = np.concatenate([np.asarray(train_fc[c], dtype=float)[:len(i)]
                                    for c, i in tr_rows])
    tr_persist = train_persist_full.loc[tr_idx].to_numpy(dtype=float)
    tr_final = apply_final(tr_raw_concat, tr_persist, W)
    train_frame = pd.DataFrame(
        {"cma_canonical": y_train_full.loc[tr_idx, "cma_canonical"].to_numpy(),
         "date": pd.to_datetime(y_train_full.loc[tr_idx, "date"]).to_numpy(),
         f"{TARGET}_true": y_train_full.loc[tr_idx, TARGET].to_numpy(dtype=float),
         f"{TARGET}_pred": tr_final})
    m_train = calculate_all_metrics(train_frame, TARGET)
    print(f"train final_nrmse={m_train['NRMSE']:.10f}", flush=True)

    test_nrmse = float(m_final["NRMSE"])
    beats = bool(np.isfinite(test_nrmse) and test_nrmse < float(RUN1_TEST))
    print(f"beats run-1 ({RUN1_TEST:.10f})? {beats}", flush=True)

    # Per-CMA raw store for TRY-2 (0 new calls later).
    val_raw_per_cma, cursor = {}, 0
    for cma in cmas:
        n = len(split_fit_val(train_series[cma])[1])
        val_raw_per_cma[cma] = [float(v) for v in
                                np.asarray(val_raw[winner_ctx][cursor:cursor + n])]
        cursor += n

    summary = {
        "variant": "pretrained_ts", "try": 1,
        "winner": {"context": winner_ctx, "device": device, "a": W["a"],
                   "b": W["b"], "w": W["w"], "cal_order": W["order"]},
        "val": {"nrmse": float(W["best"]),
                "model_only": {c: float(ctx_results[c]["raw"]) for c in CONTEXTS},
                "persist": float(persist_val_nrmse),
                "cal": {c: {"a": float(ctx_results[c]["a_pre"]),
                            "b": float(ctx_results[c]["b_pre"]),
                            "nrmse": float(ctx_results[c]["cal_n"])} for c in CONTEXTS},
                "mda": float(val_m["MDA"])},
        "test": {"nrmse": test_nrmse, "mda": float(m_final["MDA"]),
                 "nmae": float(m_final["NMAE"]),
                 "raw_nrmse": float(m_raw["NRMSE"]), "raw_mda": float(m_raw["MDA"]),
                 "persist_nrmse": float(m_persist["NRMSE"])},
        "train_nrmse": float(m_train["NRMSE"]),
        "n_chronos_calls": int(CALLS["n"]),
        "beats_run1": beats,
        "artifact": "updated" if beats else "kept-run-1",
        "wall_s": round(time.time() - t0, 1),
    }

    if beats:
        test_fc_out, cursor = {}, 0
        for cma, idx in rows:
            n = len(idx)
            test_fc_out[cma] = [float(v) for v in te_final[cursor:cursor + n]]
            cursor += n
        train_fc_out, cursor = {}, 0
        for cma, idx in tr_rows:
            n = len(idx)
            train_fc_out[cma] = [float(v) for v in tr_final[cursor:cursor + n]]
            cursor += n
        artifact = dict(old)
        artifact.update({
            "try": 1,
            "best_params": {"context": winner_ctx, "device": device},
            "best_val_nrmse": float(W["best"]),
            "val_nrmse": [{"params": {"context": c}, "device": device,
                           "val_nrmse": float(ctx_results[c]["best"])}
                          for c in CONTEXTS],
            "val_nrmse_model_only": {c: float(ctx_results[c]["raw"]) for c in CONTEXTS},
            "persist_val_nrmse": float(persist_val_nrmse),
            "calib_a": float(W["a"]), "calib_b": float(W["b"]),
            "blend_w": float(W["w"]), "blend_grid": [float(g) for g in BLEND_GRID],
            "cal_order": W["order"],
            "val_mda": float(val_m["MDA"]),
            "test_forecasts": test_fc_out, "train_forecasts": train_fc_out,
            "val_raw_per_cma": {c: [float(v) for v in np.asarray(val_raw[c])]
                                for c in CONTEXTS},
            "test_raw_per_cma": {c: [float(v) for v in np.asarray(test_raw[c])]
                                 for c in cmas},
            "n_chronos_calls_try1": int(CALLS["n"]),
        })
        joblib.dump(artifact, OUT / "pretrained_ts.joblib")
        metrics = json.loads((OUT / "metrics.json").read_text())
        metrics["pretrained_ts"] = {
            "nrmse_train": float(m_train["NRMSE"]),
            "nrmse_test": test_nrmse,
            "mda_test": float(m_final["MDA"]),
            "nmae_test": float(m_final["NMAE"]),
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("artifact + metrics.json UPDATED", flush=True)
    else:
        print("artifact KEPT (run-1)", flush=True)

    tries = OUT / "tries.jsonl"
    note = (f"TRY-1 cal+blend nesting: strict last-20%-per-CMA val backtest "
            f"({CALLS['n']} Chronos calls); ctx val raw " +
            ",".join(f"{c} {ctx_results[c]['raw']:.4f}" for c in CONTEXTS) +
            f"; global LS cal({W['a']:.4f},{W['b']:.4f}) {W['order']} w={W['w']} "
            f"val {W['best']:.4e} MDA {val_m['MDA']:.4f} (persist val "
            f"{persist_val_nrmse:.4e}); test {test_nrmse:.4e} MDA {m_final['MDA']:.4f} "
            f"vs persist {m_persist['NRMSE']:.4e}/{m_persist['MDA']:.4f}, run-1 {RUN1_TEST:.4e}; "
            f"artifact {summary['artifact']}")
    with open(tries, "a") as f:
        f.write(json.dumps({"variant": "pretrained_ts", "try": 1,
                            "val_nrmse": float(W["best"]),
                            "val_nrmse_model_only": float(ctx_results[winner_ctx]["raw"]),
                            "test_nrmse": test_nrmse, "test_mda": float(m_final["MDA"]),
                            "val_mda": float(val_m["MDA"]),
                            "winner": summary["winner"], "note": note}) + "\n")
    print("TRIES-APPENDED", flush=True)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
