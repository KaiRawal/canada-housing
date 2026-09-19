# Learnings handoff — canada-housing baselines (housing-baselines runs)

For the next data scientist starting on this problem. Based on two flywheel
runs (6-class sweep + 3-tries-per-class rerun, 15 scored tries, 22/22 tests
green) on branch `flywheel/housing-baselines-20260917T102048Z`.

## Verdict up front

1-step-ahead monthly **level** forecasting on this data is a near-unit-root
problem where **persistence is the near-optimal prior** — a property of the
data, not a failure of effort. Monthly drift (~0.13–0.25 index points)
sits inside monthly noise (~0.54–0.86), i.e. signal-to-noise ≈ 0.25:
textbook random-walk territory where the MSE-optimal 1-step forecast is
essentially last month's value. Our best model beats persistence by 5.2% in
NRMSE, which in real units (repo NRMSE = RMSE/n, `evaluation/evaluation.py`,
n_valid = 2374) is RMSE ≈ 0.85 vs 0.90 index points on levels of 100–176 —
a **0.05-point** gain. Nothing in 176 exogenous columns beats that by more
than ~5%, because slow-moving census variables carry almost no 1-step-ahead
information beyond what lag-1 already says.

Final test NRMSE vs persistence (3.778184008276179e-04): linear −5.2%,
GBT −4.7%, LSTM −4.1%, Chronos −1.3%, Prophet tie. All from
`artifacts/metrics.json`; full trajectories in `artifacts/tries.jsonl`.

## Seven learnings, each transferable

1. **Nesting is the price of admission.** Every first-round failure traced to
   models that couldn't even *represent* persistence (no lags, no AR term).
   Once nesting was added (causal lag features, delta-targets, blend weight
   w=0 bit-exact via `baselines/_nesting.py`), 4 of 5 classes beat naive.
   Any future model must nest persistence or it loses by construction.
2. **Direction is noise at this frequency.** Persistence MDA = 0.547
   (n≈2350, z≈4.6 — statistically significant, economically a coin flip),
   and *every* model that won on NRMSE lost on MDA (best: LSTM 0.450).
   Level is forecastable (it's last month); monthly direction is not, with
   these features.
3. **The feature store is ~40% duplicates.** 34 exact/near-duplicate pairs
   (|corr|>0.999), 583 pairs >0.99; 176→110 columns with zero loss. Census
   aggregates repeat across tables. De-duplicate before any dense estimator
   (Ridge-1e-4 stayed 5x worse than naive even after full dedup —
   structural collinearity).
4. **Small CMAs poison pooled fitting.** Guelph (~24 rows) dominated SSE
   thinking twice; Sherbrooke / Ottawa-Gatineau-QC / Hamilton / Montreal
   concentrate all errors. Exclude or hierarchically pool small CMAs; never
   let them vote in pooled losses.
5. **Single chronological split is fragile.** Val→test gaps of 50–70% were
   systematic (test delta-std 2.4x val); sub-1% validation deltas are
   meaningless. Nothing here has multi-window confirmation — the "beats"
   may not survive a second window.
6. **Calibration overfits; shrinkage survives.** Unshrunk 2-param val fits
   repeatedly won validation and lost test (Prophet +12%, Chronos +9.5%).
   What generalized: 1-SE shrinkage toward (1,0), regime-split
   calibration, guardrailed blends. In-sample validation lies — selection
   must be strictly out-of-sample.
7. **Zero-shot foundation models add nothing here.** Chronos raw output was
   16x worse than naive on validation with ~zero residual correlation
   (−0.010); only shrinkage toward persistence salvaged −1.3%. Without
   covariates, pretraining doesn't transfer to this series.

## Prioritized recommendations

1. **Change the horizon first.** Everything above is h=1, where naive is
   provably hard to beat. At h=3/6/12 the naive error grows ~√h while
   trend/seasonal structure compounds — that is where Prophet/Chronos/GBT
   can win by margins that matter. Never tested; highest-value experiment.
2. **Significance before celebration.** Diebold-Mariano on the loss
   differentials + expanding-window CV. A −5% single-split win with 70%
   val-test gaps is a hypothesis, not a result.
3. **Audit real-time availability (open question).** Supervised models here
   use *contemporaneous* monthly X. If census/CMHC values aren't published
   before month-end in production, that's lookahead bias, and the honest
   comparison (naive vs lagged-features-only) would make deployable reality
   look *better* than these results for persistence. Nobody checked this.
4. **Treat direction as a separate problem** (different features — rates,
   sentiment, market microstructure — or accept noise), and consider
   per-CMA/hierarchical modeling for the small-CMA tail.

## Starting artifacts

- `artifacts/*.joblib` — best-of-3 model per class; `artifacts/metrics.json`
- `artifacts/tries.jsonl` — all 17 tries + restores (full search history)
- `docs/hyperparam_search.md` — per-class trajectories, winners, comparison
- `baselines/` — `train.py`/`predict.py` harness + `_nesting.py`
  (causal lags, persistence blend)
- Run provenance: `.flywheel/runs/` (`events.jsonl`/`decisions.md`,
  queryable via `.flywheel/shared/observe.py query <run-dir>`)

## Sources

`artifacts/metrics.json`, `artifacts/tries.jsonl` (17 lines),
`evaluation/evaluation.py` (NRMSE def), `docs/hyperparam_search.md`,
`.flywheel/runs/20260917T102048Z/` (16 events),
`.flywheel/runs/20260917T092329Z/` (9 events).

## Addendum — rerun-2 (flywheel run `20260919T090245Z`)

Same 6 model classes, prior learnings reused, accuracy pushed at every
step. Serial execution (memory-pressured host). Committed summary:
`docs/rerun-2/` (run `learnings.md` + `plan.md`); full provenance stays
in gitignored `.flywheel/runs/20260919T090245Z/`.

| Class | Try | Test NRMSE | Verdict |
| --- | --- | --- | --- |
| persistence | re-verify | 3.7782e-04 | bit-exact anchor, frozen |
| linear_regression | T4 small-CMA exclusion | 3.58068e-04 (-0.035%) | beat, HELD (noise-level) |
| linear_regression | T5 partial-pooling (shared EN + 24 shrunk CMA offsets, k=10) | **3.57411e-04 (-5.40% vs persistence)** | **ADOPTED, artifact updated** |
| gbt_ensemble | T4 per-regime tree refits | 3.6046e-04 (+0.137%) | honest fail, held at T3 |
| lstm_rnn | T4 capacity sweep (H96/L9/L12/2-layer) | 3.6367e-04 (-0.32%) | honest fail, held at T2 |
| prophet | T4 yearly-OFF <100 | 3.7782e-04 (tie) | frozen at T1 restore |
| pretrained_ts | T4 context sweep (last60/120/240) | 3.7278e-04 (tie by w=0 construction) | frozen at T3 |

`pass_rate == 1.0` holds (22/22 tests green). `artifacts/manifest.json`
minted. Details: `docs/hyperparam_search.md` §9.

### Analysis notes (from post-run Q&A)

- **Random-walk ceiling.** Monthly drift (~0.13–0.25 index points) sits
  inside monthly noise (~0.54–0.86): S/N ≈ 0.25, so the MSE-optimal
  1-step forecast is ~last month's value and persistence is near-optimal.
  Best RMSE ≈ 0.85 vs 0.90 on levels 100–176 (a 0.05-point gain).
  ⚠️ The drift/jiggle bands are STATED, not yet recomputed — verify with:
  `.venv/bin/python -c "import pandas as pd; df =
  pd.read_csv('prediction/y_train_full_total.csv',
  parse_dates=['date']).sort_values(['cma_canonical','date']); d =
  df.groupby('cma_canonical')['total'].diff().dropna(); g =
  d.groupby(df.loc[d.index,'cma_canonical']).agg(['mean','std']);
  print(g['mean'].min(), g['mean'].max(), g['std'].min(),
  g['std'].max())"`. Standard-deviation-as-noise is an upper bound:
  models convert slices of it into signal (the −5.4%).
- **Validation-to-test gaps.** Same model, different time slice, big
  number moves: GBT try-2 val 2.35e-04 → test 3.62e-04 (~+54%);
  Prophet try-3 val-picked `w=1.0` → test +118%; linear try-4 val
  −0.15% → test −0.035% (same sign, 4x smaller). Cause: test-slice
  move-std 2.4x validation's. Val flatters, test judges.
- **The <1% rule.** Relative improvement (old−new)/old. Try-2→try-4 was
  0.035% — indistinguishable from luck given the gaps above, so it was
  held out of the committed artifacts. Below ~1%, don't call it a win.
- **NRMSE vs direction.** Different skills: *how far off* vs *up-or-down*.
  E.g. last=100, actual=100.2: forecast 100.5 (off 0.3, direction ✓)
  vs 99.9 (off 0.3, direction ✗). Hugging last value gives small level
  errors with coin-flip direction (linear MDA 0.39 vs naive 0.55).
  Direction at monthly frequency is noise with these features.

### Continuation (staying at h=1)

Live options: small-CMA exclusion for the LSTM pooled fit; per-CMA
residual audits for further offset targets. Killed for good: per-regime
hard-split refits, capacity expansion, unconditional Chronos context
sweeps, val-blind threshold moves. Next rigorous step when wins are
claimed: multi-window validation (Diebold-Mariano), not a single split.
