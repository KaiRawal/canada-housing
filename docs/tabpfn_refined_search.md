# TabPFN refined search — housing-tabpfn-refined (final build 20260929)

Spec: `.flywheel/problem.yaml` (problem `housing-tabpfn-refined`).
Sandbox sources: `.flywheel/runs/20260929T065840Z/sandbox/`
(`tabpfn_{shared,percma}_{delta,ratio,level}/metrics.json` + run scripts).
Prior pin: `docs/delta_search.md` (pinned TabPFN 0.8058534 / naive floor
0.9563899, chained mask). Build entry points:
`baselines/train_tabpfn_refined.py` / `baselines/predict_tabpfn_refined.py`.

## Frozen conventions

- **Boundary-chained mask, n=1728 (16) / 2398 (24).** Deltas/lags built on the
  combined train+test timeline per CMA (`groupby.diff/shift`); the first test
  delta (y_test0 − y_train_last) is predictable from known train-tail history
  and IS scored (24/24 first-test-row valid). Per-CMA n: 108 per 16-CMA;
  display Halifax 108, Charlottetown/Guelph/Kelowna/Oshawa/QC-part/Trois-Rivières
  86, Sherbrooke 46. Within-file (1712) and per-block (1664) masks are forbidden.
- **16-only fits** (train rows == 433 per CMA); display-8 never fit per-CMA.
- **Display substitution:** display-8 scored with the FROZEN `shared_delta`
  model of THIS build (fit once, reused for all three per-CMA 24-RMSEs —
  per-CMA models cannot transfer). Recorded in every per-CMA
  `display8_substitution` block and in `artifacts/manifest.json`.
- **Test scored exactly once** in the completed build; never selected on
  (zero-shot throughout — no selection exists to leak).
- Common recipe: TabPFNRegressor v3 zero-shot, `n_estimators=2`, CPU float32,
  `random_state=0`, chunked predict (x500), raw unscaled lags + CMA via
  `categorical_features_indices` (no scaling, no one-hot).
  `ignore_pretraining_limits=True` wherever passed (shared fits n≈6720 exceed
  the v3 CPU guard of 5000 — same override as the pin; per-CMA flags are no-op
  consistency flags, except `percma_ratio` which omits the flag sandbox-exact).
- Ratio features causal (`y_t/y_{t-1}` from known history only); every variant
  scored as implied deltas on the single chained mask.

## Standings (implied-delta RMSE on the 16, chained mask)

| Variant | 16-RMSE | 24-RMSE | Margin vs gate 0.8059 | Verdict |
|---|---|---|---|---|
| `tabpfn_shared_delta` (HEAD) | **0.7990183** | 0.7581221 | −0.00684 (PASSES) | new cross-loop best |
| `tabpfn_shared_ratio` | 0.8059287 | 0.7644515 | +0.000075 (MISS) | effective tie with pin |
| `tabpfn_shared_level` | 4.4748443 | 3.8290773 | +3.669 (dead) | negative control |
| `tabpfn_percma_delta` | 0.8549722 | 0.8008987 | +0.0491 (MISS) | scope loss −0.056 |
| `tabpfn_percma_ratio` | 0.8281837 | 0.7803508 | +0.0223 (MISS) | scope loss −0.022 |
| `tabpfn_percma_level` | 10.1126857 | 8.5911538 | +9.307 (catastrophic) | huge-error negative |

Build reproduces every sandbox pin bit-exact (diff 0.00e+00 on all 12
16/24 values, tol 1e-6). `gates.performance` (`delta_rmse < 0.8059`): HEAD
PASSES; all others MISS. `gates.correctness` (`pass_rate == 1.0`): 6/6 suites
green. Researcher aspirational target ≤0.765 MISSED by +0.034 on the head —
reported as miss; it is context only, not a gate.

## MANDATORY prediction-comparison section

Final table (all on the single chained mask, implied-delta RMSE-16):

| Predictor | 16-RMSE | Source |
|---|---|---|
| naive (0 / 1.0 / persistence floors) | 0.9564 | this build (bit-matches `docs/delta_search.md`) |
| pin (centered+dummies TabPFN) | 0.8059 | `docs/delta_search.md` |
| `tabpfn_shared_delta` (HEAD) | **0.7990** | this build |
| `tabpfn_shared_ratio` | 0.8059 | this build (effective tie, +7.5e-5) |
| sarimax (pooled R4) | 0.7999 | sarimax loop (`sarimax_pooled` 0.7998947) |
| TabPFN-pin | 0.8059 | `docs/delta_search.md` (same as pin) |

Head edges sarimax_pooled by −0.00088 and beats the pin by −0.00684 (~0.85%).

### Per-CMA scope analysis (shared vs per-CMA, fitted-16 only)

Display-8 entries are frozen-shared by construction (identical on both sides)
and are EXCLUDED from this verdict — only the 16 fitted CMAs compare.

- **Delta framing:** shared wins 14/16. Biggest shared wins are the volatile
  pockets — Ottawa-ON (+0.1911), KCW (+0.1337), Windsor (+0.1143) — plus
  Winnipeg (+0.0662), Saskatoon (+0.0614), Québec (+0.0508). Per-CMA wins only
  at St. John's (−0.0128) and Vancouver-hair (−0.0067).
- **Ratio framing:** shared wins 12/16 fitted (Ottawa-ON +0.0651, Winnipeg
  +0.0583, Windsor +0.0580, London +0.0442). Per-CMA pockets: Montreal
  (−0.0267), Vancouver (−0.0114), St. John's (−0.0049), KCW-hair (−0.0008).
- **Scope verdict: SHARED wins both framings** (delta −0.056, ratio −0.022
  pooled). Per-CMA pockets of resistance are recorded above but do not
  overturn the pooled verdict — cross-CMA pooling helps most where volatility
  is highest.

### Level dose-response extrapolation analysis

Error tracks `n_test_above_trainmax` (test months above the train max, /108):
Edmonton (abv=0) 1.9098, Victoria (abv=0) 1.0646, Regina (25) 1.6870,
Saskatoon (45) 2.4969, St. John's (63) 1.7282, Québec (67) 6.2305, Sudbury
(93) 6.5930, Ottawa-ON (104) 17.1101, Vancouver (106) 7.0465, and every
abv=108 CMA ≥2.23 up to Windsor 22.6605. Two findings: (1) the dose-response
confirms trend extrapolation as the killer; (2) even non-extrapolating CMAs
(Edmonton/Victoria, abv=0) still lose to persistence — TabPFN has no trend
concept at all, so the level framing is dead with or without extrapolation.
Per-scale fits do NOT rescue the shared-level cross-scale failure
(`percma_level` 10.1127 ≫ `shared_level` 4.4748 ≫ naive 0.9564).

### Preprocessing verdict

Raw unscaled lags + categorical CMA id (`shared_delta` 0.7990) >=
mean-centered deltas + 16 one-hot dummies (pin 0.8059), margin −0.0068.
CAVEAT (recorded, not hidden): the head recipe differs from the pin recipe,
so this is a recipe comparison with a confound, not a pure ablation — and the
head margin is sub-1% (~0.85%), which lineage hints flag as noise-adjacent.
The binary gate (`<0.8059`) still PASSES; no multi-window agreement was
demanded by the spec, but the caveat stands for readers.

### License / fetch notes

Weights: `tabpfn-v3-regressor-v3_default.ckpt` from the public
`Prior-Labs/tabpfn_3` repo (revision `24a16a8…`) via unauthenticated
`hf_hub_download` into the HF cache (pointer only in `manifest.json`, never
committed). TabPFN checkpoints are NON-COMMERCIAL research-only — this
academic-research repo complies (noted in trainer docstring, metrics configs,
manifest, and here).

### Target verdict

delta ~= ratio > level. Ratio matches delta within +0.0069 pooled
(effective tie at the pin: +7.5e-5 vs 0.8058534 — report "delta ~= ratio",
never a ratio beat). Level is dead in both scopes. The aspirational ≤0.765
target is missed by +0.034 on the head — unreachable in this recipe family;
reported plainly, caps the claim at "beats pin, edges sarimax, target
unreached."

## Fix-forward log (≤3 nudges; 1 used)

- **Nudge 1 (build crash, fixed):** first train attempt wrote the 3 shared
  joblibs then crashed in `save_artifact` — per-CMA `per_cma_detail` dicts
  lacked frozen display-8 entries (`AssertionError: display CMA missing`).
  Fix: `train_percma_delta`/`train_percma_ratio` now merge the frozen
  `shared_delta` display entries (with `frozen_from` markers) exactly as
  `train_percma_level` already did; trainer fix amended into the C1 commit
  (local, unpushed); partial joblibs deleted; full serial rerun completed in
  132.7s as the single finished build. No selection exists (zero-shot), so the
  crashed attempt — which never finalized metrics — cannot leak test signal;
  the rerun is the one test scoring.
- Remaining budget: 1 nudge, unused.
- **Nudge 2 (predict key-count, fixed):** with nudge 1 fixed, training
  completed but `predict_tabpfn_refined.py` rejected the per-CMA artifacts
  (`pred count 1728 != 2398`) — their `pred_delta` dicts held fitted-16 keys
  only. Fix: each per-CMA trainer now copies the 670 frozen `shared_delta`
  display-8 keys into its `pred_delta` (literal frozen substitution) and
  canonically rescores the stored 16/24 RMSEs from the merged keyed dict in
  (cma,date)-sorted order — the single source of truth, bit-agreeing with
  predict/tests joins. Pins still reproduce (tol 1e-6); trainer fix amended
  into the C1 commit (local, unpushed); full serial rerun (133.1s) is the
  single finished build; C2 re-committed with the rebuilt artifacts.

## Provenance

- Inputs ported read-only from `flywheel/housing-deltas-20260929T004134Z`
  (`prediction/{X,y}_{train,test}_full_total.csv`; 7793/2399 lines; sandbox
  CSVs bit-identical, md5-verified; hashes in `manifest.json`).
- Problem spec untouched (`.flywheel/problem.yaml` unmodified).
- Commits: C1 inputs+modules (`0322040`), C2 artifacts (`0b035af`),
  C3 tests+docs (this file + `tests/test_tabpfn_*.py`).
