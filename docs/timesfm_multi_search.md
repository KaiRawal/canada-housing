# TimesFM multivariate search — housing-timesfm-multivariate (20260929)

Spec: `.flywheel/problem.yaml` (problem `housing-timesfm-multivariate`).
Plan: `.flywheel/runs/20260929T041734Z/plan.md`.
Code: `baselines/train_timesfm_multi.py` (5 pinned configs) +
`baselines/predict_timesfm_multi.py` (blackbox scorer).
Artifacts: `artifacts/timesfm{25_uni,3_multi,3_cov,25_xreg,3_lora}.joblib` +
`artifacts/metrics.json` + `artifacts/manifest.json`.
Tests: `tests/test_timesfm{25_uni,3_multi,3_cov,25_xreg,3_lora}.py`.

Frozen conventions (no drift): boundary-chained mask n16=1728 / n24=2398
(first test delta y_test0 − y_train_last IS scored); joint 16-CMA targets +
frozen display-8 (2.5-uni fallback values, never fit on); per-step causal
demean, external scale untouched; forecasts anchored at last train target,
actuals only, never recursive; each variant's test scored exactly once after
train-only selection.

## 1. Prediction comparison (mandatory)

Final head-vs-reference table, subset-16 chained mask (n=1728). The naive
floor is recomputed on the same mask in this build (0.956390), so all
margins below are on common ground; the pooled-classical (0.7999) and
prior-loop (0.8059) references are quoted from earlier loops on the same
frozen mask convention.

| Model | RMSE16 | Margin vs head (0.8217) | Margin vs naive (0.9564) |
|---|---|---|---|
| naive floor (chained, recomputed) | 0.9564 | +0.1347 head wins (−14.1%) | — |
| timesfm25_uni (control pin) | 0.8757 | +0.0540 head wins (−6.2%) | −0.0807 (−8.4%) |
| **timesfm3_multi (HEAD, winner)** | **0.8217** | — | **−0.1347 (−14.1%)** |
| timesfm3_cov (tie-null) | 0.8217 | 0.0000 (tie) | −0.1347 |
| timesfm3_lora (tie-null) | 0.8217 | −0.0000 (−3.1e-05, noise) | −0.1347 |
| timesfm25_xreg (documented fail) | 0.9002 | −0.0785 head wins | −0.0562 (−5.9%) |
| sarimax_pooled (prior loop ref) | 0.7999 | +0.0218 head loses | −0.1565 (−16.4%) |
| TabPFN (prior loop ref) | 0.8059 | +0.0159 head loses | −0.1505 (−15.7%) |

Gate verdict: `delta_rmse < 0.8757` → head 0.8217 PASSES by −0.0540.
Correctness: `pass_rate == 1.0` over the 5 new suites (see §7).

Per-CMA win/loss analysis (subset-16; head = joint 3.0 vs control 2.5-uni
and vs naive). Head beats naive on 15/16 and beats the control on 14/16:

| CMA | head | uni | naive | head−uni | head−naive |
|---|---|---|---|---|---|
| Winnipeg, Manitoba | 0.9208 | 1.1554 | 1.0990 | **−0.2346** | −0.1782 |
| Montréal, Quebec | 1.1221 | 1.2068 | 1.2698 | **−0.0847** | −0.1477 |
| Windsor, Ontario | 1.0657 | 1.1373 | 1.2435 | **−0.0715** | −0.1778 |
| St. Catharines-Niagara, Ontario | 0.6877 | 0.7584 | 0.7517 | **−0.0707** | −0.0640 |
| Greater Sudbury, Ontario | 0.9185 | 0.9838 | 0.9756 | −0.0653 | −0.0571 |
| Edmonton, Alberta | 0.5385 | 0.5883 | 0.5771 | −0.0499 | −0.0386 (near-naive) |
| London, Ontario | 0.9705 | 1.0197 | 1.0697 | −0.0492 | −0.0992 |
| Saskatoon, Saskatchewan | 0.4951 | 0.5425 | 0.5741 | −0.0474 | −0.0791 |
| Ottawa-Gatineau ON-part | 1.0239 | 1.0630 | 1.4272 | −0.0391 | −0.4033 |
| Kitchener-Cambridge-Waterloo | 1.0600 | 1.0859 | 1.3912 | −0.0259 | −0.3312 |
| Québec, Quebec | 0.7936 | 0.8155 | 0.8896 | −0.0219 | −0.0960 |
| Victoria, British Columbia | 0.7523 | 0.7675 | 0.7977 | −0.0152 | −0.0455 |
| Regina, Saskatchewan | 0.5213 | 0.5290 | 0.5500 | −0.0077 | −0.0287 |
| Hamilton, Ontario | 0.6207 | 0.6274 | 0.6493 | −0.0067 | −0.0286 |
| St. John's, Newfoundland and Labrador | 0.4508 | 0.4468 | 0.4376 | +0.0040 (loss) | +0.0132 (loss) |
| Vancouver, British Columbia | 0.7347 | 0.7253 | 0.7931 | +0.0093 (loss) | −0.0584 |

Reading: the joint-3.0 wins come from volatile Ontario/Quebec pockets —
Winnipeg (−0.235, the standout), Montréal, Windsor, St. Catharines — where
cross-variate information damps large swings. The losses/flats are the quiet
CMAs: St. John's loses to BOTH naive (+0.013) and control (+0.004) — the
only CMA where the head cannot beat "predict zero change"; Vancouver loses
to the control (+0.009) while still beating naive; Edmonton is near-naive
in absolute terms (0.54 vs 0.58). This matches the spec hint (near-naive on
quiet CMAs; wins from volatile pockets) and calibrates expectations: joint
modeling helps where there is signal to share, and is decoration where
there is none.

## 2. Per-variant diagnosis

- **timesfm25_uni (CONTROL, 0.8757290/0.8264737).** 2.5-200M per-CMA
  univariate rolling horizon-1, ctx≤128, per-step causal demean. Reproduced
  bit-exact (drift 0.0). Required anchor; also supplies the frozen
  display-8 fallback values reused by every other variant.
- **timesfm3_multi (HEAD, 0.8217316/0.7854996).** 3.0 zero-shot joint
  (16,≤512)→1 rolling over 108 test steps, per-variate causal demean.
  Context study on train-only last-24 steps: 128→0.313382,
  **512→0.303568 winner**, 2048→0.303568 (==512: history<512, identical
  contexts — no capacity expansion). Inner naive 0.327753. Only honest
  beat of the pin (−0.0540). Winner by gate margin.
- **timesfm3_cov (tie-null, 0.8217316).** Identical loop, test_mode=no-cov.
  Ablation (3 disjoint 24-step train-only windows, means): no-cov
  0.294038; past-only 0.296467 (0/3 wins, hurts — REJECTED); past-future
  0.293362 (2/3 wins but +0.23% < 1% bar — REJECTED); both 0.293435
  (+0.21% < 1% — REJECTED). Promoted=[] → test once no-cov = tie-null.
  Covariates add nothing; decoration kept out by the 1%-improvement rule.
- **timesfm25_xreg (FALLBACK, documented FAIL 0.9001851/0.8451958).**
  2.5+XReg, test_mode=cal (see §3 era-overfit episode). Kept as Apache-2.0
  license insurance only. Fails `gates.performance` by +0.0245 — reported,
  never re-selected.
- **timesfm3_lora (tie-null, 0.8217001/0.7854759).** Base 3.0 (330.7M) +
  PEFT LoRA 0.21.0 on query_proj+value_proj, grid rank {4,8}×epochs
  {1,2,3} (AdamW lr=1e-4, clip 1.0, batch 8, seed 42, CPU; 242 joint
  (16,128)→32 train windows ends [128..369]; inner-val 32 rolling-origin
  ends [370..401], adapters-active horizon-1 chained-delta RMSE):
  zero-shot 0.293042 vs r4e1 0.293193 vs r8e1 0.293577 — val degrades with
  epochs, ep1 selected (r4, 819200 trainable). Merged r4/ep1, identical
  loop to multi, TEST SCORED ONCE → 0.8217001 (vs multi −0.00003,
  noise). Funded (epoch ≈30s << 8min guard; test 43.2s). The adapter
  learned nothing beyond the frozen base — proves the target gap is
  unreachable by adaptation. Adapter weights are ours, bundled inside the
  lora joblib.

## 3. Era-overfit episode (xreg calendar)

In-ablation (smooth 2011–16 train windows), the `cal` past-future channels
[month sin, cos, trend] promoted decisively: 3/3 window wins, mean 0.3261
vs no-cov 0.3976 (+18.0% improvement); `both` also promoted (+5.9%);
`nat` (national-mean-delta past-only) rejected 0/3 (−51%, hurts). Test
mode = cal (lowest promoted mean). On the single test scoring — dominated
by the COVID-era regime break — cal reversed to 0.9002, WORSE than no-cov
control (0.8757) by +0.0245. Mechanism: calendar smoothers fit stable
seasonality in quiet windows and cannot survive a structural break; the
test-once discipline caught it instead of hiding it. This is the episode
that vindicates "TEST SCORED EXACTLY ONCE": any re-selection after seeing
0.9002 would have laundered the overfit.

## 4. Explicit null results

- **cov:** ablation table §2; promotion rule (wins≥2/3 AND mean
  improvement≥1%); nothing promoted; test ran no-cov. Past-only channels
  (trailing delta-std 3/6/12 + national-mean-delta, causal, per-step
  demeaned) actively hurt; past-future calendar (+0.2%) is decoration.
- **xreg:** `nat` rejected (0/3, hurts); `cal` promoted in-ablation then
  honestly failed test-once (§3). Net: XReg adds nothing test-honest.
- **lora:** val degrades monotonically with epochs at both ranks (ep1
  best, ep3 worst); selected r4/ep1 ties the frozen base on test
  (−3.1e-05). Adaptation cannot reach the 0.765-class target.

## 5. Backend verdict (arm64/darwin)

Smoke-benchmarked MLX vs pytorch-CPU before committing: torch load 3.1s /
infer 0.17s per joint step vs MLX load 1.3s / infer 6.76s — **torch-CPU is
~40× faster at inference**, parity maxabs 1.1e-5. All 3.0 variants run the
pytorch backend (frozen). MLX rejected on measured speed, not taste.

## 6. License / fetch notes + target verdict

- TimesFM 3.0 weights: `timesfm-non-commercial-license-v1.0`
  (NON-COMMERCIAL, research-compliant here). Checkpoint
  `google/timesfm-3.0-pytorch` (rev
  `43046b85ec22d584a13f8098c2ed39c889e129c2`) resolved from the HF cache
  at build; pointer only in `manifest.json`, weights NEVER committed.
- TimesFM 2.5-200M weights: Apache-2.0 (`google/timesfm-2.5-200m-pytorch`,
  rev `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`) via HF cache — the
  license-insurance fallback (xreg variant).
- LoRA adapter weights are ours (r4/α8/ep1, query_proj+value_proj,
  819200 params), bundled inside `artifacts/timesfm3_lora.joblib`.
- **Target verdict: 0.765 MISSED by +0.0567 (head 0.8217).** Covariates,
  XReg, and LoRA are all exhausted (tie-nulls + honest fail + val
  degradation), and context expansion is dead (2048==512). The gap is
  unreachable by TimesFM adaptation on this mask — reported, not chased:
  no threshold moved, no blending, no regime split. Any future attempt is
  a new problem spec, not drift in this build.

## 7. Verification

- `baselines/predict_timesfm_multi.py --target total` →
  `delta_rmse=0.821732 delta_rmse_24=0.785500 pass_rate=1.0000
  winner=timesfm3_multi` (per-variant lines in run output).
- `pytest tests/test_timesfm*.py -q`: all 5 suites green
  (correctness gate `pass_rate == 1.0`).
- `ruff check` scoped to new files (`baselines/train_timesfm_multi.py`,
  `baselines/predict_timesfm_multi.py`, 5 new tests); pre-existing tree
  lint out of scope, untouched.
- Pre-existing collisions: none — `tests/` held no live suites on this
  branch (only stale `__pycache__` from other branches, left untouched,
  nothing deleted).

## 8. Deviations

- **LoRA adapter bundling fix (train-code, caught by the lora suite).**
  The first lora build captured `pm.state_dict()` AFTER
  `merge_and_unload()`, which in PEFT 0.21.0 drops the lora_A/B parameters
  (verified: 160 keys / 819200 params pre-merge → 0 post-merge), silently
  bundling an empty adapter dict. Fixed by capturing pre-merge (+ assert
  non-empty) and retrained `--only timesfm3_lora`; RMSE pin unchanged
  (drift 0.0 — merge numerics untouched, bundling is metadata only). The
  regenerated `artifacts/timesfm3_lora.joblib` + one-line train fix land in
  the tests+docs commit (C3), not C2.
- No other drift: all 5 pins reproduced within 1e-6 (uni 0.0, multi
  1.1e-16, cov 1.1e-16, xreg 1.1e-16, lora 0.0). Serial one-job-one-log
  execution per plan; no overlapping heavy jobs. `X_train` not ported
  (10MB, unused — covariates are levels-derived + deterministic calendar;
  noted in manifest). No `problem.yaml` modification.
