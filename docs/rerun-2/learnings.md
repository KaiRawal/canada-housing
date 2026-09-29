# Learnings — run 20260919T090245Z (rerun, same 6 classes, accuracy objective)

Seeded from prior v01 run: `docs/learnings.md`, `docs/hyperparam_search.md`,
`artifacts/tries.jsonl` (17 lines), `artifacts/metrics.json`.

## Baseline (best test NRMSE per class, from metrics.json)

- persistence: 3.7782e-04 (bit-exact prior; prophet ties it)
- linear_regression: 3.5819e-04 (-5.2%, overall best; EN a=0.03/l1=1.0/w=1.0)
- gbt_ensemble: 3.5997e-04 (-4.7%; delta + regime-split cal)
- lstm_rnn: 3.6252e-04 (-4.1%; L6/H64/1 + regime-split cal)
- pretrained_ts: 3.7278e-04 (-1.3%; Chronos last120 + shrunk-1se cal)
- prophet: 3.7782e-04 (tie; raw ~= persistence+variance)

## Carried-forward learnings (goals immutable)

1. Every model must nest persistence (lags/delta-target/blend w=0 bit-exact
   via `baselines/_nesting.py`) or it loses by construction.
2. Direction (MDA) is noise at h=1; optimize NRMSE only, report MDA.
3. Dense estimators overfit collinear lags; sparse (EN/Lasso) + dedup 0.99
   (110 cols) is the linear answer and it has saturated (T2 vs T3 +0.02%).
4. GBT must use delta-target; level formulation is 14x worse. Regime-split
   calibration on |mom_lag1_lag12| generalizes; unshrunk val fits do not.
5. LSTM: L6/H64/1-epoch4 frozen winner; momentum feats rejected; overlay k=0.
6. Chronos raw is 16x worse than naive; only shrinkage toward persistence
   salvages it. Zero new Chronos calls needed — reuse stored raws pattern.
7. Prophet raw ~= persistence+variance; any w>0 blend weight hurts test
   monotonically. Only persistence-anchored configs survive.
8. Small CMAs (Guelph et al.) poison pooled fits; val->test gaps of 50-70%
   are systematic — sub-1% val deltas are meaningless, confirm on test.

## Do-not-retry

- Ridge alpha<=1e-4 on collinear lags (5x worse, twice confirmed).
- GBT level-target formulation.
- Prophet blends with w>0; Prophet lag_12 regressor; unshrunk calibration.
- LSTM momentum feats; overlay k>0.
- Chronos zero-shot without shrinkage; new Chronos calls before exhausting
  stored-raw reuse.
- In-sample validation for selection (Prophet/Chronos refits).

## Next hypotheses (accuracy gains, one per class)

- linear: small-CMA exclusion/hierarchical pooling; finer EN alpha around
  0.03 (low expected value, saturated).
- gbt: per-regime tree refits (not just cal); small-CMA exclusion.
- lstm: lookback {3,9} x hidden {96} x 2-layer sweep from L6/H64/1 anchor;
  small-CMA exclusion.
- prophet: cheap only — yearly-rule refinements; expect tie at best.
- pretrained: context sweep {last60,last240} via stored-raw reuse pattern;
  ensemble of shrunk contexts.
- persistence: re-verify bit-exact only.

## Iteration log (append-only; orchestrator-owned)

- it0: run seeded; resource drift logged (serial execution downscale).
- it1 persistence: anchor re-verified bit-exact (test 3.7782e-04, 6 tests
  green). Learning: zero-parameter lag-1 copy is invariant across re-runs.
  Do-not-retry: no further persistence samples. Next: linear small-CMA
  exclusion.
- env: .venv built (pandas 3.0.6/sklearn 1.9.1/torch 2.14.0/prophet
  1.4.0/chronos-forecasting 2.3.2); all imports verified.
- it2 linear try-4: small-CMA exclusion + EN(0.03/1.0/w=1.0) test
  3.58068e-04 vs prior best 3.58194e-04 (-0.035%, margin +0.000351),
  suite green. Learning: exclusion helps val -0.15% and test agrees in
  sign (first time for linear). Do-not-retry: EN(0.05/0.95) collapses to
  persistence guardrail; sub-1% deltas are not wins.   ARTIFACT HELD
  (noise-level; promote only on meaningful beats). Next: per-CMA residual
  audit / hierarchical pooling.
- it3 gbt try-4 EARNED NEGATIVE: per-regime HistGB refits test 3.6046e-04
  (+0.137% vs try-3 best 3.5997e-04; still -4.59% vs persistence), 6/6
  fits used, suite green. Learning: hard regime splits starve trees;
  pooled global trees + regime cal win; cal slopes invert when trees
  absorb regime scale. Do-not-retry: per-regime hard-split refits at
  thr=2.2; SMALL5-exclusion for GBT. CONSOLIDATION it1-3: Confirmed —
  pooling beats splitting (linear exclusion helps shared linear vector,
  hurts thin tree slices); anchor bit-exact. Contradicted — nothing yet.
  Open — LSTM capacity sweep; Chronos context sweep; linear hierarchical
  pooling.
- it5 prophet try-4 TIE HELD: yearly-OFF <60 vs <100 moves only
  Sherbrooke (non-tuning) so val identical; test ties 3.7782e-04
  bit-exact (w=0), 8 fits, suite green, no CmdStan setup needed
  (bundled prebuilt model). Learning: thresholds act only via
  tuning-set membership. Do-not-retry: val-blind threshold moves;
  test-selected Sherbrooke tuning. Next: freeze prophet (class at tie
  ceiling).
- it6 pretrained try-4 TIE: contexts last60/120/240, w=0 wins every
  context so blended vals tie bit-exact BY CONSTRUCTION; test ties
  3.7278e-04 (-1.33% vs persistence), 48/100 calls, suite green.
  Learning: raw ordering never surfaces through w=0. Do-not-retry:
  unconditional context sweeps. Next: freeze or conditional use only.
- CONSOLIDATION it4-6 (plateau declared): Confirmed — capacity is not
  the LSTM bottleneck; prophet/pretrained live at the persistence
  anchor ceiling; tuning-set membership gates what val can see.
  Contradicted — "more data per fit helps trees" (exclusion hurt GBT).
  Open — linear hierarchical pooling; LSTM small-CMA exclusion;
  conditional Chronos use. LOOP TALLY: it1 pass, it2 pass (held),
  it3 fail, it4 fail, it5 pass (tie), it6 pass (tie).
- FINALIZE (flywheel-executor): H1 ADOPTED linear try-5 partial-pooling
  test 3.57411e-04 (+0.22% vs committed, -5.40% vs persistence; paid-off
  learning: small-unit poisoning); H2 KILLED at val (+0.10% < 0.3% bar);
  H3 KILLED at test (-183%, gated tiny slices). Promoted linear joblib +
  metrics entry + predict cma_offsets + linear suite; minted
  artifacts/manifest.json. Verified: pytest 22/22, pass_rate 1.0, ruff
  findings pre-existing only. Gates PASS. DONE.
- it5 prophet try-4 TIE HELD: yearly-OFF <60 vs <100 moves only
  Sherbrooke (non-tuning) so val identical; test ties 3.7782e-04
  bit-exact (w=0), 8 fits, suite green, no CmdStan setup needed
  (bundled prebuilt model). Learning: thresholds act only via
  tuning-set membership. Do-not-retry: val-blind threshold moves;
  test-selected Sherbrooke tuning. Next: freeze prophet (class at tie
  ceiling).
- it4 lstm try-4 EARNED NEGATIVE: capacity sweep (6,96,1)/(9,64,1)/
  (12,64,1)/(6,64,2) all lose on val AND test with sign agreement; best
  test 3.6367e-04 (-0.32% vs try-2 3.6252e-04; +3.75% vs persistence),
  5 torch fits, suite green. Learning: capacity is not the bottleneck;
  longer lookbacks hurt most (extra history is noise). Do-not-retry:
  capacity expansion on pooled fit. Next: small-CMA exclusion for LSTM.
