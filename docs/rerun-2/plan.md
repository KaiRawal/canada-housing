# Plan — run 20260919T090245Z (housing-baselines)

> Goals immutable. This plan never proposes or applies edits to
> `.flywheel/problem.yaml` goal/gates/thresholds. Pinned thresholds below
> are a VERBATIM copy of `gates:`. Any researcher threshold idea would be
> recorded under Rejected — none was proposed.

- Run: `.flywheel/runs/20260919T090245Z/`
- Inputs: `logs/` 01–06 + `learnings.md` + researcher report (ranked H1/H2/H3,
  horizon change out of scope, per `decisions.md` 2026-09-19 research entry)
  + `.flywheel/problem.yaml`
- Model: opencode/muse-spark-1.3-contributor-free
- Repo root: `/Users/bigcamel/Oxford/CoCurricular/AGF/ProjectRepositories/canada-housing`

## 1. Learnings carried forward (condensed)

### Confirmed
- C1. Every variant must nest the persistence anchor (lags / delta-target /
  `blend_w=0` bit-exact via `baselines/_nesting.py`) or it loses by construction.
  Anchor re-verified bit-exact this run (`logs/01`).
- C2. Direction metric at h=1 is noise; select on NRMSE-equivalent only, report
  direction as auxiliary (`learnings.md` #2).
- C3. Dense estimators overfit collinear lags; sparse elastic-net + dedup
  (~110-col) set is the saturated linear answer (`learnings.md` #3; `logs/02`).
- C4. Tree ensembles must use delta-target; level formulation is ~order-of-magnitude
  worse. Regime-split calibration on large-move score can generalize; unshrunk
  val-only affine fits do not (`learnings.md` #4; `logs/03` cal slopes 0.44/0.58).
- C5. Sequential-model pooled capacity is saturated at the small anchor config;
  longer windows / wider / deeper lose on val AND test with sign agreement
  (`learnings.md` #5; `logs/04` L9/L12 + H96 + 2-layer all lose).
- C6. Pretrained zero-shot raw is far worse than naive; only shrinkage toward
  the persistence leg salvages it. Stored-raw reuse covers required sweeps;
  zero new calls needed (`learnings.md` #6; `logs/06` 48/100 calls, `w=0` wins).
- C7. Additive-model raw ~= persistence + variance; any `w>0` blend weight hurts
  monotonically. Only persistence-anchored configs survive (`learnings.md` #7;
  `logs/05` blend curve monotonic harm).
- C8. Pooling beats hard splitting. Small-unit exclusion helps the shared linear
  vector (val −0.15%, test −0.035% with sign agreement — first time) but hurts
  thin tree slices. Val→test gaps of ~50–70% are systematic; sub-1% pooled-val
  deltas are noise without per-unit confirmation (`learnings.md` #8;
  `logs/02` vs `logs/03` consolidation).

### Contradicted
- X1. "More data per fit helps trees" — exclusion hurt tree val (`logs/03`).
- X2. "Capacity is the sequential bottleneck" — capacity sweep lost on val and
  test with sign agreement (`logs/04`).
- X3. "Context length / yearly-rule refinements move the anchored score" —
  both tie bit-exact by construction once `w=0` selects the persistence leg
  (`logs/05`, `logs/06`).

### Open
- O1. Linear hierarchical / partial pooling vs plain small-unit exclusion
  (+ per-unit residual audit; Sherbrooke ablation).
- O2. Sequential-model small-unit exclusion (untried; capacity axis exhausted).
- O3. Conditional / gated pretrained use (active solely where val proves a local
  `w>0` win) — only allowed pretrained direction.
- O4. Soft large-move upweighting + shrunk calibration for trees (keeps global
  trees, fixes proven hard-split failure).

## 2. Baseline diagnosis (why persistence still anchors)

- Target is near-random-walk at h=1: lag-1 copy is a near-optimal point forecast;
  every class beats or ties it only via a nested guardrail (`blend_w=0` /
  delta-target / shrunk-to-persistence leg).
- All challengers exhibit systematic val→test gaps (~49–55% this run:
  linear +48.9%, trees ~53%, sequential ~54.5%); pooled-val selection is
  dominated by the majority regime / majority units and cannot reliably rank
  thin-slice (large-move / small-unit) refits.
- Unshrunk val-only calibration over-shrinks (slopes << 1.0, inverted across
  regimes) and does not transfer; raw pretrained / additive legs are 10–16x
  above the shrunk anchor so `w=0`/`w>0`-gated selection necessarily restores
  the tie.
- Threshold / context moves that change no tuning-unit membership are val-blind
  by construction and re-tie bit-exact. Plateau declared: 2 earned negatives +
  2 ties + 1 noise-level nominal pass across 6 classes (`decisions.md` tally).

## 3. Do-not-retry (flywheel-executor MUST honor)

### 3a. Researcher-affirmed dead ends (8 — honored; report file absent from run
### dir, reconstructed from `decisions.md` research entry + `logs/01–06` + `learnings.md`)
- R1. Ridge `alpha<=1e-4` on collinear lags (5x worse, twice confirmed).
- R2. Tree level-target formulation (order-of-magnitude worse; delta-target only).
- R3. Additive-model `w>0` blends / lag_12 regressor / unshrunk calibration
  (monotonic harm; `w=0` anchor only).
- R4. Sequential-model momentum features / overlay `k>0` (rejected).
- R5. Pretrained zero-shot without shrinkage; new pretrained calls before
  exhausting stored-raw reuse.
- R6. In-sample validation for selection (pretrained / additive refits).
- R7. Finer elastic-net alpha tuning around the saturated optimum on pooled val
  alone; claiming sub-1% pooled-val deltas as wins.
- R8. Moves val cannot see: unconditional hard regime-split refits,
  unconditional context sweeps, val-blind rule-threshold refinements,
  test-selected unit tuning, unconditional shrunk-context ensembling with <2
  val beaters. Horizon change is explicitly OUT OF SCOPE (researcher).

### 3b. Run-accumulated (try-4 specifics, same force)
- Persistence: no re-sample / re-tune (zero hyperparameters, bit-exact).
- Linear: no `alpha=0.05/l1_ratio=0.95` pooled re-sample (collapses to
  persistence guardrail `blend_w=0.0`); no try-4-as-confirmed-win claims.
- Trees: no per-regime hard-split refits at the tried threshold with/without
  small-unit exclusion + unshrunk val-only affine cal (base refit lost on test
  +0.137%; exclusion lost on val).
- Sequential: no capacity expansion on pooled fit (longer lookbacks, widening,
  deepening exhausted; winner still lost val +0.23% / test +0.32%).
- Additive: no yearly-rule / smoothing-threshold refinements that change no
  tuning-unit membership; no test-selected unit tuning.
- Pretrained: no unconditional context sweeps under `w=0`-selecting shrinkage;
  no new calls chasing raw-quality ordering; no unconditional ensembling.

## 4. Chosen variants + why (gate margins decide; hints bias, never override)

- Hint disposition: `problem.yaml: hints` model-list hint (six named classes)
  is already satisfied by the six surrogate variants + six suites. Recorded as
  USED/SATISFIED. No hint justifies breaching §3 or the promotion rule. No other
  hint is acted on.

### H1 — Sparse-estimator hierarchical / partial pooling — FIRST
- What: per-unit residual audit of the nominal try-4 best, then hierarchical /
  partial-pooling (shared vector + shrunk unit offsets) vs plain small-unit
  exclusion vs next-unit ablation; sparse elastic-net family only; val-gated,
  single honest test score.
- Why FIRST: the ONLY direction with val+test sign agreement this run
  (val −0.15% → test −0.035%); all other axes produced earned negatives or
  by-construction ties. Budget: 2–4 samples.
- Kill criteria: kill if (a) per-unit audit shows excluded units are not
  systematic outliers, OR (b) pooled val does not improve over the try-4
  anchor, OR (c) test delta stays sub-noise (<~0.1% or reverses). Do not promote
  noise; hold artifact per §6.

### H2 — Tree soft large-move upweighting + shrunk calibration
- What: keep GLOBAL trees (no hard regime-split refits); upweight large-move
  rows softly (sample weights, not splits) and/or keep regime-split calibration
  ONLY with shrinkage toward (slope 1.0, intercept 0.0); selected by pooled val,
  single honest test score.
- Why: directly fixes the proven try-4 failure (hard splits starve thin slices;
  unshrunk cal over-shrinks). Budget: 3–5 samples.
- Kill criteria: kill if val-selected config does not beat the class best-so-far
  on the honest test score, or cal slopes again invert/collapse (<<1.0), or
  exclusion/weighting hurts pooled val as in try-4. Never split tree fits by
  regime (§3 R8).

### H3 — Conditional / gated pretrained use ONLY
- What: pretrained leg active SOLELY where val proves a local `w>0` win
  (per-unit or per-regime gate); otherwise the frozen shrunk-`w=0` anchored
  artifact stands. Zero new calls — stored-raw reuse only.
- Why: pure upside above a tied floor; unconditional sweeps tie by construction.
  Budget: 1–2 samples, 0 new calls.
- Kill criteria: kill (freeze class) if no gate beats the `w=0` val anchor in ≥2
  units/regimes, or gated test score does not beat the class tie, or any proposal
  requires fresh calls. Freeze on tie; never unconditional.

- Explicit non-variant: horizon change — OUT OF SCOPE per researcher; executor
  must not sample it.

## 5. Pinned thresholds (VERBATIM copy of `gates:`, unadjusted)

```yaml
gates:
  correctness: "pass_rate == 1.0"
  budgets: { samples: 200 }
```

- Correctness resolves ONLY via the global `predict` entrypoint's `pass_rate`
  (per-class `metrics.json` files carry no `pass_rate` key; variant-level
  pass/fail projections in `logs/01–06` are not gates).
- Budgets are a budget, not a score.
- Researcher threshold adjustments: REJECTED — none proposed. No goal/gate
  change at any consolidation point.

## 6. Commit split (2 commits, each green on its own)

1. **Commit 1 — code + artifacts + tests for promoted winners only.**
   Any H1/H2/H3 winner that clears its kill criteria AND the §7 promotion rule:
   tuned code, class artifact(s), class test suite updates, sandbox evidence.
   Green = class suite(s) + nesting suite pass under `timeout 600`.
2. **Commit 2 — docs / manifest / provenance.**
   `docs/hyperparam_search.md` (per-class search + winning configs),
   `artifacts/metrics.json`, `artifacts/manifest.json`, learnings roll-up.
   Green = full `predict` gate (pass_rate) + full test sweep pass under
   `timeout 600`. No code changes in this commit except manifest/metrics
   minting.

- Autonomous mode: split picked as above and logged (no interactive question).

## 7. flywheel-executor workflow (declarative)

- Constraints honored: `.venv`-only installs; `timeout 600` on every gate;
  `heavy_bg: false`; SERIAL execution — one-job-one-log (resource drift to
  serial downscale already logged in `decisions.md`; 16 GB promise vs ~146M
  free observed).
- `nohup` jobs: none heavy expected (CPU minutes on 14 logical cores). If a
  background install/fit is ever needed: exactly ONE `nohup … > <run>/logs/<job>.log 2>&1 &`
  at a time, serial; record pid in `decisions.md`.
- `pgrep` polls: `pgrep -af "<job-pattern>"` to confirm start/stop; poll no
  more often than every 30 s; never launch a second job while one matches.
- Gates under `timeout 600` (all via `.venv`):
  - `timeout 600 .venv/bin/python -m pytest tests/test_baselines_<class>.py -q`
    per touched class + nesting suite;
  - `timeout 600 .venv/bin/python -m ruff check baselines/ tests/` (or repo's
    configured ruff entrypoint);
  - notebook gate (`nbconvert --execute`) only if notebooks ship (currently none).
- Budgets: stop sampling a hypothesis at its §4 sample cap; total samples stay
  within `gates.budgets.samples`; log every sample in `sandbox/<variant>/`.
- Artifact-best-so-far promotion rule (significance-before-celebration):
  linear try-4 nominal best (`3.58068e-04` vs prior `3.58194e-04`, −0.035%) is
  HELD, NOT promoted — sub-noise per C8. Promote it at finalization ONLY if
  nothing beats it; promote any H1/H2/H3 challenger ONLY on a val-gated,
  per-unit-confirmed, above-noise honest test beat with suites green and
  artifacts untouched until promotion.
- `manifest.json` minting (per gate-contract): sha256 + shapes + dependency
  versions for every delivered artifact; mint in Commit 2 only, after winners
  freeze.
- Final gate: full `baselines/predict.py --target total` (the `blackbox.predict`
  entrypoint) to resolve `gates.correctness` (`pass_rate == 1.0`); per-model
  NRMSE from `evaluation.py` selects per-class winners (tuning only, not gates).
- Deliverables on branch `flywheel/housing-baselines-<ts>`: 6 class artifacts +
  `artifacts/metrics.json` + `artifacts/manifest.json`; 6 class test files +
  nesting test; `docs/hyperparam_search.md`. Never touch outside
  `target.scope`; never hand-edit `.flywheel/` mid-run.
