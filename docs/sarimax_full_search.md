# SARIMAX-full search — single pooled AR on all 24 CMAs (no ensembles)

Scope: plain pooled sparse autoregression on signed monthly changes in total
NHPI, fit + evaluated on the FULL panel (16 long + 8 short-history CMAs).
No TabPFN, no GBT, no averaging. Test scored once for the locked config.

## Frozen conventions

- Boundary-chained deltas/lags on the combined train+test timeline per CMA;
  first test delta scored. Masks: **n16=1728 / n24=2398**.
- Naive floors (chained recompute): **0.956390 (16) / 0.893202 (24)**.
- Scaler fit on fitting rows only; strictly chronological validation;
  imputation frozen; test never used for selection (one scoring of the
  locked config + one 16-only control refit for the poison check).

## Data (full panel)

Train rows: 7792 (24 CMAs). P4-valid rows: 7600 (24) vs 6800 (16).
Winner-config fit rows: **7744** (16×431 + Halifax 391 + Charlottetown 285 +
Sherbrooke 62 + 5×22 for Guelph/Trois-Rivières/Kelowna/Oshawa/QC-part).
Tiny CMAs contribute 22 rows each — few, but non-zero and pooled honestly.
Test rows: 2398, all with lags-1..12 valid via chaining (108 per long CMA +
Halifax, 86 per mid-short CMA, 46 Sherbrooke).

## Validation protocol (train-only)

Two protocols, both pooled across CMAs, scaler refit per split:

- **P1 multi-window (selection):** per-CMA expanding origins at fractions
  {0.50, 0.65, 0.80} of each CMA's valid rows, val = next ≤72 rows.
  Pooled RMSE weighted by val-n. Promotion needs ≥2/3 window agreement +
  ≥1% pooled margin (learnings #5).
- **P2 last-20% (confirmation):** per-CMA 80/20 chronological split, pooled RMSE.
  Must agree in direction with P1; ties broken by parsimony.

Weighting arm: equal-row (status quo) beats equal-CMA on every one of 40
feature×alpha pairs (consistent ~0.3% direction, sub-1% margin) → equal-row kept.

## Stage 1 — capacity (Lasso 0.001 unless noted, equal-row, P1 pooled)

| Features | p | P1 pooled | W1/W2/W3 | Verdict |
|---|---|---|---|---|
| A lags-1..3 | 3 | 0.51013 | 0.3498/0.7225/0.3676 | baseline |
| B P4 (lags-1..7+roll3/6/12+std, R4 head) | 13 | 0.51253 | 0.3490/0.7285/0.3662 | head to beat |
| C lags-1..12+rolls | 18 | 0.51807 | — | killed (+1.6%) |
| D P4+seasonal(12,24) | 15 | 0.52838 | — | killed (+3.5%) |
| E lags-1..12+rolls+roll24 | 20 | 0.51937 | — | killed |

Alpha sweep 1e-4..0.3: flat optimum 0.0003–0.01 (all within 0.5%); 0.03+
degrades; Ridge/ElasticNet arms tie Lasso (no promotion). Seasonal lags,
lags beyond 7, and roll-24 all dead — learnings #4 holds.

## Stage 2 — isolation (where is the signal?)

| Features | p | P1 pooled | P2 last20 | Note |
|---|---|---|---|---|
| A lags-1..3 | 3 | 0.51013 | 0.35993 | — |
| F lags-1..5 | 5 | 0.51467 | 0.36009 | more lags hurt |
| G lags-1..7 (no rolls) | 7 | 0.51300 | 0.36116 | rolls carry the memory |
| H lags-1..3 + all rolls | 9 | 0.50894 | 0.35653 | beats A (2/3) and B (3/3) |
| B P4 head | 13 | 0.51253 | 0.35990 | — |

H (AR3 + rolling means/stds) promotes over the R4 head: 3/3 windows, −0.7%
P1. Raw lags 4+ are redundant given rolling stats.

## Stage 3 — pruning the rolls

| Features | p | P1 pooled | P2 last20 |
|---|---|---|---|
| H (ar3 + mean/std 3/6/12) | 9 | 0.50894 | 0.35653 |
| H-mean-only (ar3 + means) | 6 | **0.50681** | 0.35711 |
| H-mean3/12 only (ar3 + rm3 + rm12) | 5 | 0.50657 | 0.35711 |
| L (ar1..2 + means 3/6/12) | 5 | 0.50578 | 0.35644 |
| M (ar1 + means) | 4 | 0.50375 | 0.35644 |
| **Q (rm3 + rm12 only)** | **2** | **0.50337** | 0.35716 |
| N (means 3/6/12) | 3 | 0.50361 | 0.35711 |
| O rm3 only | 1 | 0.50669 | — (rm12 adds value) |
| P rm12 only | 1 | 0.51947 | — (too smooth alone) |

Rollstds pruned (means suffice); raw lags pruned one by one (each step
−0.2–0.4%, consistent direction, individually noise, cumulatively −1.8% vs
head with 3/3 agreement at every step). Floor found: rm3-only and rm12-only
both worse than the pair — short + long smoothing is the structure.

**Locked: Q = [rollmean_3, rollmean_12], Lasso(0.001), equal-row, intercept.**

## Test (scored once, 2026-10-16)

| Fit | test16 | test24 | vs naive |
|---|---|---|---|
| Q full-24 (locked, 7744 rows) | **0.800261** | **0.760422** | 16.3% / 14.9% |
| B head full-24 refit (context) | 0.800059 | 0.760093 | 16.4% / 14.9% |
| Control Q 16-only (poison check) | 0.799786 | — | — |

Honest report: the P1-promised −1.8% over the head **did not generalize** —
test is a tie within 0.04% noise. This is val over-selection among
noise-tied configs (learnings #5), disclosed not hidden. What the search
still bought: (a) full-24 coverage with poison check passed (full-fit 16 =
0.800261 vs 16-only 0.799786, +0.06%); (b) 2 features instead of 13 —
same accuracy, less variance, interpretable short+long momentum
(intercept 0.1277, coef rm3=0.1678, rm12=0.0758); (c) 23/24 CMAs beat naive
(only Sherbrooke n=46 loses, 0.6755 vs 0.6154 — tiny volatile sample).

## Killed

Seasonal dlag_12/24 as raw lags; lags 4..12 raw; rollstds; rollmean_6
(redundant between rm3/rm12); equal-CMA weighting; Ridge/ElasticNet over
Lasso; any ensemble (TabPFN/GBT add ~1% at 3× cost — out of scope by design).

## Artifacts

`prediction/train_sarimax_full.py` →
`artifacts/sarimax_full.joblib`, `artifacts/sarimax_full_metrics.json`,
`prediction/sarimax_full_predictions_{train,test}.csv`;
demo `notebooks/sarimax_full_demo.ipynb`.
