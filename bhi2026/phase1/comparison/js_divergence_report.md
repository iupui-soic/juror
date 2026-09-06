# JS Divergence + Bootstrap Inference Report

**Pre-registered SAP.** Distribution: per-AMIA-domain top-1 share. JS in log₂ ∈ [0, 1]. Bootstrap n=2000, permutation null n=1000, RNG seed 42.

## Primary outcome

- **JS(capstone || journals) = 0.1247**, 95% bootstrap CI [0.0969, 0.1614].
- Soft-distribution variant: JS = 0.0169, 95% CI [0.0160, 0.0178] (reported as robustness).

## H2 — permutation test

Pooled the 287 capstones with 10374 journal abstracts and randomly split into groups of those sizes; recomputed JS each time.

- Null mean: 0.0067
- Null 95th percentile: 0.0123
- Observed JS: **0.1247**
- One-sided p-value: **p = 0** (n=1000 permutations)

**Interpretation.** Observed JS exceeds the null 95th percentile by a wide margin; we reject H0 of no distributional difference between the curriculum and the literature.

## H3 — per-AMIA-domain Δ (Bonferroni α=0.005)

| Domain | Δ (cap−jrn) | 95% CI | Bonferroni signif. |
|---|---:|:---:|:---:|
| D1 Health Sci | +0.0105 | [-0.0118, +0.0362] | · |
| D2 HIT | +0.0359 | [+0.0052, +0.0680] | · |
| D3 Soc/Beh | +0.0216 | [+0.0045, +0.0423] | **✓** |
| D4 InfoSci/CS | +0.0239 | [+0.0097, +0.0415] | **✓** |
| D5 Data Analytics | -0.3308 | [-0.3848, -0.2772] | **✓** |
| D6 Leadership | +0.1051 | [+0.0503, +0.1581] | **✓** |
| D7 Trans Bioinf | +0.0225 | [+0.0051, +0.0407] | **✓** |
| D8 Clinical Inf | +0.1187 | [+0.0820, +0.1568] | **✓** |
| D9 Public Health | +0.0036 | [-0.0067, +0.0173] | · |
| D10 Consumer Health | -0.0110 | [-0.0162, -0.0023] | · |

## Null / baseline experiments (Stage 4)

| Experiment | JS (mean) | 95% CI / range | n |
|---|---:|---:|---:|
| Observed: capstone vs journals | 0.1247 | [0.0969, 0.1614] | bootstrap 2000 |
| H2 permutation null | 0.0067 | [0.0022, 0.0137] | 1000 permutations |
| Random-split within journals | 0.0007 | [0.0002, 0.0014] | 200 splits |
| Random-split within capstones | 0.0248 | [0.0085, 0.0509] | 200 splits |
| Inter-journal pairs | 0.1082 | [0.0164, 0.3251] | 28 pairs |

## Notes

- Method-B distribution comes from BART-MNLI zero-shot top-1 assignments. The pre-registered robustness checks (DeBERTa-mnli secondary classifier, Method A OT, Method C LLM-judge) will re-validate these numbers.
- All bootstraps use document resampling with replacement; no stratification by year (yet).
- Permutation null is the appropriate H2 reference because it preserves the marginal AMIA distribution while shuffling the corpus label, matching the pre-registered conservative interpretation.