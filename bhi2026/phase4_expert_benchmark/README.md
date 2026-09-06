# BHI 2026 expert benchmark — 100 pairs

This package contains a 100-pair human-rating benchmark for cross-corpus
thematic alignment, sampled from 288 MS Health Informatics
capstone projects (2020-2024) and 10,374 Health Informatics journal abstracts
(2020-2025).

## What's in the package

- `benchmark_pairs.xlsx` — the main workbook. Sheets:
  - **Instructions** — full rating rubric.
  - **Rater 1** — fill in your ratings here (or in a copy named Rater_1_<yourname>.xlsx).
  - **Rater 2** — second rater's sheet.
  - **Pair Summary** — internal metadata (bucket labels are for the project lead, not for raters; ignore the `_internal_bucket` column).
  - **Adjudication** — to be filled by the project lead after both raters submit, to resolve disagreements.

- `pairs/` — one `pair_NNN_capstone.txt` and `pair_NNN_journal.txt` per pair, identical content to the spreadsheet but in plain text for raters who want to print or annotate offline.

- `INSTRUCTIONS.md` — the same rubric as the Instructions sheet, in markdown form for easy reading.

- `compute_kappa.py` — once both raters have filled in the workbook, run `python3 compute_kappa.py benchmark_pairs.xlsx` to compute Cohen's kappa (unweighted + linear-weighted + quadratic-weighted), per-bucket agreement, methodology-overlap kappa, and the confusion matrix.

## Workflow

1. Each rater fills in their own sheet (or saves a copy of the workbook).
2. Do NOT compare notes with the other rater until both are done.
3. When both are done, the project lead consolidates ratings into the Adjudication sheet, runs `compute_kappa.py`, and adjudicates disagreements (typically pairs ≥ 2 rating points apart).
4. The adjudicated set becomes the gold-standard 100-pair benchmark for calibrating the LLM-judge against humans (target: Cohen's κ ≥ 0.5 LLM-vs-adjudicated).

## Target

- Inter-rater Cohen's κ ≥ 0.6 (substantial agreement).
- Inter-rater Cohen's κ < 0.4 (poor agreement) → the rubric needs revision before the benchmark can be used.

## Time estimate

- ~3 minutes per pair × 100 = ~5 hours per rater, distributable across multiple sessions. The workbook saves your progress as you go.

## Reproducibility

`benchmark_pairs.xlsx` is the released benchmark, with both raters' ratings and
the adjudicated gold standard already filled in (sheets Rater 1, Rater 2,
Adjudication, Pair Summary). The analysis scripts read only this file:

- `compute_kappa.py benchmark_pairs.xlsx` — inter-rater Cohen's κ (unweighted +
  weighted), per-bucket agreement, methodology κ, confusion matrix.
- `compute_kappa_complete.py` / `agreement_panel.py` — the full estimator panel
  (weighted κ, ICC, Krippendorff α, Gwet AC2) used in the paper.
- `build_gold_standard.py` — re-derives the adjudicated gold standard from the
  two raters (deterministic rule); `tier2_calibration.py` scores the LLM-judge
  against it.
- `joint_agreement.py` — rating × methodology coupling analysis.

`build_benchmark.py` documents and regenerates the construction procedure: a
deterministic (seed = 42) sampler that draws 25 aligned / 25 misaligned / 50
ambiguous capstone–journal pairs (abstract-bearing only), shuffles them, and
writes a blank scaffold under `rebuild/` (it never overwrites the released
file). The capstone corpus is IRB-restricted, so capstone excerpts populate only
when the source texts are present.
