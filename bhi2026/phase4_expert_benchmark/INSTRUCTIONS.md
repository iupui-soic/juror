# BHI 2026 expert alignment benchmark — rating rubric

## Goal
Rate the THEMATIC ALIGNMENT between each (capstone document, journal abstract) pair on a **0-4 scale**, and indicate whether their methodologies overlap.

## Rating scale

| Rating | Definition |
|---|---|
| **0** | **Unrelated.** Completely different problems and methods. |
| **1** | **Tangentially related.** Same broad domain (e.g., both healthcare) but different research questions AND methods. |
| **2** | **Adjacent.** Share either the research question OR the method, but not both. |
| **3** | **Substantially overlapping.** Share the same research question and use overlapping methods; contributions are distinct. |
| **4** | **Same research question.** Essentially address the same research question with comparable methods. |

## Methodology overlap (separate column)

- **yes** — methods clearly overlap (e.g., both use BERTopic on clinical text).
- **partial** — one technique is shared but the rest of the pipeline differs.
- **no** — methods are unrelated.

## Worked examples

| Capstone | Journal | Rating | Methodology |
|---|---|---:|---|
| Builds a patient-portal diabetes-management dashboard and runs a small usability study | Tests a smartphone diabetes self-management intervention in an RCT | 3 | partial |
| Builds a Power-BI dashboard for emergency department throughput | RCT of a clinical decision support system in the ED | 2 | no |
| Analyzes BRFSS depression data with SPSS regression | Develops federated deep-learning competing-risks models for post-COVID PASC | 0 | no |
| Implements REDCap-based data collection for autism behavioral coding | Tests a smartphone app for autism symptom monitoring | 1 | no |
| Both papers describe BERTopic applied to clinical notes in different institutions | | 4 | yes |

## How to fill the workbook

1. Open `benchmark_pairs.xlsx` and switch to your assigned **Rater 1** or **Rater 2** sheet.
2. For each row (one pair):
   - Read the **capstone_excerpt** (column B), **journal_title** (column C), **journal_abstract** (column D).
   - Optionally use the plain-text files in `pairs/pair_NNN_*.txt` if you prefer.
   - Enter your **rating** (0-4) in column F.
   - Enter your **methodology** overlap (yes/partial/no) in column G.
   - Add **notes** in column H if you flagged a borderline call.
3. Do NOT discuss with the other rater until both sheets are complete.
4. Save and return the workbook to the project lead.

## Notes

- The 100 pairs are a random mix of three sampling buckets (aligned anchors, misaligned anchors, ambiguous). The bucket label is NOT shown to you so that row order can't cue your rating.
- The `cosine_sim_hint` column gives a sentence-embedding similarity in [0, 1]. Use it only as a *very rough* prior; the LLM-judge it's calibrated against can be off in either direction.
- Target inter-rater Cohen's κ ≥ 0.6 (substantial agreement).

## What happens with your ratings

Both raters' sheets are returned to the project lead, who consolidates them into the **Adjudication** sheet. Cohen's κ is computed via `compute_kappa.py`. Disagreements ≥ 2 points are adjudicated jointly by the two raters. The adjudicated 100-pair set becomes the gold-standard benchmark for the LLM-judge calibration.
