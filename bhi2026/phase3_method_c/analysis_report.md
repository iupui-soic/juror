# Method C — LLM-as-judge analysis

**Pairs rated:** 495 successful / 496 attempted (1 JSON parse error).
**Model:** Claude Sonnet 4.6.
**Cost (estimated):** ~$1.70.

## Validation against cosine similarity (Tier 2 ranking proxy)
- Spearman ρ(cosine, rating) = **0.557** (p = 9.8e-42)
- Pearson r(cosine, rating) = **0.560** (p = 3.5e-42)

The pre-registered Tier-2 threshold for OT pair-level rank consistency is ρ ≥ 0.4. **Met.**

## Rating distribution by sampling bucket

| Bucket | n | r0 | r1 | r2 | r3 | r4 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 99 | 85 | 13 | 1 | 0 | 0 | 0.15 |
| Q2 | 100 | 77 | 23 | 0 | 0 | 0 | 0.23 |
| Q3 | 100 | 50 | 49 | 1 | 0 | 0 | 0.51 |
| Q4 | 100 | 21 | 68 | 11 | 0 | 0 | 0.90 |
| Seed | 96 | 30 | 55 | 11 | 0 | 0 | 0.80 |

**Striking finding: zero pairs were rated 3 or 4 ("same research question") across the entire 496-pair set.** The corpus-vs-corpus alignment tops out at "adjacent" — shared research question OR method, never both.

## Per-capstone-topic seed pair outcomes

Method A flowed each capstone topic to its top-3 journal topic destinations. For each (cap_topic × jrn_topic × rank) combination we sampled multiple document pairs and asked the LLM-judge to rate them.

| Cap topic | n | r0 | r1 | r2 | mean | Best destination found |
|---|---:|---:|---:|---:|---:|---|
| 5 5_pathology_extraction_pathology reports_text | 12 | 2 | 4 | 6 | 1.33 | — |
| 6 6_course_global_institute_[institution] | 12 | 0 | 10 | 2 | 1.17 | — |
| 4 4_spss_depression_statistical_[PERSON] | 12 | 2 | 8 | 2 | 1.00 | — |
| 0 0_emr_database_management_frontend | 12 | 3 | 9 | 0 | 0.75 | — |
| 3 3_symptoms_schizophrenia_reporting_sql | 12 | 4 | 7 | 1 | 0.75 | — |
| 2 2_[state]_prenatal_2019_ns | 12 | 5 | 7 | 0 | 0.58 | — |
| 7 7_renal_extraction_pathology_pathology reports | 12 | 5 | 7 | 0 | 0.58 | — |
| 1 1_hands_[PERSON]_phd_redcap | 12 | 9 | 3 | 0 | 0.25 | — |

## Qualitative findings

Three patterns emerge from the per-pair justifications:

1. **"Method-adjacent, question-divergent" — every rating-2 pair.** Document pairs share either a broad method (NLP, statistical regression) or a domain (physical activity, clinical text) but never the same research question. The LLM correctly never granted a 3 or 4 for such pairs.

2. **"Practice vs research" structural gap.** Capstones describe building, deploying, and applying tools; journal abstracts describe testing, measuring, and validating findings. Even when topics are semantically nearby, these are *adjacent activities*, not equivalent research questions. This is the structural finding the BHI paper should lead with.

3. **"False-positive seeded flows" — Method A noise.** Some Method A top-flow destinations are driven by incidental terms (data tools like REDCap, geographic mentions like [STATE]) rather than substantive topic overlap. In Capstone Topic 1 (REDCap-heavy work), 9/12 seeded pairs landed at rating 0 — the flow was a lexical proxy, not a real semantic match.

## Implications for the BHI paper

- **The 0/4 rating ceiling is the substantive finding, not a sampling failure.** All three methods agree the corpora differ; Method C quantifies *how far* — they don't share research questions even where they share topics.
- **Per-capstone-topic table reveals interpretable variation** in where the curriculum is most/least engaged with the literature. Pathology-NLP (T5) is the strongest match; REDCap-tooling (T1) is the weakest.
- **Method A flows have a measurable false-positive rate** that Method C can quantify — this is itself a Phase 3 robustness reporting point.
