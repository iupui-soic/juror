# Capstone–Literature Topical Alignment

Code and derived artifacts for a study measuring the topical alignment between
Master of Science in Health Informatics capstone projects (n = 288, Spring 2020
– Fall 2024) and the contemporaneous health-informatics journal literature
(eight journals, 2022–2025, retrieved from PubMed). Four independent methods
triangulate the comparison:

- **Method A** — BERTopic topic models per corpus + optimal-transport (OT)
  alignment between topic distributions, with permutation/rotation null
  baselines.
- **Method B** — zero-shot NLI classification of every document into the ten
  AMIA informatics domains (BART-MNLI primary, DeBERTa-MNLI robustness), with
  Jensen-Shannon divergence between the corpus-level distributions.
- **Method C** — a pairwise LLM judge rating (capstone, journal-abstract) pairs
  0–4 for thematic alignment, calibrated against a 100-pair expert-rated gold
  standard.
- **Method D** — the contribution: an OT→LLM bridge that rates only the topic-
  to-topic flows the transport plan selects, reproducing pair-level judgments
  at a small fraction of the LLM cost.

A three-tier validation protocol (paraphrase-recovery, human gold-standard
calibration, cross-method convergence) plus embedding, classifier, and
cross-LLM robustness checks support the main results.

## Repository layout

| Path | Contents |
|---|---|
| `extract_text.py`, `bhi2026/build_metadata.py`, `bhi2026/redact_pii.py`, `bhi2026/extract_sections.py` | Stage 0 corpus construction (requires the restricted corpus; see Data availability) |
| `bhi2026/fetch_journals.py`, `bhi2026/journals/manifest.json` | Journal-corpus retrieval via NCBI Entrez + the manifest of what was retrieved |
| `bhi2026/run_amia_zeroshot*.py`, `run_amia_deberta.py`, `compare_bart_vs_deberta.py`, `js_divergence.py` | Method B + JS-divergence inference |
| `bhi2026/method_a_*.py`, `hdbscan_sweep.py`, `umap_hdbscan_sweep.py` | Method A, nulls, clustering sweeps, PubMedBERT robustness |
| `bhi2026/method_c_llm_judge.py`, `analyze_method_c.py` | Method C |
| `bhi2026/generate_paraphrases.py`, `paraphrase_recovery.py`, `paraphrase_method_c.py` | Tier-1 paraphrase-recovery validation |
| `bhi2026/method_d_*.py` | Method D + ablations, pair-aggregate comparison, cross-LLM robustness |
| `bhi2026/stage5_convergence.py` | Cross-method convergence analysis |
| `bhi2026/phase4_expert_benchmark/` | 100-pair expert benchmark: builder, rater instructions, agreement/κ analyses, adjudicated gold standard (`gold_standard.csv`), Tier-2 calibration |
| `bhi2026/phase5_hardening/` | Bootstrap CIs, outlier sub-clustering, expert faithfulness audit |
| `bhi2026/phase1*/ phase2*/ phase3*/ phase4*/` | Committed result artifacts (scores, ratings, reports, figures), keyed by `capstone_id`/`pmid` |
| `tools/check_release_pii.py` | Maintainer pre-commit screen (see below) |

## Setup

Python 3.9. Then:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # used by redact_pii.py
cp .env.example .env                      # fill in API keys
```

A CUDA GPU (~16–24 GB) is strongly recommended for the embedding,
topic-modeling, and zero-shot stages; the LLM-judge stages need API keys and
cost a few USD in total. Run every script from the repository root — all paths
are relative to it. Scripts that call APIs expect the keys in the environment
(`set -a && source .env && set +a`).

## Data availability

- **Capstone corpus**: IRB-restricted student work; not distributed. All
  committed artifacts reference capstones by opaque `capstone_id` only.
  Files whose source columns contained document text or source filenames are
  committed as `*_public.csv` variants with those columns removed.
- **Journal corpus**: abstracts are not redistributed. `fetch_journals.py`
  re-retrieves them from PubMed (set `ENTREZ_EMAIL` in `.env`);
  `journals/manifest.json` records the retrieval date, queries, and counts of
  the corpus used in the paper.
- **Expert benchmark**: the adjudicated ratings for the 100 expert-rated
  capstone–journal pairs are in `phase4_expert_benchmark/gold_standard.csv`;
  `build_benchmark.py` regenerates the rating scaffold when the source corpora
  are present.

## Reproducing the pipeline

Stages in dependency order (scripts within a stage run in the listed order):

1. **Corpus construction** *(restricted corpus required)* —
   `extract_text.py` → `bhi2026/build_metadata.py` → `bhi2026/redact_pii.py`
   → `bhi2026/extract_sections.py`
2. **Journal retrieval** *(network)* — `bhi2026/fetch_journals.py`
3. **Method B** *(GPU)* — `run_amia_zeroshot.py`, `run_amia_zeroshot_journals.py`
   → `js_divergence.py`; robustness: `run_amia_deberta.py` →
   `compare_bart_vs_deberta.py`; plots: `plot_amia_distribution.py`,
   `plot_capstone_vs_journal.py`
4. **Method A** *(GPU)* — `method_a_bertopic_ot.py` → `method_a_nulls.py`;
   sweeps: `hdbscan_sweep.py`, `umap_hdbscan_sweep.py`; robustness:
   `method_a_pubmedbert_robustness.py`
5. **Tier-1 validation** *(API)* — `generate_paraphrases.py` →
   `paraphrase_recovery.py` → `paraphrase_method_c.py`
6. **Method C** *(API)* — `method_c_llm_judge.py` → `analyze_method_c.py`
7. **Method D** *(API)* — `method_d_ot_llm_bridge.py`; ablations:
   `method_d_ablations.py`, `method_d_pair_aggregate.py`; cross-LLM:
   `method_d_openai_robustness.py`
8. **Convergence** — `stage5_convergence.py`
9. **Benchmark analyses** — `phase4_expert_benchmark/`: `compute_kappa.py`,
   `compute_kappa_complete.py`, `agreement_panel.py`, `joint_agreement.py`,
   `build_gold_standard.py` → `tier2_calibration.py`
10. **Hardening** — `phase5_hardening/`: `compute_cis.py`,
    `outlier_summary.py`, `build_audit_sheet.py`, `score_audit.py`
11. **Figures** — `phase3_method_d/redraw_fig2.py`

Stages 3–11 are deterministic given the corpora and seeds recorded in the
scripts (RNG seed 42 throughout), except for LLM-judge calls, whose ratings
are committed under the corresponding `phase*/` directories so every
downstream analysis can be re-run without re-querying the APIs.

## For maintainers

Before committing, run `python3 tools/check_release_pii.py` from the repo
root. It screens every file git would publish against the person-name
surfaces recorded in the (local-only) redaction log and fails if any name
appears.
