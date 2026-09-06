# Robustness analyses and baselines

Sensitivity checks, competing baselines and depth sweeps for the capstone-vs-literature
study. Scripts read only committed artifacts plus the two cached embedding files;
`common.py` reproduces the published OT solve exactly (W1 agrees with
`phase3_method_a/ot_alignment_report.json` to 2.3e-08, and the 24 top-3 flows come out in
the same order).

Run from the repo root, e.g. `python3 bhi2026/phase6_robustness/r3_outlier_handling.py`.

| Script | What it does |
|---|---|
| `common.py` | shared loaders + OT solve (verified against the paper) |
| `r2_year_matched.py` | outlier-rate-by-year test; <=2024 OT with the fit held fixed |
| `r2b_refit_2024.py` | full BERTopic re-fit on the 8,314 <=2024 abstracts, OT re-solved (GPU, ~40 s) |
| `r3_outlier_handling.py` | four outlier strategies (drop / nearest / nearest>=p10 / keep-as-topic), OT re-solved under each |
| `r4_btm_baseline.py` | Bidirectional Topic Matching (arXiv:2412.18376 SS2.2-3.1) + head-to-head vs JUROR |
| `r5_rate_new_flows.py` | rates every new flow with the unchanged JUROR prompt. `python3 ... r5_rate_new_flows.py [claude\|openai]` |
| `r6_synthesis.py` | bootstrap drho (JUROR - BTM); granularity test; per-scenario rating distributions |
| `r8_multiresolution.py` | journal topics merged to k in {8,16,32,64} (mass preserved), OT re-solved at each level |
| `r8b_rate_k8.py` | re-rates the matched-granularity (k=8) flows with the unchanged JUROR prompt |
| `r9_topk_depth.py` | rates every transport edge to rank 10 (71 flows); bounds what the top-3 cut hides |
| `r10_btm_coverage.py` | BTM coverage over the 71-edge set |
| `r11_positive_control.py` | journal corpus split into two random halves (seed 42), fit separately, OT solved between them, top-3 flows rated with the unchanged prompt; a `neutral` variant re-rates rank-1 flows with corpus labels replaced by "Corpus A/B" |
| `r12_juror_retest.py` | test-retest: the 24 main flows re-rated twice more with the same prompt and model; pairwise agreement and one-way ICC |
| `redact_surfaces.py` | scrubs PERSON/ORG surfaces out of generated text before release |

## Rater and outputs

All ratings were run with the primary rater, `claude-sonnet-4-6`
(`out/r5_new_flow_ratings_claude.csv`); the GPT-5.4-mini pass is kept alongside
(`..._openai.csv`) as the cross-LLM arm. Per-scenario rating distributions are in
`out/r6_synthesis_claude.json`.

Human validation of JUROR's own output. Two raters scored the 33-flow instrument built
by `../phase5_hardening/build_audit_sheet.py`, blind (Sheet 1 saved before Sheet 2 was
opened). Scoring:

```
cd ../phase5_hardening
python3 flow_audit_full.py <rater1>.xlsx <rater2>.xlsx
```

Writes `../phase5_hardening/out/flow_audit.json` (aggregates only). Expected output,
for checking a re-run:

| Key | Value |
|---|---|
| `paper_flows_24.juror_vs_adjudicated.spearman` | 0.720 |
| `all_flows_33.juror_vs_adjudicated.spearman` | 0.669 |
| `paper_flows_24.inter_rater.spearman` | 0.770 |
| `paper_flows_24.reliability.human_icc_1k` | 0.815 |
| `baselines_vs_human.btm_max.delta_rho_juror_minus_this` | 0.364 (Steiger p = 0.017) |
| `clustering_and_power.icc_within_capstone_topic` | 0.374 (n_eff 13.7 of 24) |
| `justification_quality.invented_overlap.both` | 2 of 33, on C5-J22 and C7-J22 |

Two properties of this dataset that affect how the script's output reads. Every rater
disagreement was one point, so the adjudication rule of
`../phase4_expert_benchmark/build_gold_standard.py` reduces to the lead rater, and the
script reports both raters separately. And the 24 flows sit in 8 capstone-topic clusters,
so the script reports a cluster bootstrap alongside the flow bootstrap and Steiger test.

The filled workbooks quote capstone text, are IRB-restricted and gitignored (`*.xlsx`);
only `out/flow_audit.json` aggregates are released.


## Release hygiene

Two things in this directory are generated text rather than derived numbers, and
both are screened before commit.

- **LLM justifications and BERTopic keyword strings** can reintroduce an
  organisation name that survived Stage-0 redaction in the source document.
  `redact_surfaces.py` replaces them with `[PERSON]`/`[ORG]`, reading its marker
  lists from the gitignored `local/person_markers.txt` and `local/org_markers.txt`
  so the committed code names no one. The rating scripts call it on write. Note
  that BERTopic lower-cases keywords, so a surname promoted by c-TF-IDF arrives
  in lower case — matching is case-insensitive and uses letter boundaries rather
  than `\b`, since topic Names join tokens with `_`.
- **`r2b_journal_topics_2024.csv`** carries BERTopic's `Representative_Docs`,
  i.e. PubMed abstract text, which this project does not redistribute. It is
  gitignored; the committed artifact is `r2b_journal_topics_2024_public.csv`
  with that column dropped, matching the convention already used for
  `phase3_method_a/journal_topics.csv`.

`tools/check_release_pii.py` remains the person-name guard. Note that its match
is case-sensitive, so a name that reaches a file in lower case (as BERTopic
keyword representations do) is not caught by it.

