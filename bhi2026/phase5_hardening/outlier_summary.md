# Outlier (HDBSCAN noise) characterization — journal corpus

- Journal corpus: **10374 abstracts**; outliers (topic = -1): **3259 (31.4%)**.
- Retained topics: **106**, covering 7115 abstracts.

HDBSCAN labels low-density points as noise by design; high noise fractions on large, thematically broad abstract corpora are expected and documented in the BERTopic literature. The question is whether the noise hides a *coherent* theme that alignment would otherwise have surfaced. Three checks say it does not.

## 1. Outliers are spread proportionally across sources and years

| Journal | % of corpus | % of outliers | outlier rate within journal |
|---|---|---|---|
| ACI | 7.6% | 9.1% | 37.6% |
| IJMI | 14.5% | 17.8% | 38.7% |
| JAMIA | 14.5% | 14.9% | 32.3% |
| JBHI | 14.5% | 10.3% | 22.4% |
| JBI | 12.1% | 11.8% | 30.4% |
| JMIR | 14.5% | 14.1% | 30.5% |
| LancetDigitalHealth | 8.0% | 8.1% | 31.8% |
| npjDigitalMed | 14.5% | 14.0% | 30.5% |

Year distribution of outliers tracks the corpus (no period dropped): 2020: 14% vs 15%, 2021: 15% vs 16%, 2022: 15% vs 16%, 2023: 18% vs 17%, 2024: 17% vs 17%, 2025: 21% vs 20%.

## 2. Outliers are low-density tails of already-represented themes

Each outlier sub-cluster (KMeans, K=15) matched to its nearest *retained* journal topic by centroid cosine. High similarity = the sub-cluster is a diffuse version of a theme the model already kept.

| sub-cluster | size | nearest retained topic | cosine |
|---|---|---|---|
| 4 | 436 | 27_risk_diabetes_prediction_machine | 0.98 |
| 11 | 348 | 12_segmentation_image_tumor_images | 0.98 |
| 5 | 258 | 104_ehealth_usability_usability evaluation_cocreation | 0.97 |
| 1 | 252 | 18_nlp_notes_deidentification_language | 0.98 |
| 3 | 241 | 1_social_media_social media_online | 0.97 |
| 8 | 235 | 2_cds_alert_alerts_decision support | 0.97 |
| 10 | 224 | 9_ehr_burnout_documentation_burden | 0.97 |
| 13 | 220 | 14_diabetes_intervention_interventions_adherence | 0.98 |
| 2 | 215 | 8_ai_intelligence_artificial intelligence_artificial | 0.98 |
| 6 | 183 | 5_prediction_disease_temporal_ehr | 0.97 |
| 0 | 174 | 79_provenance_redcap_ctsa_cdes | 0.97 |
| 12 | 152 | 61_stress_wearable_heart rate_sensor | 0.96 |
| 14 | 145 | 62_artificial intelligence_intelligence_artificial_ai | 0.93 |
| 9 | 122 | 30_ad_alzheimers_alzheimers disease_disease | 0.96 |
| 7 | 54 | 54_informatics_amia_women_dei | 0.89 |

Median nearest-retained-topic cosine across sub-clusters = **0.97** (min 0.89, max 0.98). No large sub-cluster is far from a retained theme — i.e., the discard pile contains no coherent, well-separated research program that JUROR failed to see.

## 3. Representative outlier titles (idiosyncratic, not a missing theme)

**Sub-cluster 4** (n=436):
  - Risk prediction of delirium in hospitalized patients using machine learning: An implementation and prospective evaluation study
  - Machine learning for patient risk stratification: standing on, or looking over, the shoulders of clinicians?
  - Evaluation of crowdsourced mortality prediction models as a framework for assessing artificial intelligence in medicine

**Sub-cluster 11** (n=348):
  - Unassisted Clinicians Versus Deep Learning-Assisted Clinicians in Image-Based Cancer Diagnostics: Systematic Review With Meta-analysis
  - A scoping review on multimodal deep learning in biomedical images and texts
  - Response score of deep learning for out-of-distribution sample detection of medical images

**Sub-cluster 5** (n=258):
  - Digital Health Equity and Tailored Health Care Service for People With Disability: User-Centered Design and Usability Study
  - Mapping the Landscape of Digital Health Intervention Strategies: 25-Year Synthesis
  - Empowering Capabilities of People With Chronic Conditions Supported by Digital Health Technologies: Scoping Review

**Sub-cluster 1** (n=252):
  - Natural language inference for curation of structured clinical registries from unstructured text
  - From text to data: Open-source large language models in extracting cancer related medical attributes from German pathology reports
  - Biomedical text normalization through generative modeling

**Sub-cluster 3** (n=241):
  - Digital Health Interventions for Adult Patients With Cancer Evaluated in Randomized Controlled Trials: Scoping Review
  - Feasibility and Acceptability of a Remotely Delivered, Web-Based Behavioral Intervention for Men With Prostate Cancer: Four-Arm Randomized C
  - Electronic Health Interventions and Cervical Cancer Screening: Systematic Review and Meta-Analysis

**Sub-cluster 8** (n=235):
  - Effect of digital tools to promote hospital quality and safety on adverse events after discharge
  - Evaluation of electronic health record-integrated digital health tools to engage hospitalized patients in discharge preparation
  - Information displays for automated surveillance algorithms of in-hospital patient deterioration: a scoping review
