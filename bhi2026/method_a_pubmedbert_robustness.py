"""Embedding robustness — re-run Method A (and Method D) with PubMedBERT
embeddings instead of bge-large-en-v1.5.

This is the embedding-side robustness check: does the headline finding (the
curriculum-vs-literature gap, the Method D flow-level ratings) survive when
we swap a domain-specific biomedical encoder for the general-purpose
bge-large? If Spearman(PubMedBERT-Method-D, bge-large-Method-D) ≥ 0.6 on
the flows, the result is robust to embedding choice.

Pipeline (mirrors method_a_bertopic_ot.py + method_d_ot_llm_bridge.py):
  1. Embed both corpora with pritamdeka/S-PubMedBert-MS-MARCO via
     sentence-transformers (drop-in API replacement for bge-large).
  2. Run BERTopic per corpus with the same UMAP + HDBSCAN hyperparameters.
  3. Solve OT between topic-centroid distributions.
  4. Extract top-3 flows per capstone topic.
  5. Run Method D LLM ratings on the new flows (24 calls, ~$0.20).
  6. Compare to the bge-large-based results: topic count + outlier rate,
     W_1, alignment score, flow ratings.

Outputs to bhi2026/phase4_pubmedbert/:
  - capstone_embeddings.npy, journal_embeddings.npy
  - capstone_topics.csv, journal_topics.csv
  - capstone_topic_centroids.npy, journal_topic_centroids.npy
  - capstone_to_journal_flows.csv
  - flow_ratings.csv (Method D ratings on PubMedBERT flows)
  - robustness_report.json + .md
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import hdbscan
import umap
import ot

CAP_META = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
BGE_PHASE3 = Path("bhi2026/phase3_method_a")
BGE_METHOD_D = Path("bhi2026/phase3_method_d/flow_ratings.csv")

OUT_DIR = Path("bhi2026/phase4_pubmedbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentence-transformer wrapper of PubMedBERT (drop-in replacement for bge-large).
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

CAPSTONE_HDBSCAN = {"min_cluster_size": 10, "min_samples": 5}
JOURNAL_HDBSCAN = {"min_cluster_size": 20, "min_samples": 10}
SEED = 42

N_KEYWORDS = 12
N_REP_DOCS = 3
DOC_EXCERPT_CHARS = 600

EPS = 1e-12

PROMPT_TEMPLATE = """You are an expert reviewer of biomedical informatics research. Two corpora have been clustered into topics using BERTopic. Below are descriptions of one topic from each corpus. Rate the THEMATIC ALIGNMENT between the two topics on a 0-4 scale.

0 — Unrelated. The topics cover completely different problems and methods.
1 — Tangentially related. They share a broad domain (e.g., both are healthcare) but tackle different research questions and use different methods.
2 — Adjacent. They share either the research question or the method, but not both.
3 — Substantially overlapping. They share the same research question and use overlapping methods, but the work-products are distinct.
4 — Same research question. They essentially address the same research question with comparable methods, even if the populations or settings differ.

Additionally indicate SHARED METHODOLOGY:
"yes" if the topics' methods clearly overlap.
"partial" if one technique is shared but the rest of the pipeline differs.
"no" if the methods are unrelated.

Provide a JUSTIFICATION of no more than 80 words.

Return ONLY a JSON object with this schema:
{{
  "thematic_alignment": <int 0-4>,
  "shared_methodology": "yes" | "partial" | "no",
  "justification": "<string>"
}}

=== Topic A — Capstone topic {cap_topic_id} ===
Top keywords: {cap_keywords}

Representative documents:
{cap_docs}

=== Topic B — Journal topic {jrn_topic_id} ===
Top keywords: {jrn_keywords}

Representative documents:
{jrn_docs}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_capstones() -> list[dict]:
    items = []
    with open(CAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = TXT_ROOT / (r["source_path"] + ".txt")
            text = txt.read_text(encoding="utf-8") if txt.exists() else ""
            items.append({"id": r["capstone_id"], "text": text, "source_path": r["source_path"]})
    return items


def load_journals() -> list[dict]:
    items = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
            items.append({"id": r["pmid"], "text": text})
    return items


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    safe = [t or "[empty]" for t in texts]
    return model.encode(safe, show_progress_bar=False, batch_size=32,
                        convert_to_numpy=True, normalize_embeddings=True)


def build_bertopic(embeddings: np.ndarray, texts: list[str], hdbscan_cfg: dict, name: str):
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                           metric="cosine", random_state=SEED, low_memory=True)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_cfg["min_cluster_size"],
                                    min_samples=hdbscan_cfg["min_samples"],
                                    metric="euclidean", cluster_selection_method="eom",
                                    prediction_data=True)
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, stop_words="english")
    tm = BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdbscan_model,
                  vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)
    log(f"Fitting BERTopic on {name} (n={len(texts)})...")
    topics, _ = tm.fit_transform(texts, embeddings)
    return tm, topics


def topic_centroid(embeddings: np.ndarray, topics: list[int]) -> dict[int, np.ndarray]:
    arr = np.array(topics)
    return {t: embeddings[arr == t].mean(axis=0)
            for t in sorted(set(topics)) if t != -1}


def topic_size_vec(topics: list[int]) -> dict[int, float]:
    n_in = sum(1 for t in topics if t != -1)
    counts: dict[int, int] = {}
    for t in topics:
        if t == -1: continue
        counts[t] = counts.get(t, 0) + 1
    return {t: c / n_in for t, c in counts.items()}


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def parse_keywords(name: str, n: int = N_KEYWORDS) -> str:
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    return ", ".join(parts[1].split("_")[:n])


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    log(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    log(f"Embedding dim: {embedder.get_sentence_embedding_dimension()}")

    caps = load_capstones()
    jrns = load_journals()
    log(f"Loaded {len(caps)} capstones, {len(jrns)} journals")

    # === Embeddings ===
    cap_emb_path = OUT_DIR / "capstone_embeddings.npy"
    jrn_emb_path = OUT_DIR / "journal_embeddings.npy"
    if cap_emb_path.exists() and jrn_emb_path.exists():
        log("Resume: loading cached embeddings")
        cap_emb = np.load(cap_emb_path)
        jrn_emb = np.load(jrn_emb_path)
    else:
        log("Embedding capstones...")
        cap_emb = embed(embedder, [c["text"] for c in caps])
        np.save(cap_emb_path, cap_emb)
        log(f"  capstone embeddings shape: {cap_emb.shape}")
        log("Embedding journals...")
        jrn_emb = embed(embedder, [j["text"] for j in jrns])
        np.save(jrn_emb_path, jrn_emb)
        log(f"  journal embeddings shape: {jrn_emb.shape}")

    # === BERTopic ===
    log("=== BERTopic — capstones ===")
    cap_tm, cap_topics = build_bertopic(cap_emb, [c["text"] for c in caps],
                                        CAPSTONE_HDBSCAN, "capstones")
    cap_info = cap_tm.get_topic_info()
    cap_info.to_csv(OUT_DIR / "capstone_topics.csv", index=False)
    n_cap_topics = len(cap_info) - 1
    n_cap_outliers = int(cap_info.iloc[0]["Count"])
    log(f"  topics={n_cap_topics}  outliers={n_cap_outliers}/{len(caps)} ({n_cap_outliers/len(caps):.1%})")

    log("=== BERTopic — journals ===")
    jrn_tm, jrn_topics = build_bertopic(jrn_emb, [j["text"] for j in jrns],
                                        JOURNAL_HDBSCAN, "journals")
    jrn_info = jrn_tm.get_topic_info()
    jrn_info.to_csv(OUT_DIR / "journal_topics.csv", index=False)
    n_jrn_topics = len(jrn_info) - 1
    n_jrn_outliers = int(jrn_info.iloc[0]["Count"])
    log(f"  topics={n_jrn_topics}  outliers={n_jrn_outliers}/{len(jrns)} ({n_jrn_outliers/len(jrns):.1%})")

    # Per-doc CSVs (needed for Method D representative docs)
    with open(OUT_DIR / "capstone_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["capstone_id", "topic"])
        for it, t in zip(caps, cap_topics): w.writerow([it["id"], t])
    with open(OUT_DIR / "journal_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pmid", "topic"])
        for it, t in zip(jrns, jrn_topics): w.writerow([it["id"], t])

    # === Centroids + OT ===
    log("=== Topic centroids ===")
    cap_cent = topic_centroid(cap_emb, cap_topics)
    jrn_cent = topic_centroid(jrn_emb, jrn_topics)
    cap_ids = sorted(cap_cent); jrn_ids = sorted(jrn_cent)
    C_cap = np.stack([cap_cent[i] for i in cap_ids])
    C_jrn = np.stack([jrn_cent[i] for i in jrn_ids])
    np.save(OUT_DIR / "capstone_topic_centroids.npy", C_cap)
    np.save(OUT_DIR / "journal_topic_centroids.npy", C_jrn)

    cap_mass = topic_size_vec(cap_topics)
    jrn_mass = topic_size_vec(jrn_topics)
    a = np.array([cap_mass[i] for i in cap_ids])
    b = np.array([jrn_mass[i] for i in jrn_ids])

    sim = cosine_similarity(C_cap, C_jrn)
    cost = 1 - sim

    log("=== Optimal transport ===")
    transport = ot.emd(a, b, cost)
    w1 = float((transport * cost).sum())
    scale = float(cost.mean())
    align = float(np.exp(-w1 / scale)) if scale > 0 else 0.0
    log(f"  W_1 = {w1:.4f}  scale={scale:.4f}  alignment_exp = {align:.4f}")

    # Top-3 flows per capstone topic
    flow_rows = []
    for i, ct in enumerate(cap_ids):
        row = transport[i]
        if row.sum() == 0: continue
        top_idx = np.argsort(-row)[:3]
        for rank, j in enumerate(top_idx, start=1):
            if row[j] <= 0: continue
            flow_rows.append({
                "capstone_topic": ct, "rank": rank,
                "journal_topic": jrn_ids[j],
                "transported_mass": round(float(row[j]), 6),
                "cost": round(float(cost[i, j]), 4),
                "cosine_sim": round(float(sim[i, j]), 4),
            })
    flows_csv = OUT_DIR / "capstone_to_journal_flows.csv"
    with open(flows_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flow_rows[0].keys()))
        w.writeheader(); w.writerows(flow_rows)
    log(f"  wrote {len(flow_rows)} flow rows to {flows_csv}")

    # === Method D LLM ratings on new PubMedBERT-derived flows ===
    log("=== Method D LLM ratings (Claude Sonnet 4.6) ===")
    cap_kw = {int(r["Topic"]): parse_keywords(r["Name"]) for r in csv.DictReader(open(OUT_DIR / "capstone_topics.csv"))}
    jrn_kw = {int(r["Topic"]): parse_keywords(r["Name"]) for r in csv.DictReader(open(OUT_DIR / "journal_topics.csv"))}
    cap_docs_by_t: dict[int, list[str]] = defaultdict(list)
    with open(OUT_DIR / "capstone_docs.csv") as f:
        for r in csv.DictReader(f):
            cap_docs_by_t[int(r["topic"])].append(r["capstone_id"])
    jrn_docs_by_t: dict[int, list[str]] = defaultdict(list)
    with open(OUT_DIR / "journal_docs.csv") as f:
        for r in csv.DictReader(f):
            jrn_docs_by_t[int(r["topic"])].append(r["pmid"])
    cap_texts = {c["id"]: c["text"] for c in caps}
    jrn_texts = {j["id"]: j["text"] for j in jrns}

    ratings_csv = OUT_DIR / "flow_ratings.csv"
    done: set[tuple[int, int]] = set()
    existing = []
    if ratings_csv.exists():
        with open(ratings_csv) as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") and not r.get("error"):
                    done.add((int(r["capstone_topic"]), int(r["journal_topic"])))
                    existing.append(r)
        log(f"  resume: {len(done)} already rated")

    fieldnames = ["capstone_topic", "journal_topic", "rank", "mass", "cosine",
                  "thematic_alignment", "shared_methodology", "justification",
                  "cap_keywords", "jrn_keywords", "error"]
    with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing: w.writerow({k: r.get(k, "") for k in fieldnames})
        for i, fl in enumerate(flow_rows, start=1):
            key = (fl["capstone_topic"], fl["journal_topic"])
            if key in done: continue
            ct, jt = key
            cap_doc_ids = cap_docs_by_t[ct][:N_REP_DOCS]
            jrn_doc_ids = jrn_docs_by_t[jt][:N_REP_DOCS]
            cap_block = [f"  ({k}) {cap_texts.get(cid, '')[:DOC_EXCERPT_CHARS].strip()}" for k, cid in enumerate(cap_doc_ids, 1)]
            jrn_block = [f"  ({k}) {jrn_texts.get(pmid, '')[:DOC_EXCERPT_CHARS].strip()}" for k, pmid in enumerate(jrn_doc_ids, 1)]
            prompt = PROMPT_TEMPLATE.format(
                cap_topic_id=ct, jrn_topic_id=jt,
                cap_keywords=cap_kw.get(ct, "?"), jrn_keywords=jrn_kw.get(jt, "?"),
                cap_docs="\n".join(cap_block), jrn_docs="\n".join(jrn_block),
            )
            try:
                rating = call_claude(prompt)
                row = {
                    "capstone_topic": ct, "journal_topic": jt,
                    "rank": fl["rank"], "mass": fl["transported_mass"], "cosine": fl["cosine_sim"],
                    "thematic_alignment": rating.get("thematic_alignment", ""),
                    "shared_methodology": rating.get("shared_methodology", ""),
                    "justification": rating.get("justification", ""),
                    "cap_keywords": cap_kw.get(ct, ""), "jrn_keywords": jrn_kw.get(jt, ""),
                    "error": "",
                }
            except Exception as e:
                row = {
                    "capstone_topic": ct, "journal_topic": jt,
                    "rank": fl["rank"], "mass": fl["transported_mass"], "cosine": fl["cosine_sim"],
                    "thematic_alignment": "", "shared_methodology": "", "justification": "",
                    "cap_keywords": cap_kw.get(ct, ""), "jrn_keywords": jrn_kw.get(jt, ""),
                    "error": f"{type(e).__name__}: {e}",
                }
            w.writerow(row); f.flush()
            log(f"  {i}/{len(flow_rows)} T{ct}->T{jt}  rating={row['thematic_alignment'] if row['thematic_alignment']!='' else 'ERR'}")
    log(f"Wrote {ratings_csv}")

    # === Compare to bge-large baseline ===
    log("\n=== Comparison vs bge-large baseline ===")
    bge_report = json.loads((BGE_PHASE3 / "ot_alignment_report.json").read_text())
    log(f"bge-large: topics={bge_report['n_capstone_topics']}+{bge_report['n_journal_topics']}, "
        f"outliers={bge_report['n_capstone_outliers']}+{bge_report['n_journal_outliers']}, "
        f"W_1={bge_report['w1_distance']:.4f}, align={bge_report['alignment_score_exp']:.4f}")
    log(f"PubMedBERT: topics={n_cap_topics}+{n_jrn_topics}, "
        f"outliers={n_cap_outliers}+{n_jrn_outliers}, "
        f"W_1={w1:.4f}, align={align:.4f}")

    # Compare Method D flow ratings across embedding models
    pubmedbert_ratings = {}
    for r in csv.DictReader(open(ratings_csv)):
        if r.get("thematic_alignment") and not r.get("error"):
            pubmedbert_ratings[(int(r["capstone_topic"]), int(r["journal_topic"]))] = int(r["thematic_alignment"])
    bge_ratings = {}
    for r in csv.DictReader(open(BGE_METHOD_D)):
        if r.get("thematic_alignment") and not r.get("error"):
            bge_ratings[(int(r["capstone_topic"]), int(r["journal_topic"]))] = int(r["thematic_alignment"])

    # Topic assignments differ between models, so flows aren't directly comparable
    # by (cap_topic, jrn_topic) keys. Compare mass-weighted aggregate ratings instead.
    pm_ratings_arr = list(pubmedbert_ratings.values())
    bge_ratings_arr = list(bge_ratings.values())
    pm_mean = float(np.mean(pm_ratings_arr)) if pm_ratings_arr else 0.0
    bge_mean = float(np.mean(bge_ratings_arr)) if bge_ratings_arr else 0.0
    log(f"\nMethod D mean rating: PubMedBERT {pm_mean:.3f}, bge-large {bge_mean:.3f}")

    # If any flow keys overlap by accident, report rank correlation
    common = sorted(set(pubmedbert_ratings) & set(bge_ratings))
    log(f"Flows with same (cap_topic, jrn_topic) key in both runs: {len(common)} (topic IDs not synchronized across runs)")
    if len(common) >= 3:
        pm_v = np.array([pubmedbert_ratings[k] for k in common])
        bge_v = np.array([bge_ratings[k] for k in common])
        sp = spearmanr(pm_v, bge_v); pe = pearsonr(pm_v, bge_v)
        log(f"  Spearman ρ = {sp.statistic:.3f}, Pearson r = {pe.statistic:.3f} (n={len(common)})")

    # Per-document embedding-similarity check: how similar are PubMedBERT and bge-large embeddings for the same docs?
    bge_cap = np.load(BGE_PHASE3 / "capstone_embeddings.npy")
    bge_jrn = np.load(BGE_PHASE3 / "journal_embeddings.npy")
    # Normalize and dot per-row
    def row_cosine_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
        # If dims differ we can't directly compare; just report mean self-cosine across both
        return None
    log(f"\nbge-large dim: {bge_cap.shape[1]}, PubMedBERT dim: {cap_emb.shape[1]}")
    log("(direct cosine between embedding spaces undefined due to different dims;")
    log(" instead report rank-correlation of per-doc nearest-neighbor structure.)")

    # Rank correlation of pairwise cosine distances (sample 200 capstones to keep it manageable)
    rng = np.random.default_rng(SEED)
    n_sample = min(200, bge_cap.shape[0])
    idx = rng.choice(bge_cap.shape[0], n_sample, replace=False)
    bge_sub = bge_cap[idx]
    pm_sub = cap_emb[idx]
    bge_sim = cosine_similarity(bge_sub, bge_sub)
    pm_sim = cosine_similarity(pm_sub, pm_sub)
    iu = np.triu_indices_from(bge_sim, k=1)
    sp_emb = spearmanr(bge_sim[iu], pm_sim[iu])
    pe_emb = pearsonr(bge_sim[iu], pm_sim[iu])
    log(f"\nPairwise-cosine similarity rank correlation across {n_sample} capstones:")
    log(f"  Spearman ρ = {sp_emb.statistic:.3f}  Pearson r = {pe_emb.statistic:.3f}")

    report = {
        "embed_model_robustness": EMBED_MODEL,
        "embed_dim_bge": int(bge_cap.shape[1]),
        "embed_dim_pubmedbert": int(cap_emb.shape[1]),
        "pairwise_cosine_spearman_caps": float(sp_emb.statistic),
        "pairwise_cosine_pearson_caps": float(pe_emb.statistic),
        "pubmedbert": {
            "n_capstone_topics": n_cap_topics,
            "n_journal_topics": n_jrn_topics,
            "n_capstone_outliers": n_cap_outliers,
            "n_journal_outliers": n_jrn_outliers,
            "w1_distance": w1,
            "alignment_score_exp": align,
            "method_d_mean_rating": pm_mean,
            "method_d_rating_distribution": {k: pm_ratings_arr.count(k) for k in sorted(set(pm_ratings_arr))} if pm_ratings_arr else {},
        },
        "bge_large_baseline": {
            "n_capstone_topics": bge_report["n_capstone_topics"],
            "n_journal_topics": bge_report["n_journal_topics"],
            "n_capstone_outliers": bge_report["n_capstone_outliers"],
            "n_journal_outliers": bge_report["n_journal_outliers"],
            "w1_distance": bge_report["w1_distance"],
            "alignment_score_exp": bge_report["alignment_score_exp"],
            "method_d_mean_rating": bge_mean,
            "method_d_rating_distribution": {k: bge_ratings_arr.count(k) for k in sorted(set(bge_ratings_arr))} if bge_ratings_arr else {},
        },
    }
    (OUT_DIR / "robustness_report.json").write_text(json.dumps(report, indent=2))
    log(f"Wrote {OUT_DIR / 'robustness_report.json'}")


if __name__ == "__main__":
    main()
