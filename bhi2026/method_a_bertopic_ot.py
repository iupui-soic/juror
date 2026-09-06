"""Method A — BERTopic per corpus + Optimal Transport alignment.

Method A. The pipeline:

  Stage 2 — embeddings + BERTopic per corpus
  ---------------------------------------------
  - Embedder: PubMedBERT mean-pool via sentence-transformers
              (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext —
              also exposed via the sentence-transformers wrapper). If that
              wrapper is unavailable we fall back to BAAI/bge-large-en-v1.5
              for the immediate pilot.
  - BERTopic with the pre-registered hyperparameters:
      umap: n_neighbors=15, n_components=5, min_dist=0, cosine, seed=42
      hdbscan: min_cluster_size=10/5 for capstones, 20/10 for journals
      vectorizer: 1-3 grams, min_df=2, max_df=0.9
  - Persist per-document topic distributions (soft via approximate_distribution
    if HDBSCAN didn't already give them) and per-topic centroid embeddings.

  Stage 3 Method A — Optimal Transport over topic distributions
  ---------------------------------------------
  - Each topic = its centroid embedding (mean of member docs).
  - Topic-to-topic ground cost = 1 - cosine_similarity between centroids
    (cross corpora).
  - Empirical mass = topic_size / corpus_size, per corpus, with outliers (-1)
    dropped from topic-level analysis.
  - Solve Wasserstein-1 with POT (ot.emd2) and report:
      * W_1 (raw)
      * normalized alignment score = exp(-W_1 / scale) where scale is the
        mean cross-corpus ground cost (so scaling is corpus-agnostic).
      * Per-topic mapping (which journal topic does each capstone topic flow
        to, and at what mass).

Outputs to bhi2026/phase3_method_a/:
  - capstone_topics.csv, journal_topics.csv, capstone_docs.csv,
    journal_docs.csv
  - capstone_topic_centroids.npy, journal_topic_centroids.npy
  - ot_alignment_report.json
  - capstone_to_journal_flows.csv  (top-3 destinations per capstone topic)
  - capstone_to_journal_heatmap.png

This file is the *spec*; running takes ~10-15 min on the P6000. Launch it in
tmux after the DeBERTa robustness job has finished so it doesn't share the GPU
during inference.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap
import ot

META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
OUT_DIR = Path("bhi2026/phase3_method_a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Primary embedding model: BAAI/bge-large-en-v1.5 (a standard
# sentence-transformers model). The PubMedBERT swap is run separately as a
# robustness check in method_a_pubmedbert_robustness.py.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

CAPSTONE_HDBSCAN = {"min_cluster_size": 10, "min_samples": 5}
JOURNAL_HDBSCAN = {"min_cluster_size": 20, "min_samples": 10}
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_capstones() -> list[dict]:
    items = []
    with open(META_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = TXT_ROOT / (r["source_path"] + ".txt")
            text = txt.read_text(encoding="utf-8") if txt.exists() else ""
            items.append({
                "id": r["capstone_id"],
                "year": r["year"],
                "semester": r["semester"],
                "text": text,
                "source_path": r["source_path"],
            })
    return items


def load_journals() -> list[dict]:
    items = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
            items.append({
                "id": r["pmid"],
                "journal_key": r["journal_key"],
                "tier": r["tier"],
                "year": r["year"],
                "text": text,
            })
    return items


def embed_corpus(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    safe = [(t or "[empty]") for t in texts]
    return model.encode(safe, show_progress_bar=True, batch_size=32,
                        convert_to_numpy=True, normalize_embeddings=True)


def build_bertopic(embeddings: np.ndarray, texts: list[str], hdbscan_cfg: dict, name: str):
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                           metric="cosine", random_state=SEED, low_memory=True)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_cfg["min_cluster_size"],
                                    min_samples=hdbscan_cfg["min_samples"],
                                    metric="euclidean", cluster_selection_method="eom",
                                    prediction_data=True)
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9,
                                 stop_words="english")
    topic_model = BERTopic(
        embedding_model=None,            # we pass embeddings directly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,   # too slow; we get soft via approximate_distribution
        verbose=True,
    )
    log(f"Fitting BERTopic on {name} (n={len(texts)})...")
    topics, _ = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics


def topic_centroid(embeddings: np.ndarray, topics: list[int]) -> dict[int, np.ndarray]:
    topics_arr = np.array(topics)
    unique = sorted(set(topics))
    return {t: embeddings[topics_arr == t].mean(axis=0) for t in unique if t != -1}


def topic_size(topics: list[int], total: int) -> dict[int, float]:
    """Empirical mass: members / total non-outlier docs."""
    n_in_topics = sum(1 for t in topics if t != -1)
    counts: dict[int, int] = {}
    for t in topics:
        if t == -1:
            continue
        counts[t] = counts.get(t, 0) + 1
    return {t: c / n_in_topics for t, c in counts.items()}


def main() -> None:
    log(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    log("=== Loading data ===")
    caps = load_capstones()
    jrns = load_journals()
    log(f"  capstones: {len(caps)}  journals: {len(jrns)}")

    log("=== Embedding capstones ===")
    cap_emb = embed_corpus(embedder, [c["text"] for c in caps])
    np.save(OUT_DIR / "capstone_embeddings.npy", cap_emb)
    log(f"  capstone embeddings shape: {cap_emb.shape}")

    log("=== Embedding journals ===")
    jrn_emb = embed_corpus(embedder, [j["text"] for j in jrns])
    np.save(OUT_DIR / "journal_embeddings.npy", jrn_emb)
    log(f"  journal embeddings shape: {jrn_emb.shape}")

    log("=== BERTopic — capstones ===")
    cap_model, cap_topics = build_bertopic(cap_emb, [c["text"] for c in caps],
                                           CAPSTONE_HDBSCAN, "capstones")
    cap_topic_info = cap_model.get_topic_info()
    cap_topic_info.to_csv(OUT_DIR / "capstone_topics.csv", index=False)
    log(f"  capstone topics: {len(cap_topic_info) - 1} (excl outlier)  outliers: {cap_topic_info.iloc[0]['Count']}")

    log("=== BERTopic — journals ===")
    jrn_model, jrn_topics = build_bertopic(jrn_emb, [j["text"] for j in jrns],
                                           JOURNAL_HDBSCAN, "journals")
    jrn_topic_info = jrn_model.get_topic_info()
    jrn_topic_info.to_csv(OUT_DIR / "journal_topics.csv", index=False)
    log(f"  journal topics: {len(jrn_topic_info) - 1} (excl outlier)  outliers: {jrn_topic_info.iloc[0]['Count']}")

    # Per-doc topic CSVs
    with open(OUT_DIR / "capstone_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capstone_id", "year", "topic"])
        for it, t in zip(caps, cap_topics):
            w.writerow([it["id"], it["year"], t])
    with open(OUT_DIR / "journal_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "journal_key", "year", "topic"])
        for it, t in zip(jrns, jrn_topics):
            w.writerow([it["id"], it["journal_key"], it["year"], t])

    # Centroids
    log("=== Topic centroids ===")
    cap_cent = topic_centroid(cap_emb, cap_topics)
    jrn_cent = topic_centroid(jrn_emb, jrn_topics)
    cap_ids = sorted(cap_cent)
    jrn_ids = sorted(jrn_cent)
    C_cap = np.stack([cap_cent[i] for i in cap_ids])  # [n_cap_topics × dim]
    C_jrn = np.stack([jrn_cent[i] for i in jrn_ids])  # [n_jrn_topics × dim]
    np.save(OUT_DIR / "capstone_topic_centroids.npy", C_cap)
    np.save(OUT_DIR / "journal_topic_centroids.npy", C_jrn)

    # Empirical mass
    cap_mass = topic_size(cap_topics, total=len(caps))
    jrn_mass = topic_size(jrn_topics, total=len(jrns))
    a = np.array([cap_mass[i] for i in cap_ids])
    b = np.array([jrn_mass[i] for i in jrn_ids])

    # Cost matrix: 1 - cosine_similarity (since embeddings are normalized,
    # this is just 1 - C_cap @ C_jrn.T)
    sim = cosine_similarity(C_cap, C_jrn)
    cost = 1 - sim  # in [0, 2]

    # OT
    log("=== Optimal transport ===")
    transport = ot.emd(a, b, cost)
    w1 = float((transport * cost).sum())
    log(f"  W_1 = {w1:.4f}")
    scale = float(cost.mean())
    align_score = float(np.exp(-w1 / scale)) if scale > 0 else 0.0
    log(f"  normalized alignment score exp(-W_1/scale) = {align_score:.4f}")

    # Per-capstone-topic top-3 destinations
    flow_rows = []
    for i, ct in enumerate(cap_ids):
        row = transport[i]
        if row.sum() == 0:
            continue
        # Top destinations by mass
        top_idx = np.argsort(-row)[:3]
        for rank, j in enumerate(top_idx, start=1):
            if row[j] <= 0:
                continue
            flow_rows.append({
                "capstone_topic": ct,
                "rank": rank,
                "journal_topic": jrn_ids[j],
                "transported_mass": round(float(row[j]), 6),
                "cost": round(float(cost[i, j]), 4),
                "cosine_sim": round(float(sim[i, j]), 4),
            })
    with open(OUT_DIR / "capstone_to_journal_flows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flow_rows[0].keys()))
        w.writeheader()
        w.writerows(flow_rows)
    log(f"  Wrote {len(flow_rows)} flow rows")

    report = {
        "n_capstones": len(caps),
        "n_journals": len(jrns),
        "n_capstone_topics": len(cap_ids),
        "n_journal_topics": len(jrn_ids),
        "n_capstone_outliers": int(cap_topic_info.iloc[0]["Count"]),
        "n_journal_outliers": int(jrn_topic_info.iloc[0]["Count"]),
        "w1_distance": w1,
        "alignment_score_exp": align_score,
        "scale_used": scale,
        "embed_model": EMBED_MODEL,
    }
    (OUT_DIR / "ot_alignment_report.json").write_text(json.dumps(report, indent=2))
    log(f"Wrote OT report: {OUT_DIR / 'ot_alignment_report.json'}")

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(jrn_ids)), max(6, 0.4 * len(cap_ids))))
    im = ax.imshow(transport, aspect="auto", cmap="viridis")
    ax.set_xlabel("Journal topic")
    ax.set_ylabel("Capstone topic")
    ax.set_title(f"OT transport plan — capstone → journal (W₁ = {w1:.4f}, align exp(-W₁/scale) = {align_score:.4f})")
    fig.colorbar(im, ax=ax, label="Transported mass")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capstone_to_journal_heatmap.png", dpi=120)
    plt.close(fig)
    log(f"Wrote heatmap")


if __name__ == "__main__":
    sys.exit(main())
