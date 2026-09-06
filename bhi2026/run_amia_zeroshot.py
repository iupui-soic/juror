"""Phase 1 pilot: zero-shot classification of the 288 capstones into the 10
AMIA foundational domains.

Method B1 (NLI-based): facebook/bart-large-mnli with the AMIA domain
descriptions as candidate hypotheses. Each capstone is scored against all 10
domains; we treat the per-document distribution as a soft assignment (sigmoid
multilabel) and aggregate to a per-corpus distribution.

Outputs (in bhi2026/phase1/):
- per_document_scores.csv  — capstone_id, D1..D10 sigmoid scores, argmax label
- corpus_distribution.csv  — domain, mean_score, top1_share, weighted_share, top3_share
- distribution_plot.png    — bar chart of per-domain mean scores

Notes on text input:
- Use full text from texts/ (not redacted) since the NLI model is content-driven
  and redaction artifacts add noise. Privacy-compliant redacted copies live in
  texts_redacted/ for downstream public release.
- Documents are truncated to 1024 tokens (well within bart-large-mnli's 1024
  position limit). For posters this is the entire document; for the few long
  Report PDFs this captures the abstract + introduction + early methods.
"""

from __future__ import annotations

import os

# Skip transformers' TF integration (we have a broken tensorflow/protobuf combo
# in this environment) and stick to pytorch.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
TAXONOMY = Path("bhi2026/amia_taxonomy.json")
OUT_DIR = Path("bhi2026/phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This text is about {}."
MAX_CHARS = 4096  # ~1024 tokens, fits bart-large-mnli context.

def load_taxonomy() -> tuple[list[str], list[str], list[str]]:
    """Return (ids, short_labels, hypothesis_descriptions)."""
    data = json.loads(TAXONOMY.read_text())
    ids = [d["id"] for d in data["domains"]]
    labels = [d["label"] for d in data["domains"]]
    # The NLI hypothesis: use the longer description as the candidate label so
    # the model has more signal than just the short title.
    hypotheses = [f"{d['label']} — {d['description']}" for d in data["domains"]]
    return ids, labels, hypotheses


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    return text[:max_chars]


def main() -> None:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
    print(f"Loading {MODEL_NAME} ...")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )

    ids, labels, hypotheses = load_taxonomy()
    print(f"AMIA domains: {len(labels)}")
    for d_id, lab in zip(ids, labels):
        print(f"  {d_id}: {lab}")

    with open(META_CSV, encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    print(f"\nDocuments to classify: {len(meta_rows)}")

    rows: list[dict] = []
    for i, m in enumerate(meta_rows, start=1):
        cap_id = m["capstone_id"]
        rel = m["source_path"]
        txt_path = TXT_ROOT / (rel + ".txt")
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        text = truncate(text)

        if not text.strip():
            row = {"capstone_id": cap_id, "source_path": rel, "argmax_id": "NA", "argmax_label": "NA"}
            for d_id in ids:
                row[d_id] = 0.0
            rows.append(row)
            continue

        result = clf(
            text,
            candidate_labels=hypotheses,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )
        # result["labels"] is sorted by score desc; map back to our canonical ids.
        scores_by_hypothesis = dict(zip(result["labels"], result["scores"]))
        row = {"capstone_id": cap_id, "source_path": rel}
        score_vec = []
        for d_id, hyp in zip(ids, hypotheses):
            s = scores_by_hypothesis[hyp]
            row[d_id] = round(float(s), 6)
            score_vec.append(s)
        top_idx = int(np.argmax(score_vec))
        row["argmax_id"] = ids[top_idx]
        row["argmax_label"] = labels[top_idx]
        rows.append(row)

        if i % 10 == 0 or i == len(meta_rows):
            print(f"  {i}/{len(meta_rows)}  {cap_id}  top={row['argmax_id']} ({row[row['argmax_id']]:.2f})")

    # Write per-document scores
    fieldnames = ["capstone_id", "source_path", *ids, "argmax_id", "argmax_label"]
    out_doc = OUT_DIR / "per_document_scores.csv"
    with open(out_doc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-document scores: {out_doc}")

    # Aggregate corpus distribution
    score_matrix = np.array([[r[d_id] for d_id in ids] for r in rows if r["argmax_id"] != "NA"])
    mean_scores = score_matrix.mean(axis=0)
    weighted = score_matrix.sum(axis=0) / score_matrix.sum()

    argmax_labels = [r["argmax_id"] for r in rows if r["argmax_id"] != "NA"]
    n_valid = len(argmax_labels)
    top1_share = {d_id: argmax_labels.count(d_id) / n_valid for d_id in ids}

    # Top-3 share: each document contributes 1/3 to each of its top 3 domains
    top3_share = {d_id: 0.0 for d_id in ids}
    for r in rows:
        if r["argmax_id"] == "NA":
            continue
        per_doc = sorted([(d_id, r[d_id]) for d_id in ids], key=lambda x: -x[1])[:3]
        for d_id, _ in per_doc:
            top3_share[d_id] += 1 / 3 / n_valid

    dist_rows = []
    for d_id, label, ms, ws in zip(ids, labels, mean_scores, weighted):
        dist_rows.append({
            "domain_id": d_id,
            "domain_label": label,
            "mean_sigmoid_score": round(float(ms), 4),
            "weighted_share": round(float(ws), 4),
            "top1_share": round(top1_share[d_id], 4),
            "top3_share": round(top3_share[d_id], 4),
        })

    out_dist = OUT_DIR / "corpus_distribution.csv"
    with open(out_dist, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)
    print(f"Wrote corpus distribution: {out_dist}\n")

    # Print summary
    print("=== Per-AMIA-domain capstone distribution ===")
    print(f"{'D':<4} {'Label':<60} {'mean':>6} {'top1':>6} {'top3':>6}")
    for r in sorted(dist_rows, key=lambda x: -x["top1_share"]):
        print(f"{r['domain_id']:<4} {r['domain_label'][:60]:<60} {r['mean_sigmoid_score']:>6.3f} {r['top1_share']:>6.2%} {r['top3_share']:>6.2%}")


if __name__ == "__main__":
    main()
