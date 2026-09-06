"""AMIA zero-shot classification of the journal corpus (10,374 abstracts).

Same Method B1 (NLI) as the capstone pilot, optimized for journal-scale:
- Documents are pushed through the pipeline in batches (batch_size=16) so the
  GPU stays saturated.
- max_chars=1500 since abstracts are short (median ~250 words). Long abstracts
  are truncated at the model's tokenizer.
- Periodic checkpoint to disk every CHECKPOINT_EVERY documents so an
  interruption (tmux detach + lost SSH + node reboot) doesn't lose progress.
  Restart resumes from the checkpoint.

Output (bhi2026/phase1/journals/):
- per_article_scores.csv  — pmid, journal_key, year, D1..D10 scores, argmax
- corpus_distribution.csv — per-domain aggregate, plus per-tier and per-year
- checkpoint.json         — { "n_done": int, "rows": [...] }, for resume
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

import numpy as np
import torch
from transformers import pipeline

TAXONOMY = Path("bhi2026/amia_taxonomy.json")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
OUT_DIR = Path("bhi2026/phase1/journals")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

MODEL_NAME = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This text is about {}."
MAX_CHARS = 1500
BATCH_SIZE = 16
CHECKPOINT_EVERY = 200  # save every N documents


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_taxonomy():
    data = json.loads(TAXONOMY.read_text())
    ids = [d["id"] for d in data["domains"]]
    labels = [d["label"] for d in data["domains"]]
    hypotheses = [f"{d['label']} — {d['description']}" for d in data["domains"]]
    return ids, labels, hypotheses


def load_articles() -> list[dict]:
    rows = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = (r.get("title", "") + ". " + r.get("abstract", "")).strip()
            rows.append({
                "pmid": r["pmid"],
                "journal_key": r["journal_key"],
                "tier": r["tier"],
                "year": r["year"],
                "text": text[:MAX_CHARS] if text else "",
            })
    return rows


def load_checkpoint() -> tuple[int, list[dict]]:
    if not CHECKPOINT.exists():
        return 0, []
    data = json.loads(CHECKPOINT.read_text())
    return int(data.get("n_done", 0)), list(data.get("rows", []))


def save_checkpoint(n_done: int, rows: list[dict]) -> None:
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"n_done": n_done, "rows": rows}, ensure_ascii=False))
    tmp.replace(CHECKPOINT)


def main() -> None:
    device = 0 if torch.cuda.is_available() else -1
    log(f"Device: {'cuda' if device == 0 else 'cpu'}")
    log(f"Loading {MODEL_NAME} ...")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if device == 0 else None,
    )

    ids, labels, hypotheses = load_taxonomy()
    articles = load_articles()
    log(f"Total articles to classify: {len(articles)}")

    n_done, rows = load_checkpoint()
    if n_done:
        log(f"Resuming from checkpoint at doc {n_done}/{len(articles)} ({len(rows)} rows cached)")

    t0 = time.time()
    batch_start = n_done
    while batch_start < len(articles):
        batch = articles[batch_start : batch_start + BATCH_SIZE]
        texts = [a["text"] for a in batch]
        # Some abstracts may be empty (rare); supply a placeholder so the
        # pipeline returns an entry for every input.
        safe_texts = [t if t else "[empty]" for t in texts]

        results = clf(
            safe_texts,
            candidate_labels=hypotheses,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=BATCH_SIZE,
        )
        # results is a list aligned with input order.
        if isinstance(results, dict):  # single-element edge case
            results = [results]
        for art, res in zip(batch, results):
            scores_by_hyp = dict(zip(res["labels"], res["scores"]))
            row = {
                "pmid": art["pmid"],
                "journal_key": art["journal_key"],
                "tier": art["tier"],
                "year": art["year"],
            }
            score_vec = []
            for d_id, hyp in zip(ids, hypotheses):
                s = float(scores_by_hyp.get(hyp, 0.0))
                row[d_id] = round(s, 6)
                score_vec.append(s)
            top = int(np.argmax(score_vec))
            row["argmax_id"] = ids[top]
            row["argmax_label"] = labels[top]
            rows.append(row)

        batch_start += len(batch)

        # Progress + checkpoint
        if batch_start % CHECKPOINT_EVERY == 0 or batch_start >= len(articles):
            elapsed = time.time() - t0
            rate = (batch_start - n_done) / elapsed if elapsed > 0 else 0
            remaining = (len(articles) - batch_start) / rate if rate > 0 else 0
            log(
                f"  {batch_start}/{len(articles)}  "
                f"({batch_start/len(articles):.1%})  "
                f"rate={rate:.2f} docs/s  "
                f"eta={remaining/60:.1f} min"
            )
            save_checkpoint(batch_start, rows)

    # Write final per-article CSV
    fieldnames = ["pmid", "journal_key", "tier", "year", *ids, "argmax_id", "argmax_label"]
    out_csv = OUT_DIR / "per_article_scores.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote per-article scores: {out_csv}")

    # Aggregate distributions
    valid = [r for r in rows if r.get("argmax_id")]
    score_matrix = np.array([[r[d_id] for d_id in ids] for r in valid])
    mean_scores = score_matrix.mean(axis=0)
    weighted = score_matrix.sum(axis=0) / score_matrix.sum()
    argmax_labels = [r["argmax_id"] for r in valid]
    n = len(valid)
    top1_share = {d_id: argmax_labels.count(d_id) / n for d_id in ids}

    # Per-tier
    by_tier: dict[str, dict[str, float]] = {}
    for tier in ("core", "digital_health"):
        sub = [r for r in valid if r["tier"] == tier]
        if not sub:
            continue
        labs = [r["argmax_id"] for r in sub]
        by_tier[tier] = {d_id: labs.count(d_id) / len(sub) for d_id in ids}

    # Per-year (across all journals)
    years = sorted({r["year"] for r in valid if r["year"]})
    by_year: dict[str, dict[str, float]] = {}
    for yr in years:
        sub = [r for r in valid if r["year"] == yr]
        if not sub:
            continue
        labs = [r["argmax_id"] for r in sub]
        by_year[yr] = {d_id: labs.count(d_id) / len(sub) for d_id in ids}

    # Write corpus distribution CSV
    out_dist = OUT_DIR / "corpus_distribution.csv"
    with open(out_dist, "w", newline="", encoding="utf-8") as f:
        cols = ["domain_id", "domain_label", "mean_sigmoid", "weighted_share", "top1_share_all"]
        cols += [f"top1_share_{t}" for t in by_tier]
        cols += [f"top1_share_{y}" for y in years]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, d_id in enumerate(ids):
            row = {
                "domain_id": d_id,
                "domain_label": labels[i],
                "mean_sigmoid": round(float(mean_scores[i]), 4),
                "weighted_share": round(float(weighted[i]), 4),
                "top1_share_all": round(top1_share[d_id], 4),
            }
            for tier, share in by_tier.items():
                row[f"top1_share_{tier}"] = round(share[d_id], 4)
            for yr in years:
                row[f"top1_share_{yr}"] = round(by_year[yr][d_id], 4)
            w.writerow(row)
    log(f"Wrote corpus distribution: {out_dist}")

    # Headline print
    log("\n=== Journal corpus per-AMIA-domain distribution ===")
    log(f"{'D':<4} {'Label':<55} {'mean':>6} {'top1':>7}")
    for i, d_id in enumerate(ids):
        log(f"{d_id:<4} {labels[i][:55]:<55} {mean_scores[i]:>6.3f} {top1_share[d_id]:>7.2%}")

    elapsed = time.time() - t0
    log(f"\nTotal elapsed (this run): {elapsed/60:.1f} min")
    log(f"Total documents classified: {len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
