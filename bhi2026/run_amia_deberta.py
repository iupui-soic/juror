"""Method B1.2 robustness check — DeBERTa-v3-large-mnli zero-shot over BOTH
the capstone corpus (n=288) and the journal corpus (n=10,374).

Same AMIA taxonomy, same hypothesis template, same batching as the BART-MNLI
runs so we can compare distributions head-to-head. Stores results under
bhi2026/phase1_deberta/ to avoid clobbering the BART artifacts.

Why this model: DeBERTa-v3-large is the current SOTA for NLI on MNLI; if
it agrees with BART on the per-domain ranking and the JS divergence direction,
the headline finding is not an artifact of one classifier's biases. If it
disagrees materially, that itself is a reportable robustness limitation.

Run via tmux for SSH-independence:
    tmux new-session -d -s deberta \
        "CUDA_VISIBLE_DEVICES=0 python3 -u bhi2026/run_amia_deberta.py \
         2>&1 | tee bhi2026/phase1_deberta/run.log"
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
META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")

OUT_DIR = Path("bhi2026/phase1_deberta")
CAP_DIR = OUT_DIR / "capstones"
JRN_DIR = OUT_DIR / "journals"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAP_DIR.mkdir(parents=True, exist_ok=True)
JRN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
HYPOTHESIS_TEMPLATE = "This text is about {}."
MAX_CHARS_CAP = 4096       # posters, same as BART run
MAX_CHARS_JRN = 1500       # abstracts, same as BART journals
BATCH_SIZE = 8
CHECKPOINT_EVERY = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_taxonomy():
    data = json.loads(TAXONOMY.read_text())
    ids = [d["id"] for d in data["domains"]]
    labels = [d["label"] for d in data["domains"]]
    hyps = [f"{d['label']} — {d['description']}" for d in data["domains"]]
    return ids, labels, hyps


def classify_corpus(clf, items: list[dict], ids, hypotheses, max_chars, ckpt_path: Path, out_csv: Path, extra_fields: list[str]) -> None:
    """items: list of dicts each with at least {"id", "text", **extra_fields}"""
    # Resume
    rows: list[dict] = []
    n_done = 0
    if ckpt_path.exists():
        data = json.loads(ckpt_path.read_text())
        n_done = int(data.get("n_done", 0))
        rows = list(data.get("rows", []))
        log(f"Resuming at {n_done}/{len(items)}")

    t0 = time.time()
    batch_start = n_done
    while batch_start < len(items):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        texts = [(it["text"] or "[empty]")[:max_chars] for it in batch]
        results = clf(
            texts,
            candidate_labels=hypotheses,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=BATCH_SIZE,
        )
        if isinstance(results, dict):
            results = [results]
        for it, res in zip(batch, results):
            scores_by_hyp = dict(zip(res["labels"], res["scores"]))
            row = {"id": it["id"]}
            for ef in extra_fields:
                row[ef] = it.get(ef, "")
            vec = []
            for d_id, hyp in zip(ids, hypotheses):
                s = float(scores_by_hyp.get(hyp, 0.0))
                row[d_id] = round(s, 6)
                vec.append(s)
            top = int(np.argmax(vec))
            row["argmax_id"] = ids[top]
            rows.append(row)
        batch_start += len(batch)

        if batch_start % CHECKPOINT_EVERY == 0 or batch_start >= len(items):
            elapsed = time.time() - t0
            rate = (batch_start - n_done) / elapsed if elapsed > 0 else 0
            remaining = (len(items) - batch_start) / rate if rate > 0 else 0
            log(f"  {batch_start}/{len(items)}  ({batch_start/len(items):.1%})  "
                f"rate={rate:.2f} docs/s  eta={remaining/60:.1f} min")
            ckpt_path.write_text(json.dumps({"n_done": batch_start, "rows": rows}))

    # Final CSV
    fieldnames = ["id"] + extra_fields + ids + ["argmax_id"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {out_csv}")


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

    # --- Capstones ---
    log("=== Capstones ===")
    cap_items: list[dict] = []
    with open(META_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt_path = TXT_ROOT / (r["source_path"] + ".txt")
            text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
            cap_items.append({
                "id": r["capstone_id"],
                "source_path": r["source_path"],
                "year": r["year"],
                "semester": r["semester"],
                "text": text,
            })
    log(f"  {len(cap_items)} capstones loaded")
    classify_corpus(
        clf, cap_items, ids, hypotheses,
        max_chars=MAX_CHARS_CAP,
        ckpt_path=CAP_DIR / "checkpoint.json",
        out_csv=CAP_DIR / "per_document_scores.csv",
        extra_fields=["source_path", "year", "semester"],
    )

    # --- Journals ---
    log("=== Journals ===")
    jrn_items: list[dict] = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
            jrn_items.append({
                "id": r["pmid"],
                "journal_key": r["journal_key"],
                "tier": r["tier"],
                "year": r["year"],
                "text": text,
            })
    log(f"  {len(jrn_items)} journal articles loaded")
    classify_corpus(
        clf, jrn_items, ids, hypotheses,
        max_chars=MAX_CHARS_JRN,
        ckpt_path=JRN_DIR / "checkpoint.json",
        out_csv=JRN_DIR / "per_article_scores.csv",
        extra_fields=["journal_key", "tier", "year"],
    )

    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
