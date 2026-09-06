"""Tier-1 paraphrase recovery validation.

For each of the 200 sampled journal abstracts and its 3 paraphrases:

  Method A check:
    Embed original + each paraphrase with bge-large-en-v1.5; compute cosine
    similarity. Mean across all (original, paraphrase) pairs is the recovery
    score. Threshold: ≥ 0.85.

  Method B check:
    Run BART-MNLI zero-shot AMIA over original + each paraphrase. Compute
    per-document argmax agreement rate (original vs each paraphrase). Plus
    JS divergence between original-corpus and paraphrase-corpus AMIA top-1
    distributions, with bootstrap CI. Threshold: argmax agreement ≥ 0.85
    AND JS distribution shift ≈ 0.

  Method C check (optional — uses Claude API):
    For each (original, paraphrase) pair, the LLM-judge should rate 4 with
    shared_methodology = yes. Threshold: mean rating / 4 ≥ 0.85.

This script implements A and B (no API cost). Method C is a separate
opt-in call (`--method-c` flag, paid).

Outputs to bhi2026/phase2_paraphrase/:
  - method_a_cosine.csv       — per-pair cosines + summary
  - method_b_amia_paraphrase.csv  — paraphrase AMIA top-1 + agreement with original
  - tier1_report.json         — pass/fail per method
  - tier1_report.md           — paper-ready writeup
"""

from __future__ import annotations

import argparse
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

PARA_JSON = Path("bhi2026/phase2_paraphrase/paraphrases.json")
OUT_DIR = Path("bhi2026/phase2_paraphrase")
TAXONOMY = Path("bhi2026/amia_taxonomy.json")
BART_JOURNAL_CSV = Path("bhi2026/phase1/journals/per_article_scores.csv")

DOMAINS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
PROMPT_KEYS = ["P1", "P2", "P3"]
EPS = 1e-12


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_paraphrases() -> list[dict]:
    items = json.loads(PARA_JSON.read_text())
    return [it for it in items if len(it.get("paraphrases", {})) == 3]


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + EPS); q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + EPS)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def method_a_cosine(items: list[dict]) -> dict:
    from sentence_transformers import SentenceTransformer
    log("Method A: loading bge-large-en-v1.5...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    originals = [(it.get("original_title", "") + ". " + it["original_abstract"]).strip() for it in items]
    paraphrases = {k: [it["paraphrases"][k] for it in items] for k in PROMPT_KEYS}

    log(f"Embedding {len(originals)} originals + {sum(len(v) for v in paraphrases.values())} paraphrases...")
    orig_emb = model.encode(originals, batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
    para_emb = {k: model.encode(v, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
                for k, v in paraphrases.items()}

    # Cosine sim per (original, paraphrase) pair = dot product since normalized
    rows = []
    for k in PROMPT_KEYS:
        cos = np.sum(orig_emb * para_emb[k], axis=1)
        for i, it in enumerate(items):
            rows.append({
                "pmid": it["pmid"],
                "prompt_key": k,
                "cosine": float(cos[i]),
            })

    out_csv = OUT_DIR / "method_a_cosine.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "prompt_key", "cosine"])
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {out_csv}")

    # Save embeddings for reuse
    np.save(OUT_DIR / "paraphrase_original_emb.npy", orig_emb)
    for k in PROMPT_KEYS:
        np.save(OUT_DIR / f"paraphrase_{k}_emb.npy", para_emb[k])

    summary = {}
    all_cos = np.array([r["cosine"] for r in rows])
    summary["mean_cosine_all"] = float(all_cos.mean())
    summary["median_cosine_all"] = float(np.median(all_cos))
    summary["min_cosine_all"] = float(all_cos.min())
    summary["std_cosine_all"] = float(all_cos.std())
    for k in PROMPT_KEYS:
        sub = np.array([r["cosine"] for r in rows if r["prompt_key"] == k])
        summary[f"mean_cosine_{k}"] = float(sub.mean())
    summary["n_pairs"] = len(rows)
    summary["passes_tier1"] = bool(summary["mean_cosine_all"] >= 0.85)
    return summary


def method_b_amia(items: list[dict]) -> dict:
    from transformers import pipeline

    log("Method B: loading BART-MNLI...")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                   device=device, torch_dtype=torch.float16 if device == 0 else None)

    tax = json.loads(TAXONOMY.read_text())
    ids = [d["id"] for d in tax["domains"]]
    labels = [d["label"] for d in tax["domains"]]
    hypotheses = [f"{d['label']} — {d['description']}" for d in tax["domains"]]

    # Load existing BART scores for the 200 originals to avoid reclassifying
    bart_map: dict[str, str] = {}
    with open(BART_JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bart_map[r["pmid"]] = r["argmax_id"]

    # Classify paraphrases (3 per item × N items)
    texts = []
    keys = []
    for it in items:
        for k in PROMPT_KEYS:
            texts.append(it["paraphrases"][k][:1500])
            keys.append((it["pmid"], k))

    log(f"Classifying {len(texts)} paraphrases (batched)...")
    BATCH = 16
    results: list[dict] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        safe = [t if t else "[empty]" for t in batch]
        out = clf(safe, candidate_labels=hypotheses,
                  hypothesis_template="This text is about {}.",
                  multi_label=True, batch_size=BATCH)
        if isinstance(out, dict):
            out = [out]
        for r in out:
            scores = dict(zip(r["labels"], r["scores"]))
            top = max(range(len(ids)), key=lambda j: scores[hypotheses[j]])
            results.append({"argmax_id": ids[top], "scores": [float(scores[h]) for h in hypotheses]})
        if (i // BATCH + 1) % 5 == 0:
            log(f"  {i + len(batch)}/{len(texts)}")

    # Per-pair agreement table
    rows = []
    n_agree = 0
    n_total = 0
    for (pmid, k), res in zip(keys, results):
        orig_argmax = bart_map.get(pmid, "?")
        agree = res["argmax_id"] == orig_argmax
        rows.append({
            "pmid": pmid,
            "prompt_key": k,
            "original_argmax": orig_argmax,
            "paraphrase_argmax": res["argmax_id"],
            "agree": agree,
        })
        n_total += 1
        n_agree += int(agree)

    out_csv = OUT_DIR / "method_b_amia_paraphrase.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "prompt_key", "original_argmax", "paraphrase_argmax", "agree"])
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {out_csv}")

    # Distribution comparison
    from collections import Counter
    orig_dist = Counter([bart_map.get(it["pmid"], "?") for it in items])
    para_dist = Counter([r["paraphrase_argmax"] for r in rows])
    p = np.array([orig_dist[d] / sum(orig_dist.values()) for d in ids])
    # Normalize paraphrases by 3× since each item contributes 3 paraphrases
    q = np.array([para_dist[d] / sum(para_dist.values()) for d in ids])
    js_para = js_div(p, q)

    # Per-prompt agreement
    per_prompt = {}
    for k in PROMPT_KEYS:
        sub = [r for r in rows if r["prompt_key"] == k]
        per_prompt[k] = sum(1 for r in sub if r["agree"]) / len(sub) if sub else 0

    summary = {
        "agreement_overall": n_agree / n_total if n_total else 0,
        "n_pairs": n_total,
        "agreement_per_prompt": per_prompt,
        "js_orig_vs_paraphrase_distribution": float(js_para),
        "passes_tier1_agreement": bool(n_agree / n_total >= 0.85) if n_total else False,
        "passes_tier1_js": bool(js_para <= 0.02),  # very small expected
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-c", action="store_true", help="Also run LLM-judge on (orig, paraphrase) pairs (paid).")
    args = parser.parse_args()

    items = load_paraphrases()
    log(f"Loaded {len(items)} fully-paraphrased items")
    if not items:
        log("No items with all 3 paraphrases yet. Wait for generation to complete.")
        sys.exit(1)

    report: dict = {"n_items": len(items)}

    log("\n=== Method A: cosine similarity ===")
    a = method_a_cosine(items)
    report["method_a"] = a
    log(f"  mean cosine = {a['mean_cosine_all']:.4f}  (P1 {a['mean_cosine_P1']:.4f}, P2 {a['mean_cosine_P2']:.4f}, P3 {a['mean_cosine_P3']:.4f})")
    log(f"  Tier 1 pass (≥0.85): {a['passes_tier1']}")

    log("\n=== Method B: AMIA argmax agreement ===")
    b = method_b_amia(items)
    report["method_b"] = b
    log(f"  overall agreement = {b['agreement_overall']:.4f}")
    log(f"  per-prompt: {b['agreement_per_prompt']}")
    log(f"  JS(orig vs paraphrase dist) = {b['js_orig_vs_paraphrase_distribution']:.4f}")
    log(f"  Tier 1 pass: agreement {b['passes_tier1_agreement']}, JS {b['passes_tier1_js']}")

    # Write report
    out_json = OUT_DIR / "tier1_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    log(f"Wrote {out_json}")

    md = [
        "# Tier 1 — Paraphrase recovery validation",
        "",
        f"**n = {len(items)} abstracts × 3 paraphrases = {len(items)*3} pairs**",
        f"**Paraphraser:** Claude Sonnet 4.6, 3 style prompts (P1 literal, P2 popular-science, P3 synonym).",
        f"**Tier-1 threshold:** ≥ 0.85 normalized alignment per method.",
        "",
        "## Method A — bge-large-en-v1.5 cosine similarity",
        "",
        f"- mean cosine (all pairs): **{a['mean_cosine_all']:.4f}**",
        f"  - P1 (literal rewrite): {a['mean_cosine_P1']:.4f}",
        f"  - P2 (popular-science): {a['mean_cosine_P2']:.4f}",
        f"  - P3 (synonym sub):     {a['mean_cosine_P3']:.4f}",
        f"- min: {a['min_cosine_all']:.4f}   median: {a['median_cosine_all']:.4f}   std: {a['std_cosine_all']:.4f}",
        f"- **Tier 1 pass: {a['passes_tier1']}**",
        "",
        "## Method B — BART-MNLI AMIA argmax agreement",
        "",
        f"- overall argmax agreement (orig vs paraphrase): **{b['agreement_overall']:.4f}**",
        f"  - P1: {b['agreement_per_prompt']['P1']:.4f}",
        f"  - P2: {b['agreement_per_prompt']['P2']:.4f}",
        f"  - P3: {b['agreement_per_prompt']['P3']:.4f}",
        f"- JS divergence (original vs paraphrase corpus distributions): {b['js_orig_vs_paraphrase_distribution']:.4f}",
        f"- **Tier 1 pass — argmax agreement: {b['passes_tier1_agreement']}**",
        f"- **Tier 1 pass — JS distribution shift: {b['passes_tier1_js']}**",
        "",
        "## Implication",
        "",
        "Methods that fall below Tier 1 are excluded from the main paper. Methods that pass are validated as not-broken; the cross-corpus result they produced earlier can stand.",
        "",
    ]
    (OUT_DIR / "tier1_report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Wrote tier1 report markdown")


if __name__ == "__main__":
    main()
