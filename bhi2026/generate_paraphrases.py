"""Generate the 200-abstract × 3-paraphrase recovery set.

Sample 200 journal abstracts at random (with a fixed seed). For each, ask
Claude Sonnet 4.6 to generate three paraphrases under three style prompts:

  P1 — "literal rewrite": same meaning, different vocabulary
  P2 — "popular-science summary": same content, lay style
  P3 — "synonym substitution": substitute technical terms with synonyms while
       preserving every claim

Each paraphrase output is stored verbatim in JSON:

  {
    "pmid": "...",
    "journal_key": "...",
    "original_title": "...",
    "original_abstract": "...",
    "paraphrases": {
      "P1": "...",
      "P2": "...",
      "P3": "..."
    }
  }

Run with:
    set -a; source .env; set +a
    python3 bhi2026/generate_paraphrases.py --execute --n 200

Dry-run by default. Resume-supported: existing entries with all three
paraphrases populated are skipped.

Cost estimate at Sonnet 4.6 pricing:
  600 calls × ~600 in + 400 out = 360k in + 240k out
  ~ $1.08 input + $3.60 output  = ~ $4.70 total
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
OUT_DIR = Path("bhi2026/phase2_paraphrase")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "paraphrases.json"

RNG_SEED = 42

PROMPTS = {
    "P1": (
        "Rewrite the following biomedical abstract using different words and sentence structures. "
        "Preserve every factual claim, all quantitative results, and the overall meaning. "
        "Do not add new information, do not remove information, and do not editorialize. "
        "Match the formal academic register of the original. "
        "Return ONLY the rewritten abstract, no preamble.\n\nOriginal abstract:\n{abstract}"
    ),
    "P2": (
        "Rewrite the following biomedical abstract as a popular-science summary suitable for an educated lay reader. "
        "Preserve all factual claims and quantitative results, but use accessible language, shorter sentences, and concrete examples where helpful. "
        "Do not add new information or speculation. "
        "Return ONLY the rewritten summary, no preamble.\n\nOriginal abstract:\n{abstract}"
    ),
    "P3": (
        "Rewrite the following biomedical abstract by substituting technical and methodological terms with their accepted synonyms or definitional equivalents. "
        "Examples: \"machine learning\" might become \"data-driven predictive modeling\"; \"electronic health record\" might become \"electronic medical record\"; \"cohort study\" might become \"observational follow-up study\". "
        "Preserve every factual claim and the overall meaning. "
        "Return ONLY the rewritten abstract, no preamble.\n\nOriginal abstract:\n{abstract}"
    ),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_abstracts(n: int, seed: int) -> list[dict]:
    import random
    rnd = random.Random(seed)
    rows = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("abstract") and len(r["abstract"]) >= 200:
                rows.append({
                    "pmid": r["pmid"],
                    "journal_key": r["journal_key"],
                    "tier": r["tier"],
                    "year": r["year"],
                    "original_title": r.get("title", ""),
                    "original_abstract": r["abstract"],
                })
    rnd.shuffle(rows)
    return rows[:n]


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    sampled = sample_abstracts(args.n, RNG_SEED)
    log(f"Sampled {len(sampled)} abstracts")

    # Resume
    existing: dict[str, dict] = {}
    if OUT_JSON.exists():
        existing_list = json.loads(OUT_JSON.read_text())
        for item in existing_list:
            existing[item["pmid"]] = item
        log(f"Resume: {len(existing)} pmids already on disk")

    # Initialize: for any sampled pmid not in existing, add a skeleton
    for item in sampled:
        if item["pmid"] not in existing:
            existing[item["pmid"]] = {
                **item,
                "paraphrases": {},
                "errors": {},
            }

    if not args.execute:
        log("Dry run; not calling API.")
        log(f"Estimated cost at Sonnet 4.6 pricing (600 calls): ~$4.70")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(2)

    # Only process the current sample (so reducing --n on resume actually
    # reduces work, instead of re-iterating the full prior cache).
    sampled_pmids = [it["pmid"] for it in sampled]
    total_calls = sum(
        1 for pmid in sampled_pmids for k in PROMPTS
        if k not in existing[pmid].get("paraphrases", {})
    )
    log(f"API calls remaining: {total_calls}")

    n_calls = 0
    for pmid in sampled_pmids:
        item = existing[pmid]
        for key, template in PROMPTS.items():
            if key in item.get("paraphrases", {}):
                continue
            prompt = template.format(abstract=item["original_abstract"])
            try:
                text = call_claude(prompt, model=args.model)
                item.setdefault("paraphrases", {})[key] = text
                item.setdefault("errors", {}).pop(key, None)
            except Exception as e:
                item.setdefault("errors", {})[key] = f"{type(e).__name__}: {e}"
            n_calls += 1
            if n_calls % 20 == 0 or n_calls == total_calls:
                log(f"  {n_calls}/{total_calls} calls; saving checkpoint")
                # Order output stable by sampled order
                ordered = [existing[i["pmid"]] for i in sampled]
                OUT_JSON.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    # Final save
    ordered = [existing[i["pmid"]] for i in sampled]
    OUT_JSON.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {OUT_JSON}")

    # Summary
    n_full = sum(1 for item in ordered if len(item.get("paraphrases", {})) == 3)
    n_err = sum(1 for item in ordered for k in item.get("errors", {}))
    log(f"Abstracts with all 3 paraphrases: {n_full}/{len(ordered)}")
    log(f"Failed paraphrase calls: {n_err}")


if __name__ == "__main__":
    main()
