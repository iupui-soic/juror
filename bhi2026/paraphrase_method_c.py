"""Method C Tier-1 check — LLM-judge rates (original, paraphrase) pairs.

For each of 60 paraphrased journal abstracts × 3 paraphrase prompts = 180
pairs, ask Claude Sonnet 4.6 to rate the thematic alignment 0-4 using the
exact same rubric as Method C. Pre-specified retention criterion: the mean
rating must be ≥ 3.4 (= 0.85 × 4) for Method C to be retained.

Cost: ~$0.60. Resume-supported.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

PARA_JSON = Path("bhi2026/phase2_paraphrase/paraphrases.json")
OUT_DIR = Path("bhi2026/phase2_paraphrase")
OUT_CSV = OUT_DIR / "method_c_paraphrase_ratings.csv"

PROMPT_TEMPLATE = """You are an expert reviewer of biomedical informatics research. Rate the THEMATIC ALIGNMENT between the following two documents on a 0-4 scale.

0 — Unrelated. The documents cover completely different problems and methods.
1 — Tangentially related. They share a broad domain (e.g., both are healthcare) but tackle different research questions and use different methods.
2 — Adjacent. They share either the research question or the method, but not both.
3 — Substantially overlapping. They share the same research question and use overlapping methods, but the contributions are distinct.
4 — Same research question. They essentially address the same research question with comparable methods, even if the populations or settings differ.

Additionally indicate SHARED METHODOLOGY:
"yes" if the documents use overlapping methods (e.g., both use BERTopic on patient notes).
"partial" if one technique is shared but the rest of the pipeline differs.
"no" if the methods are unrelated.

Provide a JUSTIFICATION of no more than 60 words.

Return ONLY a JSON object with this schema:
{{
  "thematic_alignment": <int 0-4>,
  "shared_methodology": "yes" | "partial" | "no",
  "justification": "<string>"
}}

=== Document A ===
{doc_a}

=== Document B ===
{doc_b}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def build_prompt(original: str, paraphrase: str) -> str:
    return PROMPT_TEMPLATE.format(doc_a=original[:3500], doc_b=paraphrase[:3500])


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(2)

    items = json.loads(PARA_JSON.read_text())
    items = [it for it in items if len(it.get("paraphrases", {})) == 3]
    log(f"Loaded {len(items)} fully-paraphrased items")

    # Build pair list
    pairs = []
    for it in items:
        orig = (it.get("original_title", "") + ". " + it["original_abstract"]).strip()
        for k in ["P1", "P2", "P3"]:
            pairs.append({
                "pmid": it["pmid"],
                "prompt_key": k,
                "doc_a": orig,
                "doc_b": it["paraphrases"][k],
            })
    log(f"Total pairs to rate: {len(pairs)}")

    # Resume
    done: set[tuple[str, str]] = set()
    existing_rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") not in ("", None) and not r.get("error"):
                    done.add((r["pmid"], r["prompt_key"]))
                    existing_rows.append(r)
        log(f"Resume: {len(done)} pairs already rated")

    fieldnames = ["pmid", "prompt_key", "thematic_alignment", "shared_methodology", "justification", "error"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        f.flush()
        n_done = len(done)
        for i, p in enumerate(pairs, start=1):
            if (p["pmid"], p["prompt_key"]) in done:
                continue
            prompt = build_prompt(p["doc_a"], p["doc_b"])
            try:
                rating = call_claude(prompt)
                row = {
                    "pmid": p["pmid"], "prompt_key": p["prompt_key"],
                    "thematic_alignment": rating.get("thematic_alignment", ""),
                    "shared_methodology": rating.get("shared_methodology", ""),
                    "justification": rating.get("justification", ""),
                    "error": "",
                }
            except Exception as e:
                row = {
                    "pmid": p["pmid"], "prompt_key": p["prompt_key"],
                    "thematic_alignment": "", "shared_methodology": "",
                    "justification": "", "error": f"{type(e).__name__}: {e}",
                }
            writer.writerow(row)
            f.flush()
            n_done += 1
            if n_done % 20 == 0 or i == len(pairs):
                log(f"  {n_done}/{len(pairs)} pairs rated")

    log(f"Wrote {OUT_CSV}")

    # Summary
    rows = list(csv.DictReader(open(OUT_CSV)))
    ratings = [int(r["thematic_alignment"]) for r in rows if not r["error"] and r["thematic_alignment"]]
    n_err = sum(1 for r in rows if r["error"])
    if not ratings:
        log("No successful ratings")
        return
    mean = sum(ratings) / len(ratings)
    normalized = mean / 4.0
    from collections import Counter
    dist = Counter(ratings)
    log(f"Successful: {len(ratings)}  errors: {n_err}")
    log(f"Rating distribution: {dict(sorted(dist.items()))}")
    log(f"Mean rating: {mean:.3f} / 4")
    log(f"Normalized (mean/4): {normalized:.3f}")
    log(f"Tier 1 pass (≥0.85): {normalized >= 0.85}")

    # Per-prompt breakdown
    log("\nPer-prompt:")
    for k in ["P1", "P2", "P3"]:
        sub = [int(r["thematic_alignment"]) for r in rows
               if not r["error"] and r["prompt_key"] == k and r["thematic_alignment"]]
        if sub:
            m = sum(sub) / len(sub)
            log(f"  {k}: mean = {m:.3f}, n = {len(sub)}, dist = {dict(sorted(Counter(sub).items()))}")


if __name__ == "__main__":
    main()
