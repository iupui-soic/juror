"""Method C — LLM-as-judge pairwise alignment rating.

Method C. Per the pre-registered SAP, this is the third independent view of
cross-corpus alignment (alongside Method A optimal-transport and Method B
zero-shot taxonomic classification).

Pipeline:
  1. Pair sampling. Sample 500 (capstone, journal abstract) pairs
     stratified across the cosine-similarity range and across capstone topics.
     Cosine similarity uses the bge-large embeddings computed in Method A.
  2. LLM rating. For each pair, ask the LLM to return a structured rating:
       - thematic_alignment: int 0-4 (0 = unrelated, 4 = same research question)
       - shared_methodology: one of {"yes", "no", "partial"}
       - justification: ≤ 60 words
  3. Aggregate:
       - mean alignment, distribution of ratings
       - Pearson with cosine similarity (sanity check: should be > 0.4)
       - When the 100-pair expert benchmark is available, validate calibration
         (pre-registered target: Cohen's κ ≥ 0.5 vs human-adjudicated).

Important: this script does NOT auto-run. It needs:
  - ANTHROPIC_API_KEY set in the environment
  - explicit invocation: python3 bhi2026/method_c_llm_judge.py --execute
  - Method A artifacts: bhi2026/phase3_method_a/{capstone_embeddings,journal_embeddings}.npy

Cost estimate (Claude Sonnet 4.6):
  500 pairs × (~400 input tokens + 150 output tokens) = 200k in + 75k out
  At Sonnet 4.6 pricing ($3 / M input, $15 / M output):
    ~ $0.60 input + $1.13 output = ~ $1.73 total

If you want to test the pipeline first, pass --execute --n-pairs 10 to make
just 10 API calls (~$0.04).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PHASE3 = Path("bhi2026/phase3_method_a")
META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
OUT_DIR = Path("bhi2026/phase3_method_c")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_N_PAIRS = 500
RNG_SEED = 42

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

=== Document A (capstone poster excerpt) ===
{capstone_text}

=== Document B (journal abstract) ===
{journal_text}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_capstone_texts() -> dict[str, str]:
    out = {}
    with open(META_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = TXT_ROOT / (r["source_path"] + ".txt")
            out[r["capstone_id"]] = txt.read_text(encoding="utf-8") if txt.exists() else ""
    return out


def load_journal_texts() -> dict[str, dict]:
    out = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["pmid"]] = {
                "journal_key": r["journal_key"],
                "year": r["year"],
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
            }
    return out


def sample_pairs(cap_emb: np.ndarray, cap_ids: list[str],
                 jrn_emb: np.ndarray, jrn_ids: list[str],
                 n_pairs: int, seed: int,
                 method_a_dir: Path | None = None) -> list[dict]:
    """Stratified sample by cosine similarity quartile (Q1..Q4) + optional
    Method-A top-flow seeded pairs.

    Strategy:
      - 80% of n_pairs: cosine-quartile stratification (20% per quartile).
      - 20% of n_pairs: seeded from Method A capstone→journal top-flow
        destinations (8 capstone topics × top-3 journal targets each = 24
        capstone-topic × journal-topic combinations; sample n_seed/24 capstone
        docs × journal-doc pairs per combo). This anchors the high-alignment
        end so the LLM-judge sees plausibly aligned pairs and learns to use the
        upper part of the 0-4 scale.
    """
    rng = np.random.default_rng(seed)

    sim = cap_emb @ jrn_emb.T  # [n_cap × n_jrn], normalized embeds → cosine
    n_cap, n_jrn = sim.shape

    # ---- Quartile-stratified (80% of n_pairs) ----
    n_quartile_total = int(n_pairs * 0.8)
    per_bucket = n_quartile_total // 4

    flat = sim.reshape(-1)
    qs = np.quantile(flat, [0.25, 0.5, 0.75])

    used: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    for bucket, (lo, hi) in enumerate([(-1.0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1.01)]):
        mask = (flat >= lo) & (flat < hi)
        idx = np.flatnonzero(mask)
        chosen = rng.choice(idx, size=min(per_bucket, len(idx)), replace=False)
        for c in chosen:
            i, j = divmod(int(c), n_jrn)
            pairs.append({
                "bucket": f"Q{bucket + 1}",
                "capstone_id": cap_ids[i],
                "pmid": jrn_ids[j],
                "cosine_sim": float(sim[i, j]),
            })
            used.add((i, j))

    # ---- Method A top-flow seeded (20% of n_pairs) ----
    n_seed_target = n_pairs - len(pairs)
    if method_a_dir is None or n_seed_target <= 0:
        return pairs

    cap_docs_path = method_a_dir / "capstone_docs.csv"
    jrn_docs_path = method_a_dir / "journal_docs.csv"
    flows_path = method_a_dir / "capstone_to_journal_flows.csv"
    if not (cap_docs_path.exists() and jrn_docs_path.exists() and flows_path.exists()):
        print(f"  WARNING: Method A artifacts not in {method_a_dir}; skipping seeded sampling")
        return pairs

    # capstone_id -> topic, pmid -> topic
    cap_topic_by_id: dict[str, int] = {}
    with open(cap_docs_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cap_topic_by_id[r["capstone_id"]] = int(r["topic"])
    jrn_topic_by_pmid: dict[str, int] = {}
    with open(jrn_docs_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            jrn_topic_by_pmid[r["pmid"]] = int(r["topic"])

    # Build doc lists per topic
    cap_docs_by_topic: dict[int, list[str]] = {}
    for cid, t in cap_topic_by_id.items():
        cap_docs_by_topic.setdefault(t, []).append(cid)
    jrn_docs_by_topic: dict[int, list[str]] = {}
    for pmid, t in jrn_topic_by_pmid.items():
        jrn_docs_by_topic.setdefault(t, []).append(pmid)

    # capstone_id / pmid -> embedding row index for cosine recomputation
    cap_index = {cid: i for i, cid in enumerate(cap_ids)}
    jrn_index = {pmid: i for i, pmid in enumerate(jrn_ids)}

    flows = []
    with open(flows_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            flows.append((int(r["capstone_topic"]), int(r["journal_topic"]),
                          int(r["rank"]), float(r["transported_mass"])))
    # Keep only non-outlier topics on both sides (BERTopic outliers = -1).
    flows = [f for f in flows if f[0] != -1 and f[1] != -1]
    if not flows:
        return pairs

    per_combo = max(1, n_seed_target // len(flows))
    seeded: list[dict] = []
    for cap_t, jrn_t, rank, mass in flows:
        cap_pool = cap_docs_by_topic.get(cap_t, [])
        jrn_pool = jrn_docs_by_topic.get(jrn_t, [])
        if not cap_pool or not jrn_pool:
            continue
        for _ in range(per_combo):
            cid = cap_pool[rng.integers(0, len(cap_pool))]
            pmid = jrn_pool[rng.integers(0, len(jrn_pool))]
            i, j = cap_index[cid], jrn_index[pmid]
            if (i, j) in used:
                continue
            used.add((i, j))
            seeded.append({
                "bucket": f"Seed-T{cap_t}->T{jrn_t}-r{rank}",
                "capstone_id": cid,
                "pmid": pmid,
                "cosine_sim": float(sim[i, j]),
            })
            if len(seeded) >= n_seed_target:
                break
        if len(seeded) >= n_seed_target:
            break

    pairs.extend(seeded)
    return pairs


def build_prompt(cap_text: str, journal_title: str, journal_abstract: str,
                 max_cap_chars: int = 3500, max_jrn_chars: int = 2500) -> str:
    cap_excerpt = cap_text[:max_cap_chars]
    jrn_text = f"{journal_title}\n\n{journal_abstract}"[:max_jrn_chars]
    return PROMPT_TEMPLATE.format(capstone_text=cap_excerpt, journal_text=jrn_text)


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    """Single Claude API call returning the parsed JSON rating."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Strip optional code fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually call the LLM API (and incur cost). Without this flag, the script only writes the pair-sample and prompt previews.")
    parser.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    # --- Load Method A embeddings ---
    cap_emb_path = PHASE3 / "capstone_embeddings.npy"
    jrn_emb_path = PHASE3 / "journal_embeddings.npy"
    if not cap_emb_path.exists() or not jrn_emb_path.exists():
        log("ERROR: Method A embeddings not found. Run method_a_bertopic_ot.py first.")
        sys.exit(1)

    cap_emb = np.load(cap_emb_path)
    jrn_emb = np.load(jrn_emb_path)
    log(f"Loaded embeddings — capstones {cap_emb.shape}, journals {jrn_emb.shape}")

    # IDs must line up with the embed order. We assume the embed scripts
    # iterated metadata + journal_corpus.csv in order, so we replay that.
    cap_ids = []
    with open(META_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cap_ids.append(r["capstone_id"])
    jrn_ids = []
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            jrn_ids.append(r["pmid"])
    assert len(cap_ids) == cap_emb.shape[0]
    assert len(jrn_ids) == jrn_emb.shape[0]

    # --- Sample pairs (cosine-quartile + Method A top-flow seeded) ---
    pairs = sample_pairs(cap_emb, cap_ids, jrn_emb, jrn_ids, args.n_pairs, RNG_SEED,
                         method_a_dir=PHASE3)
    n_quartile = sum(1 for p in pairs if p["bucket"].startswith("Q"))
    n_seeded = sum(1 for p in pairs if p["bucket"].startswith("Seed"))
    log(f"Sampled {len(pairs)} pairs: {n_quartile} quartile-stratified + {n_seeded} Method-A seeded")
    pairs_csv = OUT_DIR / "sampled_pairs.csv"
    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "capstone_id", "pmid", "cosine_sim"])
        w.writeheader()
        w.writerows(pairs)
    log(f"Wrote {pairs_csv}")

    # --- Prepare prompts (preview the first 3 so we can eyeball before spending API tokens) ---
    cap_texts = load_capstone_texts()
    jrn_texts = load_journal_texts()
    log("First 3 prompt previews (truncated):")
    for p in pairs[:3]:
        cap = cap_texts.get(p["capstone_id"], "")
        jt = jrn_texts.get(p["pmid"], {})
        prompt = build_prompt(cap, jt.get("title", ""), jt.get("abstract", ""))
        log(f"--- {p['bucket']} cap={p['capstone_id']} pmid={p['pmid']} cos={p['cosine_sim']:.3f} ---")
        log(prompt[:500] + " ...\n")

    if not args.execute:
        log(f"\n--execute not set; stopping after preview. Estimated cost for {args.n_pairs} pairs:")
        log(f"  ~{args.n_pairs * 400 / 1e6:.3f} M input tokens × $3 = ${args.n_pairs * 400 / 1e6 * 3:.2f}")
        log(f"  ~{args.n_pairs * 150 / 1e6:.3f} M output tokens × $15 = ${args.n_pairs * 150 / 1e6 * 15:.2f}")
        sys.exit(0)

    # --- Call LLM ---
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set; aborting.")
        sys.exit(2)

    log(f"Executing {len(pairs)} API calls with model {args.model}...")
    out_csv = OUT_DIR / "ratings.csv"
    fieldnames = ["bucket", "capstone_id", "pmid", "cosine_sim",
                  "thematic_alignment", "shared_methodology", "justification", "error"]

    # Resume support: load existing successful rows by (capstone_id, pmid) so
    # an interrupted run doesn't re-spend API budget.
    done_keys: set[tuple[str, str]] = set()
    existing_rows: list[dict] = []
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") not in ("", None) or r.get("error"):
                    existing_rows.append(r)
                    if not r.get("error"):
                        done_keys.add((r["capstone_id"], r["pmid"]))
        log(f"  resume: {len(done_keys)} pairs already rated; will skip them")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Re-emit existing successful rows up front.
        for r in existing_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        f.flush()

        for i, p in enumerate(pairs, start=1):
            if (p["capstone_id"], p["pmid"]) in done_keys:
                continue
            cap = cap_texts.get(p["capstone_id"], "")
            jt = jrn_texts.get(p["pmid"], {})
            prompt = build_prompt(cap, jt.get("title", ""), jt.get("abstract", ""))
            try:
                rating = call_claude(prompt, model=args.model)
                row = dict(p)
                row.update({k: rating.get(k, "") for k in ["thematic_alignment", "shared_methodology", "justification"]})
                row["error"] = ""
            except Exception as e:
                row = dict(p)
                row["thematic_alignment"] = ""
                row["shared_methodology"] = ""
                row["justification"] = ""
                row["error"] = f"{type(e).__name__}: {e}"
            writer.writerow(row)
            f.flush()
            if i % 25 == 0:
                log(f"  {i}/{len(pairs)} pairs processed (some may be resumes)")

    log(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
