"""Method D — Flow-level LLM-bridged Optimal Transport.

A new method that operates at the OT flow granularity: one capstone topic →
one journal topic edge in the transport plan = one LLM call. Concretely:

  For each non-trivial transport flow T[i,j] (top-3 destinations per capstone
  topic; 24 flows total in our corpus):

    1. Extract Topic A (capstone topic i): top keywords + 3 representative
       documents.
    2. Extract Topic B (journal topic j): top keywords + 3 representative
       documents.
    3. Single LLM call: rate the thematic alignment of these topics 0-4 with
       the same rubric as Method C, plus a free-form justification.
    4. The flow alignment score is then aggregated by transport mass.

Validation hypothesis: the flow-level rating from Method D should track the
average of Method C's per-pair ratings *within the same flow*. If Spearman
ρ ≥ 0.6 across the 24 flows, the method recovers the pair-level signal at
~25× lower API cost.

Cost: 24 calls × ~$0.008 = ~$0.20.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

csv.field_size_limit(sys.maxsize)

FLOWS_CSV = Path("bhi2026/phase3_method_a/capstone_to_journal_flows.csv")
CAP_TOPICS = Path("bhi2026/phase3_method_a/capstone_topics.csv")
JRN_TOPICS = Path("bhi2026/phase3_method_a/journal_topics.csv")
CAP_DOCS = Path("bhi2026/phase3_method_a/capstone_docs.csv")
JRN_DOCS = Path("bhi2026/phase3_method_a/journal_docs.csv")
CAP_META = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
METHOD_C_CSV = Path("bhi2026/phase3_method_c/ratings.csv")

OUT_DIR = Path("bhi2026/phase3_method_d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_KEYWORDS = 12
N_REP_DOCS = 3
DOC_EXCERPT_CHARS = 600

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


def parse_keywords(name: str, n: int = N_KEYWORDS) -> str:
    """Parse BERTopic's 'Name' column into a comma-separated keyword string."""
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    kws = parts[1].split("_")
    return ", ".join(kws[:n])


def load_topic_keywords(path: Path, key_col: str = "Topic") -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r[key_col])
            except (KeyError, ValueError):
                continue
            out[t] = parse_keywords(r.get("Name", ""))
    return out


def load_doc_topics(path: Path, id_col: str) -> dict[int, list[str]]:
    """topic_id -> list of doc ids in that topic, in file order."""
    by_topic: dict[int, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["topic"])
            except (KeyError, ValueError):
                continue
            by_topic[t].append(r[id_col])
    return by_topic


def load_capstone_texts() -> dict[str, str]:
    """capstone_id -> source_path full text."""
    out = {}
    with open(CAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = TXT_ROOT / (r["source_path"] + ".txt")
            out[r["capstone_id"]] = p.read_text(encoding="utf-8") if p.exists() else ""
    return out


def load_journal_texts() -> dict[str, str]:
    """pmid -> 'title. abstract'."""
    out = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["pmid"]] = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
    return out


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    log("Loading flows + topics + docs...")
    flows = []
    with open(FLOWS_CSV) as f:
        for r in csv.DictReader(f):
            flows.append({
                "capstone_topic": int(r["capstone_topic"]),
                "journal_topic": int(r["journal_topic"]),
                "rank": int(r["rank"]),
                "mass": float(r["transported_mass"]),
                "cosine": float(r["cosine_sim"]),
            })
    log(f"  {len(flows)} flows to rate")

    cap_kw = load_topic_keywords(CAP_TOPICS)
    jrn_kw = load_topic_keywords(JRN_TOPICS)
    cap_by_topic = load_doc_topics(CAP_DOCS, "capstone_id")
    jrn_by_topic = load_doc_topics(JRN_DOCS, "pmid")
    cap_texts = load_capstone_texts()
    jrn_texts = load_journal_texts()

    # Resume support
    out_csv = OUT_DIR / "flow_ratings.csv"
    done: set[tuple[int, int]] = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") and not r.get("error"):
                    done.add((int(r["capstone_topic"]), int(r["journal_topic"])))
        log(f"Resume: {len(done)} flows already rated")

    fieldnames = ["capstone_topic", "journal_topic", "rank", "mass", "cosine",
                  "thematic_alignment", "shared_methodology", "justification",
                  "cap_keywords", "jrn_keywords", "error"]
    rows_existing = []
    if out_csv.exists():
        with open(out_csv) as f:
            rows_existing = [r for r in csv.DictReader(f)]

    log(f"Executing LLM calls for {len(flows) - len(done)} new flows...")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        for i, fl in enumerate(flows, start=1):
            key = (fl["capstone_topic"], fl["journal_topic"])
            if key in done:
                continue
            cap_t, jrn_t = key
            cap_doc_ids = cap_by_topic[cap_t][:N_REP_DOCS]
            jrn_doc_ids = jrn_by_topic[jrn_t][:N_REP_DOCS]
            cap_doc_blocks = []
            for k, cid in enumerate(cap_doc_ids, 1):
                excerpt = cap_texts.get(cid, "")[:DOC_EXCERPT_CHARS].strip()
                cap_doc_blocks.append(f"  ({k}) {excerpt}")
            jrn_doc_blocks = []
            for k, pmid in enumerate(jrn_doc_ids, 1):
                excerpt = jrn_texts.get(pmid, "")[:DOC_EXCERPT_CHARS].strip()
                jrn_doc_blocks.append(f"  ({k}) {excerpt}")
            prompt = PROMPT_TEMPLATE.format(
                cap_topic_id=cap_t,
                jrn_topic_id=jrn_t,
                cap_keywords=cap_kw.get(cap_t, "?"),
                jrn_keywords=jrn_kw.get(jrn_t, "?"),
                cap_docs="\n".join(cap_doc_blocks),
                jrn_docs="\n".join(jrn_doc_blocks),
            )
            try:
                rating = call_claude(prompt)
                row = {
                    "capstone_topic": cap_t, "journal_topic": jrn_t,
                    "rank": fl["rank"], "mass": fl["mass"], "cosine": fl["cosine"],
                    "thematic_alignment": rating.get("thematic_alignment", ""),
                    "shared_methodology": rating.get("shared_methodology", ""),
                    "justification": rating.get("justification", ""),
                    "cap_keywords": cap_kw.get(cap_t, ""),
                    "jrn_keywords": jrn_kw.get(jrn_t, ""),
                    "error": "",
                }
            except Exception as e:
                row = {
                    "capstone_topic": cap_t, "journal_topic": jrn_t,
                    "rank": fl["rank"], "mass": fl["mass"], "cosine": fl["cosine"],
                    "thematic_alignment": "", "shared_methodology": "",
                    "justification": "",
                    "cap_keywords": cap_kw.get(cap_t, ""),
                    "jrn_keywords": jrn_kw.get(jrn_t, ""),
                    "error": f"{type(e).__name__}: {e}",
                }
            w.writerow(row)
            f.flush()
            log(f"  {i}/{len(flows)} flow T{cap_t}->T{jrn_t} rated={row['thematic_alignment']}")
    log(f"Wrote {out_csv}")

    # --- Validation: compare to Method C pair ratings within the same flow ---
    log("\n=== Validation against Method C pair ratings ===")
    # Read flow ratings
    flow_rows = list(csv.DictReader(open(out_csv)))
    flow_rows = [r for r in flow_rows if r["thematic_alignment"] and not r["error"]]
    log(f"Valid flow ratings: {len(flow_rows)}")

    # Read Method C pair ratings and the per-pair correlations file (to get cap_topic/jrn_topic from there).
    pair_rows = list(csv.DictReader(open(METHOD_C_CSV)))
    pair_rows = [r for r in pair_rows if not r["error"]]
    log(f"Method C pair ratings: {len(pair_rows)}")

    # We need to map each Method C pair to its (cap_topic, jrn_topic). Use the docs CSVs.
    cap_topic_of: dict[str, int] = {}
    with open(CAP_DOCS) as f:
        for r in csv.DictReader(f):
            cap_topic_of[r["capstone_id"]] = int(r["topic"])
    jrn_topic_of: dict[str, int] = {}
    with open(JRN_DOCS) as f:
        for r in csv.DictReader(f):
            jrn_topic_of[r["pmid"]] = int(r["topic"])

    # Group pair ratings by (cap_topic, jrn_topic)
    pair_by_flow: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in pair_rows:
        ct = cap_topic_of.get(r["capstone_id"])
        jt = jrn_topic_of.get(r["pmid"])
        if ct is None or jt is None or ct == -1 or jt == -1:
            continue
        pair_by_flow[(ct, jt)].append(int(r["thematic_alignment"]))

    log(f"Distinct (cap_topic, jrn_topic) flows seen in Method C: {len(pair_by_flow)}")

    # For each flow with both a Method D rating and >=1 Method C pair rating, build the validation table.
    val_rows = []
    for fr in flow_rows:
        ct = int(fr["capstone_topic"])
        jt = int(fr["journal_topic"])
        pair_ratings = pair_by_flow.get((ct, jt), [])
        val_rows.append({
            "capstone_topic": ct, "journal_topic": jt,
            "method_d_rating": int(fr["thematic_alignment"]),
            "method_c_n_pairs": len(pair_ratings),
            "method_c_mean": float(np.mean(pair_ratings)) if pair_ratings else None,
            "method_c_max": float(np.max(pair_ratings)) if pair_ratings else None,
            "cap_keywords": fr["cap_keywords"][:80],
            "jrn_keywords": fr["jrn_keywords"][:80],
            "mass": float(fr["mass"]),
        })
    val_csv = OUT_DIR / "method_d_vs_method_c.csv"
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        w.writeheader(); w.writerows(val_rows)
    log(f"Wrote {val_csv}")

    # Per-flow correlation
    matched = [r for r in val_rows if r["method_c_mean"] is not None]
    if len(matched) >= 3:
        d_arr = np.array([r["method_d_rating"] for r in matched])
        c_arr = np.array([r["method_c_mean"] for r in matched])
        sp = spearmanr(d_arr, c_arr)
        pe = pearsonr(d_arr, c_arr)
        log(f"\nValidation across {len(matched)} flows:")
        log(f"  Spearman ρ(Method D, mean Method C) = {sp.statistic:.3f} (p={sp.pvalue:.3g})")
        log(f"  Pearson r(Method D, mean Method C)  = {pe.statistic:.3f} (p={pe.pvalue:.3g})")
        log(f"  Method D mean: {d_arr.mean():.2f}, Method C mean: {c_arr.mean():.2f}")
        log(f"  Threshold for methods contribution: Spearman ≥ 0.6 → {'PASS' if sp.statistic >= 0.6 else 'FAIL'}")

        # Quick mass-weighted aggregate
        mass_arr = np.array([r["mass"] for r in matched])
        weighted_d = float((d_arr * mass_arr).sum() / mass_arr.sum())
        log(f"  Method D mass-weighted alignment: {weighted_d:.3f} / 4")

        # Save report
        report = {
            "n_flows_rated": len(flow_rows),
            "n_flows_with_method_c_overlap": len(matched),
            "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
            "method_d_mean_rating": float(d_arr.mean()),
            "method_c_mean_of_means": float(c_arr.mean()),
            "method_d_mass_weighted": weighted_d,
            "threshold_pass": bool(sp.statistic >= 0.6),
        }
        (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
        log(f"Wrote {OUT_DIR / 'validation_report.json'}")

        # Scatter
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for r in matched:
            ax.scatter(r["method_c_mean"], r["method_d_rating"], s=60, alpha=0.75)
            label = f"T{r['capstone_topic']}->T{r['journal_topic']}"
            ax.annotate(label, (r["method_c_mean"], r["method_d_rating"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        # Diagonal
        lo = min(c_arr.min(), d_arr.min()) - 0.2
        hi = max(c_arr.max(), d_arr.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
        ax.set_xlabel("Method C mean pair rating within flow")
        ax.set_ylabel("Method D single flow-level LLM rating")
        ax.set_title(f"Method D vs Method C validation — Spearman ρ = {sp.statistic:.3f} (n={len(matched)} flows)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "method_d_vs_method_c.png", dpi=120)
        plt.close(fig)
        log(f"Wrote {OUT_DIR / 'method_d_vs_method_c.png'}")


if __name__ == "__main__":
    main()
