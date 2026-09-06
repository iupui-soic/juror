"""Method D cross-LLM robustness check — re-run the 24 main flows with
OpenAI gpt-5.4-mini and compare to the Claude Sonnet 4.6 ratings.

If the two models agree (Spearman ρ ≥ 0.6 across the 24 flows), Method D's
result is not Claude-specific. If they disagree, that's a reportable
limitation: the flow-level rating depends on the judge.

Cost: 24 calls × ~$0.001 (gpt-mini pricing) ≈ $0.03.

Outputs: bhi2026/phase3_method_d_openai/
  - flow_ratings_openai.csv          — 24 GPT-5.4-mini flow ratings + justifications
  - cross_llm_comparison.csv         — Claude vs OpenAI per flow
  - cross_llm_comparison.png         — scatter
  - report.json                      — correlation summary
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
CLAUDE_RATINGS = Path("bhi2026/phase3_method_d/flow_ratings.csv")
METHOD_C_CSV = Path("bhi2026/phase3_method_c/ratings.csv")

OUT_DIR = Path("bhi2026/phase3_method_d_openai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = "gpt-5.4-mini"
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
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    return ", ".join(parts[1].split("_")[:n])


def load_topic_keywords(path: Path) -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["Topic"])] = parse_keywords(r.get("Name", ""))
            except (KeyError, ValueError):
                continue
    return out


def load_doc_topics(path: Path, id_col: str) -> dict[int, list[str]]:
    by_topic: dict[int, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                by_topic[int(r["topic"])].append(r[id_col])
            except (KeyError, ValueError):
                continue
    return by_topic


def load_capstone_texts() -> dict[str, str]:
    out = {}
    with open(CAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = TXT_ROOT / (r["source_path"] + ".txt")
            out[r["capstone_id"]] = p.read_text(encoding="utf-8") if p.exists() else ""
    return out


def load_journal_texts() -> dict[str, str]:
    out = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["pmid"]] = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
    return out


def call_openai(prompt: str, model: str) -> dict:
    """OpenAI Chat Completions with structured JSON output."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_completion_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    return json.loads(text)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY not set"); sys.exit(2)

    log(f"Loading flows + topics + docs...")
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
    cap_docs_by_t = load_doc_topics(CAP_DOCS, "capstone_id")
    jrn_docs_by_t = load_doc_topics(JRN_DOCS, "pmid")
    cap_texts = load_capstone_texts()
    jrn_texts = load_journal_texts()

    out_csv = OUT_DIR / "flow_ratings_openai.csv"
    done: set[tuple[int, int]] = set()
    existing = []
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") and not r.get("error"):
                    done.add((int(r["capstone_topic"]), int(r["journal_topic"])))
                    existing.append(r)
        log(f"Resume: {len(done)} flows already rated")

    fieldnames = ["capstone_topic", "journal_topic", "rank", "mass", "cosine",
                  "thematic_alignment", "shared_methodology", "justification",
                  "cap_keywords", "jrn_keywords", "error"]

    log(f"Executing {len(flows) - len(done)} OpenAI calls with model {OPENAI_MODEL}...")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        for i, fl in enumerate(flows, start=1):
            key = (fl["capstone_topic"], fl["journal_topic"])
            if key in done:
                continue
            cap_t, jrn_t = key
            cap_doc_ids = cap_docs_by_t[cap_t][:N_REP_DOCS]
            jrn_doc_ids = jrn_docs_by_t[jrn_t][:N_REP_DOCS]
            cap_doc_blocks = [f"  ({k}) {cap_texts.get(cid, '')[:DOC_EXCERPT_CHARS].strip()}"
                              for k, cid in enumerate(cap_doc_ids, 1)]
            jrn_doc_blocks = [f"  ({k}) {jrn_texts.get(pmid, '')[:DOC_EXCERPT_CHARS].strip()}"
                              for k, pmid in enumerate(jrn_doc_ids, 1)]
            prompt = PROMPT_TEMPLATE.format(
                cap_topic_id=cap_t, jrn_topic_id=jrn_t,
                cap_keywords=cap_kw.get(cap_t, "?"),
                jrn_keywords=jrn_kw.get(jrn_t, "?"),
                cap_docs="\n".join(cap_doc_blocks),
                jrn_docs="\n".join(jrn_doc_blocks),
            )
            try:
                rating = call_openai(prompt, OPENAI_MODEL)
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
            w.writerow(row); f.flush()
            log(f"  {i}/{len(flows)} T{cap_t}->T{jrn_t}  rating={row['thematic_alignment'] or 'ERR'}")

    # === Compare to Claude ratings ===
    log("\n=== Comparing GPT-5.4-mini vs Claude Sonnet 4.6 ===")
    openai_rows = [r for r in csv.DictReader(open(out_csv))
                   if r.get("thematic_alignment") and not r.get("error")]
    claude_rows = [r for r in csv.DictReader(open(CLAUDE_RATINGS))
                   if r.get("thematic_alignment") and not r.get("error")]
    openai_by_flow = {(int(r["capstone_topic"]), int(r["journal_topic"])): int(r["thematic_alignment"])
                      for r in openai_rows}
    claude_by_flow = {(int(r["capstone_topic"]), int(r["journal_topic"])): int(r["thematic_alignment"])
                      for r in claude_rows}

    common_keys = sorted(set(openai_by_flow) & set(claude_by_flow))
    log(f"Flows rated by both models: {len(common_keys)}")

    o_arr = np.array([openai_by_flow[k] for k in common_keys])
    c_arr = np.array([claude_by_flow[k] for k in common_keys])

    sp_oc = spearmanr(o_arr, c_arr)
    pe_oc = pearsonr(o_arr, c_arr)
    agree = int((o_arr == c_arr).sum())
    log(f"Exact agreement: {agree}/{len(common_keys)} ({agree/len(common_keys):.1%})")
    log(f"|diff| ≤ 1 agreement: {((np.abs(o_arr - c_arr) <= 1).sum())}/{len(common_keys)} ({((np.abs(o_arr - c_arr) <= 1).sum())/len(common_keys):.1%})")
    log(f"Spearman ρ(GPT, Claude) = {sp_oc.statistic:.3f} (p={sp_oc.pvalue:.3g})")
    log(f"Pearson r(GPT, Claude)  = {pe_oc.statistic:.3f}")
    log(f"GPT mean: {o_arr.mean():.2f}, Claude mean: {c_arr.mean():.2f}")

    # Compare to Method C means within flow
    cap_topic_of: dict[str, int] = {}
    with open(CAP_DOCS) as f:
        for r in csv.DictReader(f):
            cap_topic_of[r["capstone_id"]] = int(r["topic"])
    jrn_topic_of: dict[str, int] = {}
    with open(JRN_DOCS) as f:
        for r in csv.DictReader(f):
            jrn_topic_of[r["pmid"]] = int(r["topic"])
    c_by_flow: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in csv.DictReader(open(METHOD_C_CSV)):
        if r["error"] or not r["thematic_alignment"]:
            continue
        ct = cap_topic_of.get(r["capstone_id"])
        jt = jrn_topic_of.get(r["pmid"])
        if ct is None or jt is None or ct == -1 or jt == -1:
            continue
        c_by_flow[(ct, jt)].append(int(r["thematic_alignment"]))
    c_means = {k: float(np.mean(v)) for k, v in c_by_flow.items() if v}

    # GPT vs Method C
    matched = [(k, openai_by_flow[k], c_means[k]) for k in common_keys if k in c_means]
    log(f"\nGPT vs Method C means: {len(matched)} flows matched")
    if matched:
        gpt_v = np.array([m[1] for m in matched])
        c_v = np.array([m[2] for m in matched])
        sp_gc = spearmanr(gpt_v, c_v)
        pe_gc = pearsonr(gpt_v, c_v)
        log(f"  Spearman ρ(GPT, Method C) = {sp_gc.statistic:.3f}  (p={sp_gc.pvalue:.3g})")
        log(f"  Pearson r(GPT, Method C)  = {pe_gc.statistic:.3f}")

    # Save comparison CSV
    cmp_csv = OUT_DIR / "cross_llm_comparison.csv"
    with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capstone_topic", "journal_topic", "rank",
                    "gpt5_4_mini_rating", "claude_sonnet_4_6_rating", "diff",
                    "method_c_pair_mean"])
        for k in common_keys:
            w.writerow([k[0], k[1], "?",
                        openai_by_flow[k], claude_by_flow[k], openai_by_flow[k] - claude_by_flow[k],
                        round(c_means.get(k, float("nan")), 4) if k in c_means else ""])

    # Report
    report = {
        "openai_model": OPENAI_MODEL,
        "claude_model": "claude-sonnet-4-6",
        "n_flows_both_rated": len(common_keys),
        "exact_agreement": agree,
        "exact_agreement_rate": agree / len(common_keys),
        "diff_le_1_rate": float((np.abs(o_arr - c_arr) <= 1).mean()),
        "spearman_gpt_vs_claude": float(sp_oc.statistic),
        "pearson_gpt_vs_claude": float(pe_oc.statistic),
        "spearman_p": float(sp_oc.pvalue),
        "gpt_mean": float(o_arr.mean()),
        "claude_mean": float(c_arr.mean()),
    }
    if matched:
        report["gpt_vs_method_c"] = {
            "n_matched": len(matched),
            "spearman": float(sp_gc.statistic),
            "pearson": float(pe_gc.statistic),
        }
        # Also fetch the original Claude-vs-Method-C numbers for comparison
        claude_report_path = Path("bhi2026/phase3_method_d/validation_report.json")
        if claude_report_path.exists():
            cr = json.loads(claude_report_path.read_text())
            report["claude_vs_method_c"] = {
                "n_matched": cr["n_flows_with_method_c_overlap"],
                "spearman": cr["spearman"],
                "pearson": cr["pearson"],
            }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    log(f"Wrote {OUT_DIR / 'report.json'}")

    # Scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in common_keys:
        ax.scatter(claude_by_flow[k], openai_by_flow[k],
                   s=70, alpha=0.75)
        ax.annotate(f"T{k[0]}->T{k[1]}", (claude_by_flow[k], openai_by_flow[k]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    lo = min(o_arr.min(), c_arr.min()) - 0.5
    hi = max(o_arr.max(), c_arr.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Claude Sonnet 4.6 rating")
    ax.set_ylabel(f"{OPENAI_MODEL} rating")
    ax.set_title(f"Method D cross-LLM — Spearman ρ = {sp_oc.statistic:.3f} (n={len(common_keys)} flows)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_llm_comparison.png", dpi=120)
    plt.close(fig)
    log(f"Wrote {OUT_DIR / 'cross_llm_comparison.png'}")


if __name__ == "__main__":
    main()
