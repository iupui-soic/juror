"""Method D ablations / stress tests.

  A. Keywords-only — drop the representative document excerpts. Does the LLM
     do real semantic reasoning on the doc content, or is it pattern-matching
     keywords? If ablation A's Spearman ρ stays > 0.6, we have an even cheaper
     variant; if it collapses, the rep docs were doing real work.

  B. Top-5 flows — extend from top-3 per capstone topic (24 flows) to top-5
     (40 flows). Tests whether the method scales beyond the dominant flows.

  C. Cosine-centroid validation — does Method D's rating correlate with the
     simple cosine similarity of topic centroids? If yes, both methods are
     consistent with embedding-space geometry. If Method D is more
     discriminative than cosine alone, that's evidence the LLM adds value.

Outputs to bhi2026/phase3_method_d_ablations/.
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

PHASE3_A = Path("bhi2026/phase3_method_a")
PHASE3_D = Path("bhi2026/phase3_method_d")
CAP_TOPICS = PHASE3_A / "capstone_topics.csv"
JRN_TOPICS = PHASE3_A / "journal_topics.csv"
CAP_DOCS = PHASE3_A / "capstone_docs.csv"
JRN_DOCS = PHASE3_A / "journal_docs.csv"
CAP_META = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
METHOD_C_CSV = Path("bhi2026/phase3_method_c/ratings.csv")
CAP_CENT = PHASE3_A / "capstone_topic_centroids.npy"
JRN_CENT = PHASE3_A / "journal_topic_centroids.npy"

OUT_DIR = Path("bhi2026/phase3_method_d_ablations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_KEYWORDS = 12
N_REP_DOCS = 3
DOC_EXCERPT_CHARS = 600

EPS = 1e-12

PROMPT_WITH_DOCS = """You are an expert reviewer of biomedical informatics research. Two corpora have been clustered into topics using BERTopic. Below are descriptions of one topic from each corpus. Rate the THEMATIC ALIGNMENT between the two topics on a 0-4 scale.

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

PROMPT_KEYWORDS_ONLY = """You are an expert reviewer of biomedical informatics research. Two corpora have been clustered into topics using BERTopic. Below are the top keywords of one topic from each corpus. Rate the THEMATIC ALIGNMENT between the two topics on a 0-4 scale.

0 — Unrelated. The topics cover completely different problems and methods.
1 — Tangentially related. They share a broad domain (e.g., both are healthcare) but tackle different research questions and use different methods.
2 — Adjacent. They share either the research question or the method, but not both.
3 — Substantially overlapping. They share the same research question and use overlapping methods, but the work-products are distinct.
4 — Same research question. They essentially address the same research question with comparable methods, even if the populations or settings differ.

Additionally indicate SHARED METHODOLOGY:
"yes" if the topics' methods clearly overlap.
"partial" if one technique is shared but the rest of the pipeline differs.
"no" if the methods are unrelated.

Provide a JUSTIFICATION of no more than 60 words.

Return ONLY a JSON object with this schema:
{{
  "thematic_alignment": <int 0-4>,
  "shared_methodology": "yes" | "partial" | "no",
  "justification": "<string>"
}}

=== Topic A — Capstone topic {cap_topic_id} ===
Top keywords: {cap_keywords}

=== Topic B — Journal topic {jrn_topic_id} ===
Top keywords: {jrn_keywords}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_keywords(name: str, n: int = N_KEYWORDS) -> str:
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    kws = parts[1].split("_")
    return ", ".join(kws[:n])


def load_topic_keywords(path: Path) -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["Topic"])
            except (KeyError, ValueError):
                continue
            out[t] = parse_keywords(r.get("Name", ""))
    return out


def load_doc_topics(path: Path, id_col: str) -> dict[int, list[str]]:
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


def topic_size(docs_by_topic: dict[int, list[str]], topic_ids: list[int]) -> np.ndarray:
    """Empirical mass per topic, excluding -1 outliers."""
    sizes = np.array([len(docs_by_topic[t]) for t in topic_ids], dtype=float)
    return sizes / (sizes.sum() + EPS)


def compute_top_k_flows(top_k: int) -> list[dict]:
    """Recompute OT transport plan and return top-k flows per capstone topic."""
    import ot
    from sklearn.metrics.pairwise import cosine_similarity

    C_cap = np.load(CAP_CENT)
    C_jrn = np.load(JRN_CENT)
    cap_topic_ids = list(range(C_cap.shape[0]))
    jrn_topic_ids = list(range(C_jrn.shape[0]))

    cap_docs = load_doc_topics(CAP_DOCS, "capstone_id")
    jrn_docs = load_doc_topics(JRN_DOCS, "pmid")
    a = topic_size(cap_docs, cap_topic_ids)
    b = topic_size(jrn_docs, jrn_topic_ids)
    sim = cosine_similarity(C_cap, C_jrn)
    cost = 1 - sim
    T = ot.emd(a, b, cost)

    flows = []
    for i, ct in enumerate(cap_topic_ids):
        row = T[i]
        top_idx = np.argsort(-row)[:top_k]
        for rank, j in enumerate(top_idx, start=1):
            if row[j] <= 0:
                continue
            flows.append({
                "capstone_topic": ct,
                "rank": rank,
                "journal_topic": jrn_topic_ids[j],
                "transported_mass": float(row[j]),
                "cosine_sim": float(sim[i, j]),
            })
    return flows


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


def rate_flows(flows: list[dict], prompt_template: str, label: str,
               cap_kw: dict[int, str], jrn_kw: dict[int, str],
               cap_docs_by_t: dict[int, list[str]],
               jrn_docs_by_t: dict[int, list[str]],
               cap_texts: dict[str, str], jrn_texts: dict[str, str],
               out_path: Path) -> list[dict]:
    """Rate each flow with the given prompt. Resume-supported."""
    done: set[tuple[int, int]] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") and not r.get("error"):
                    done.add((int(r["capstone_topic"]), int(r["journal_topic"])))
                    existing_rows.append(r)
        log(f"  resume: {len(done)} of {len(flows)} already rated")

    fieldnames = ["capstone_topic", "journal_topic", "rank", "mass", "cosine",
                  "thematic_alignment", "shared_methodology", "justification",
                  "cap_keywords", "jrn_keywords", "error"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        for i, fl in enumerate(flows, start=1):
            key = (fl["capstone_topic"], fl["journal_topic"])
            if key in done:
                continue
            cap_t, jrn_t = key
            cap_doc_ids = cap_docs_by_t[cap_t][:N_REP_DOCS]
            jrn_doc_ids = jrn_docs_by_t[jrn_t][:N_REP_DOCS]
            cap_doc_blocks = []
            for k, cid in enumerate(cap_doc_ids, 1):
                cap_doc_blocks.append(f"  ({k}) {cap_texts.get(cid, '')[:DOC_EXCERPT_CHARS].strip()}")
            jrn_doc_blocks = []
            for k, pmid in enumerate(jrn_doc_ids, 1):
                jrn_doc_blocks.append(f"  ({k}) {jrn_texts.get(pmid, '')[:DOC_EXCERPT_CHARS].strip()}")
            prompt = prompt_template.format(
                cap_topic_id=cap_t, jrn_topic_id=jrn_t,
                cap_keywords=cap_kw.get(cap_t, "?"),
                jrn_keywords=jrn_kw.get(jrn_t, "?"),
                cap_docs="\n".join(cap_doc_blocks),
                jrn_docs="\n".join(jrn_doc_blocks),
            )
            try:
                rating = call_claude(prompt)
                row = {
                    "capstone_topic": cap_t, "journal_topic": jrn_t,
                    "rank": fl["rank"], "mass": fl["transported_mass"],
                    "cosine": fl["cosine_sim"],
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
                    "rank": fl["rank"], "mass": fl["transported_mass"],
                    "cosine": fl["cosine_sim"],
                    "thematic_alignment": "", "shared_methodology": "", "justification": "",
                    "cap_keywords": cap_kw.get(cap_t, ""),
                    "jrn_keywords": jrn_kw.get(jrn_t, ""),
                    "error": f"{type(e).__name__}: {e}",
                }
            w.writerow(row); f.flush()
            log(f"  [{label}] {i}/{len(flows)} T{cap_t}->T{jrn_t} rating={row['thematic_alignment']}")

    out = list(csv.DictReader(open(out_path)))
    return [r for r in out if r.get("thematic_alignment") and not r.get("error")]


def method_c_means_by_flow() -> dict[tuple[int, int], dict]:
    """Compute Method C mean rating per (cap_topic, jrn_topic) flow."""
    cap_topic_of: dict[str, int] = {}
    with open(CAP_DOCS) as f:
        for r in csv.DictReader(f):
            cap_topic_of[r["capstone_id"]] = int(r["topic"])
    jrn_topic_of: dict[str, int] = {}
    with open(JRN_DOCS) as f:
        for r in csv.DictReader(f):
            jrn_topic_of[r["pmid"]] = int(r["topic"])

    by_flow: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in csv.DictReader(open(METHOD_C_CSV)):
        if r["error"] or not r["thematic_alignment"]:
            continue
        ct = cap_topic_of.get(r["capstone_id"])
        jt = jrn_topic_of.get(r["pmid"])
        if ct is None or jt is None or ct == -1 or jt == -1:
            continue
        by_flow[(ct, jt)].append(int(r["thematic_alignment"]))

    out: dict[tuple[int, int], dict] = {}
    for k, vals in by_flow.items():
        out[k] = {"n": len(vals), "mean": float(np.mean(vals)), "max": int(np.max(vals))}
    return out


def validate(label: str, flow_rows: list[dict], c_by_flow: dict[tuple[int, int], dict]) -> dict:
    matched_d = []
    matched_c = []
    matched_keys = []
    for r in flow_rows:
        ct = int(r["capstone_topic"]); jt = int(r["journal_topic"])
        c = c_by_flow.get((ct, jt))
        if c:
            matched_d.append(int(r["thematic_alignment"]))
            matched_c.append(c["mean"])
            matched_keys.append((ct, jt))
    if len(matched_d) < 3:
        return {"label": label, "n_matched": len(matched_d), "spearman": None, "pearson": None}
    sp = spearmanr(matched_d, matched_c)
    pe = pearsonr(matched_d, matched_c)
    return {
        "label": label,
        "n_total_flows": len(flow_rows),
        "n_matched_with_method_c": len(matched_d),
        "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
        "method_d_mean": float(np.mean(matched_d)),
        "method_c_mean_of_means": float(np.mean(matched_c)),
    }


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    cap_kw = load_topic_keywords(CAP_TOPICS)
    jrn_kw = load_topic_keywords(JRN_TOPICS)
    cap_docs_by_t = load_doc_topics(CAP_DOCS, "capstone_id")
    jrn_docs_by_t = load_doc_topics(JRN_DOCS, "pmid")
    cap_texts = load_capstone_texts()
    jrn_texts = load_journal_texts()
    c_by_flow = method_c_means_by_flow()

    # Use the existing top-3 flows from method_d
    top3_flows = []
    with open(PHASE3_A / "capstone_to_journal_flows.csv") as f:
        for r in csv.DictReader(f):
            top3_flows.append({
                "capstone_topic": int(r["capstone_topic"]),
                "journal_topic": int(r["journal_topic"]),
                "rank": int(r["rank"]),
                "transported_mass": float(r["transported_mass"]),
                "cosine_sim": float(r["cosine_sim"]),
            })

    results = {}

    # === Ablation A: keywords-only on the same top-3 flows ===
    log("=== Ablation A: keywords-only (24 flows) ===")
    a_path = OUT_DIR / "ablation_A_keywords_only.csv"
    a_rows = rate_flows(top3_flows, PROMPT_KEYWORDS_ONLY, "A",
                        cap_kw, jrn_kw, cap_docs_by_t, jrn_docs_by_t,
                        cap_texts, jrn_texts, a_path)
    results["ablation_A_keywords_only"] = validate("A_keywords_only", a_rows, c_by_flow)
    r = results["ablation_A_keywords_only"]
    log(f"  Spearman vs Method C = {r['spearman']:.3f}  Pearson = {r['pearson']:.3f}  n_matched = {r['n_matched_with_method_c']}")

    # === Ablation B: top-5 flows (40 calls) using full prompt ===
    log("\n=== Ablation B: top-5 flows ===")
    top5_flows = compute_top_k_flows(5)
    log(f"  generated {len(top5_flows)} top-5 flows")
    b_path = OUT_DIR / "ablation_B_top5_full.csv"
    b_rows = rate_flows(top5_flows, PROMPT_WITH_DOCS, "B",
                        cap_kw, jrn_kw, cap_docs_by_t, jrn_docs_by_t,
                        cap_texts, jrn_texts, b_path)
    results["ablation_B_top5_full"] = validate("B_top5_full", b_rows, c_by_flow)
    r = results["ablation_B_top5_full"]
    log(f"  Spearman vs Method C = {r['spearman']:.3f}  Pearson = {r['pearson']:.3f}  n_matched = {r['n_matched_with_method_c']}")

    # === Ablation C: cosine-centroit-only baseline (no LLM) ===
    log("\n=== Ablation C: cosine-centroid baseline (no LLM) ===")
    # Use the cosine of each top-3 flow (already stored) vs Method C mean.
    cos_d = []; mean_c = []
    for fl in top3_flows:
        c = c_by_flow.get((fl["capstone_topic"], fl["journal_topic"]))
        if c:
            cos_d.append(fl["cosine_sim"]); mean_c.append(c["mean"])
    sp = spearmanr(cos_d, mean_c); pe = pearsonr(cos_d, mean_c)
    results["ablation_C_cosine_only"] = {
        "n_matched": len(cos_d),
        "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
    }
    log(f"  Spearman(cosine, Method C mean) = {sp.statistic:.3f}  Pearson = {pe.statistic:.3f}  n={len(cos_d)}")

    # === Side-by-side summary ===
    # Pull main Method D result for context
    main_d = None
    try:
        with open(PHASE3_D / "validation_report.json") as f:
            main_d = json.load(f)
    except FileNotFoundError:
        pass
    if main_d:
        results["method_d_main"] = {
            "label": "main_top3_with_docs",
            "n_matched_with_method_c": main_d["n_flows_with_method_c_overlap"],
            "spearman": main_d["spearman"], "pearson": main_d["pearson"],
        }

    (OUT_DIR / "ablation_report.json").write_text(json.dumps(results, indent=2))
    log(f"\nWrote {OUT_DIR / 'ablation_report.json'}")

    # === Render side-by-side comparison ===
    rows = []
    for key in ["method_d_main", "ablation_A_keywords_only", "ablation_B_top5_full", "ablation_C_cosine_only"]:
        if key not in results:
            continue
        r = results[key]
        rows.append({
            "variant": key,
            "n_matched": r.get("n_matched_with_method_c") or r.get("n_matched"),
            "spearman_vs_method_c": round(r["spearman"], 4),
            "pearson_vs_method_c": round(r["pearson"], 4),
        })
    out_csv = OUT_DIR / "ablation_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"Wrote {out_csv}")

    log("\n=== Side-by-side ===")
    log(f"{'Variant':<35} n  Spearman ρ   Pearson r")
    for r in rows:
        log(f"  {r['variant']:<33} {r['n_matched']:>3}  {r['spearman_vs_method_c']:>6.3f}      {r['pearson_vs_method_c']:>6.3f}")


if __name__ == "__main__":
    main()
