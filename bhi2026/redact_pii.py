"""Stage-0 PII / organization redaction pass.

For each capstone .txt file, replace PERSON and ORG entities (detected by
spaCy en_core_web_sm) with class-name placeholders so downstream embedding and
topic modeling don't surface names of students, mentors, or affiliated
hospitals/companies.

This is intentionally conservative:
- PERSON  -> [PERSON]
- ORG     -> [ORG]
- GPE     -> kept (place names are research-relevant, e.g. "Randolph County")
- DATE    -> kept
- Acronyms <= 3 chars that spaCy mis-tags as ORG (e.g. "AI", "ML", "EMR") are
  whitelisted because they carry topic signal.

Output: raw_data/OneDrive_extracted_text/texts_redacted/<rel>.txt
Also writes bhi2026/metadata/redaction_log.csv with per-document counts and the
unique entities redacted (truncated) for spot-checking.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import spacy

TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
OUT_ROOT = Path("raw_data/OneDrive_extracted_text/texts_redacted")
META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
LOG_PATH = Path("bhi2026/metadata/redaction_log.csv")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Domain acronyms that NER frequently mis-tags as ORG but carry topic signal.
WHITELIST_ACRONYMS = {
    "AI", "ML", "DL", "NLP", "EMR", "EHR", "HIE", "HIT",
    "ICU", "ED", "PHI", "PII", "API", "SQL", "BMI",
    "COVID", "COVID-19", "SARS-CoV-2", "ICD", "CPT",
    "SDOH", "PHQ", "URICA", "GIS", "AWS", "GCP", "Azure",
    "PowerBI", "MIMIC", "FHIR", "HL7", "CDC", "WHO", "NIH",
    "HIPAA", "OCR", "CV", "RAG", "LLM", "GPT", "BERT",
}

# Tokens whose presence in an ORG span marks it as a real organization.
# Only ORG entities matching one of these patterns are redacted; other ORG
# entities (which spaCy frequently mis-assigns to domain phrases like
# "Electronic Medical Record" or "Health Information Exchange") are kept.
# Site-specific organization names (local institutions, hospital systems) are
# loaded from the gitignored local/org_markers.txt, one per line, so the
# public code names no site.
_EXTRA = Path("local/org_markers.txt")
_EXTRA_ORGS = "|".join(
    re.escape(l.strip()) for l in _EXTRA.read_text().splitlines() if l.strip()
) if _EXTRA.exists() else ""
ORG_MARKERS = re.compile(
    r"\b("
    r"University|College|School|Institute|Hospital|Clinic|Medical Center|"
    r"Health System|Healthcare|Health Care|Health Network|"
    r"Corporation|Corp\.?|Company|Inc\.?|LLC|Ltd\.?|Foundation|"
    r"Department of|Faculty of|Pharmaceuticals|Pharma|Laboratories|Labs|"
    r"County|" + ((_EXTRA_ORGS + "|") if _EXTRA_ORGS else "")
    + r"Center for|Bureau|Agency|Society|Council|Association"
    r")\b",
    re.IGNORECASE,
)

# Spacy NLP pipeline — load once at module import.
print("Loading spaCy en_core_web_sm...")
NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])


def redact(text: str) -> tuple[str, Counter[str]]:
    """Return (redacted_text, counts_by_placeholder)."""
    if not text.strip():
        return text, Counter()

    doc = NLP(text)
    # Collect spans to replace, sorted by start so we can rebuild the string.
    spans: list[tuple[int, int, str, str]] = []  # (start, end, label, surface)
    for ent in doc.ents:
        if ent.label_ not in {"PERSON", "ORG"}:
            continue
        surface = ent.text.strip()
        if not surface:
            continue
        # Whitelist short domain acronyms misclassified as ORG.
        if ent.label_ == "ORG" and surface.upper() in WHITELIST_ACRONYMS:
            continue
        if ent.label_ == "ORG" and len(surface) <= 3 and surface.isupper():
            # Other 1-3 char uppercase tokens are probably acronyms we want to keep.
            continue
        # Only redact ORG if the span looks like an actual organization.
        if ent.label_ == "ORG" and not ORG_MARKERS.search(surface):
            continue
        spans.append((ent.start_char, ent.end_char, ent.label_, surface))

    if not spans:
        return text, Counter()

    # Sort and dedupe overlapping spans (keep first).
    spans.sort()
    deduped: list[tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, label, surface in spans:
        if start < last_end:
            continue
        deduped.append((start, end, label, surface))
        last_end = end

    out_parts: list[str] = []
    counts: Counter[str] = Counter()
    cursor = 0
    for start, end, label, surface in deduped:
        out_parts.append(text[cursor:start])
        out_parts.append(f"[{label}]")
        counts[label] += 1
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts), counts


def sample_examples(text: str, max_examples: int = 5) -> list[str]:
    """Return up to N PERSON/ORG surface forms for the audit log."""
    if not text.strip():
        return []
    doc = NLP(text)
    seen: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in {"PERSON", "ORG"}:
            continue
        s = ent.text.strip()
        if not s or s.upper() in WHITELIST_ACRONYMS:
            continue
        if s in seen:
            continue
        seen.append(s)
        if len(seen) >= max_examples:
            break
    return seen


def main() -> None:
    with open(META_CSV, encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    log_rows = []
    for i, row in enumerate(meta_rows, start=1):
        cap_id = row["capstone_id"]
        rel = row["source_path"]
        txt_path = TXT_ROOT / (rel + ".txt")
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

        redacted, counts = redact(text)
        examples = sample_examples(text, max_examples=5)

        out_path = OUT_ROOT / (rel + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(redacted, encoding="utf-8")

        log_rows.append({
            "capstone_id": cap_id,
            "source_path": rel,
            "n_person": counts.get("PERSON", 0),
            "n_org": counts.get("ORG", 0),
            "sample_redacted_surface": "|".join(examples),
        })

        if i % 25 == 0 or i == len(meta_rows):
            print(f"  {i}/{len(meta_rows)}  {rel[:70]}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    total_person = sum(r["n_person"] for r in log_rows)
    total_org = sum(r["n_org"] for r in log_rows)
    docs_with_person = sum(1 for r in log_rows if r["n_person"])
    docs_with_org = sum(1 for r in log_rows if r["n_org"])
    print(f"\nDone. Redacted text in {OUT_ROOT}")
    print(f"  PERSON redactions: {total_person} across {docs_with_person} docs")
    print(f"  ORG    redactions: {total_org} across {docs_with_org} docs")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
