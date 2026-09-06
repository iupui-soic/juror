"""Section-aware extraction layered on top of the flat text extraction.

For each capstone, split its extracted text into the standard poster sections
(Introduction, Practicum Scope, Learning Objectives, Practicum Duties,
Practicum Outcomes, Conclusion, plus Methods/Results/References when present)
and emit a per-document JSON file.

Output structure (one JSON per capstone, keyed by capstone_id):
{
  "capstone_id": "cap_0001",
  "source_path": "FA20 B584 Presentations/Presentation/...",
  "sections": {
    "Introduction": "...",
    "Practicum Scope": "...",
    ...
  },
  "preamble": "...",   # text before any detected section heading
  "full_text": "..."   # concatenation, for embeddings that want it
}

Strategy:
- Use the standard capstone poster section headings, with a few additions.
- For each section heading H_i found in the document, the section's text is
  everything from the end of H_i to the start of the next H_j.
- Posters often repeat headings ("Introduction Introduction") because of
  duplicate slide masters; collapse consecutive duplicates.
- For documents where no heading is found, the entire text is stored as the
  "preamble" and `sections` is empty.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")
OUT_DIR = Path("bhi2026/sections")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order matters: longer/more specific phrases first so they're matched before
# shorter substrings (e.g. "Practicum Outcomes Learning Objectives" before
# "Practicum Outcomes" alone).
SECTION_HEADINGS = [
    "Practicum Outcomes Learning Objectives",
    "Learning Objectives",
    "Practicum Outcomes",
    "Practicum Scope",
    "Practicum Duties",
    "Introduction",
    "Background",
    "Objective",
    "Objectives",
    "Methods",
    "Methodology",
    "Results",
    "Findings",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "References",
    "Timeline",
    "Purpose",
    "Significance",
    "Abstract",
]


def build_heading_regex() -> re.Pattern[str]:
    alt = "|".join(re.escape(h) for h in SECTION_HEADINGS)
    return re.compile(rf"(?<![A-Za-z])({alt})(?![A-Za-z])", re.IGNORECASE)


HEADING_RE = build_heading_regex()


def normalize_heading(matched: str) -> str:
    """Canonicalize a matched heading string to its title-case form."""
    lower = matched.lower()
    for canon in SECTION_HEADINGS:
        if canon.lower() == lower:
            return canon
    return matched.title()


def split_into_sections(text: str) -> tuple[str, dict[str, str]]:
    """Return (preamble, sections_dict).

    Consecutive matches of the same heading are collapsed (poster slides often
    duplicate the heading once per slide).
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return text.strip(), {}

    # Collapse consecutive same-name headings into a single boundary.
    collapsed: list[re.Match[str]] = []
    for m in matches:
        if collapsed and normalize_heading(collapsed[-1].group(1)) == normalize_heading(m.group(1)):
            # Skip; keep the first occurrence as the boundary.
            continue
        collapsed.append(m)

    preamble = text[: collapsed[0].start()].strip()
    sections: dict[str, list[str]] = {}
    for i, m in enumerate(collapsed):
        name = normalize_heading(m.group(1))
        start = m.end()
        end = collapsed[i + 1].start() if i + 1 < len(collapsed) else len(text)
        body = text[start:end].strip()
        sections.setdefault(name, []).append(body)

    # Join repeated occurrences of the same heading with a newline.
    merged = {name: "\n".join(b for b in bodies if b).strip()
              for name, bodies in sections.items()}
    # Drop sections that ended up empty after stripping.
    merged = {k: v for k, v in merged.items() if v}
    return preamble, merged


def main() -> None:
    with open(META_CSV, encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    summary = []
    for row in meta_rows:
        cap_id = row["capstone_id"]
        src_path = row["source_path"]
        txt_path = TXT_ROOT / (src_path + ".txt")
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

        preamble, sections = split_into_sections(text)

        out = {
            "capstone_id": cap_id,
            "source_path": src_path,
            "preamble": preamble,
            "sections": sections,
            "full_text": text,
        }
        out_path = OUT_DIR / f"{cap_id}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        summary.append({
            "capstone_id": cap_id,
            "num_sections": len(sections),
            "section_names": "|".join(sections.keys()),
            "preamble_words": len(preamble.split()),
            "total_section_words": sum(len(v.split()) for v in sections.values()),
        })

    # Write summary CSV
    summary_path = OUT_DIR.parent / "metadata/section_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {len(summary)} per-capstone JSON files to {OUT_DIR}")
    print(f"Wrote section summary to {summary_path}")

    sec_counts = {}
    for s in summary:
        sec_counts[s["num_sections"]] = sec_counts.get(s["num_sections"], 0) + 1
    print("\nDistribution of section counts per document:")
    for n in sorted(sec_counts):
        print(f"  {n} sections: {sec_counts[n]} docs")

    common = {}
    for s in summary:
        for name in s["section_names"].split("|"):
            if name:
                common[name] = common.get(name, 0) + 1
    print("\nMost common detected sections:")
    for name, c in sorted(common.items(), key=lambda x: -x[1])[:12]:
        print(f"  {name:<40s} {c}")


if __name__ == "__main__":
    main()
