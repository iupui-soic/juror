"""Scrub person and organisation surfaces out of generated text before release.

The capstone corpus is redacted for PERSON and ORG at Stage 0, but BERTopic
keyword representations and LLM justifications are generated downstream and can
reintroduce a surface that survived in the source text -- c-TF-IDF will promote a
supervisor's surname to a topic keyword, lower-cased. Marker lists are read from
the gitignored local/person_markers.txt and local/org_markers.txt, so this file
names no one.

Usage:
    from redact_surfaces import scrub, scrub_csv
    scrub_csv(path, ["justification", "cap_keywords", "jrn_keywords"])
"""
import csv, re, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
_LOCAL = Path(__file__).resolve().parent.parent.parent / "local"
_ORGS = _LOCAL / "org_markers.txt"
_PERSONS = _LOCAL / "person_markers.txt"


def _patterns():
    """(regex, placeholder) pairs, longest surface first; ORG before PERSON.

    Letter boundaries rather than \\b: BERTopic joins topic-Name tokens with "_",
    which \\b treats as a word character."""
    out = []
    for path, tag in ((_ORGS, "[ORG]"), (_PERSONS, "[PERSON]")):
        if not path.exists():
            continue
        seen = {" ".join(l.split()) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        for m in sorted(seen, key=len, reverse=True):
            body = r"[\s_]*".join(map(re.escape, m.split()))
            out.append((re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE), tag))
    return out


PATTERNS = _patterns()


def scrub(text: str) -> str:
    if not text:
        return text
    for p, tag in PATTERNS:
        text = p.sub(tag, text)
    return text


def scrub_csv(path, columns) -> int:
    path = Path(path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return 0
    n = 0
    for r in rows:
        for c in columns:
            if c in r and r[c]:
                s = scrub(r[c])
                if s != r[c]:
                    r[c] = s; n += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n


if __name__ == "__main__":
    cols = ["justification", "cap_keywords", "jrn_keywords"]
    for p in sys.argv[1:]:
        print(f"{p}: {scrub_csv(p, cols)} field(s) scrubbed")
